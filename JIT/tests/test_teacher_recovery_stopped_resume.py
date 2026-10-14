"""Stopped post-learning continuation keeps the completed campaign prefix."""
import json
from pathlib import Path
import pytest
from .test_teacher_recovery_evaluation_resume import failed_fixture
from jit_dvgc.generative_bridge.protocol import atomic_json
from jit_dvgc.generative_bridge.contracts import digest,file_sha


def stopped_fixture(tmp_path,monkeypatch):
    root,student,phase=failed_fixture(tmp_path,monkeypatch);old=root/'round_0001'
    old.rename(root/'round_0012_payload');old.symlink_to(root/'round_0012_payload',target_is_directory=True)
    rd=root/'round_0012';rd.symlink_to(root/'round_0012_payload',target_is_directory=True)
    mpath=Path(student['learner']['path']);m=json.loads(mpath.read_text());m['lifetime_transitions']=384000
    atomic_json(mpath,m);student['learner']['sha256']=file_sha(mpath)
    previous=dict(round=11,actor=student,learner=student['learner'],generator={'checkpoint_manifest_sha256':'G'})
    journal=[]
    for i in range(1,12):
        directory=root/f'prefix{i}';directory.mkdir();manifest={**m,'lifetime_transitions':i*32000}
        atomic_json(directory/'learner.json',manifest);receipt={'path':str(directory/'learner.json'),'sha256':file_sha(directory/'learner.json')}
        source={**previous,'round':i,'learner':receipt};atomic_json(directory/'current_source.json',source)
        atomic_json(directory/'costs.json',[dict(stage='all',phase='completed',charged_interactions=100)])
        atomic_json(directory/'status.json',dict(phase='completed',training_transitions=32000,charged_interactions=100))
        if i==1:old.unlink()
        (root/f'round_{i:04d}').symlink_to(directory,target_is_directory=True)
        journal.append(dict(round=i,learner=receipt,charged_interactions=100))
        if i==11:previous=source
    # Student references used the old round1 path; retain that immutable alias separately.
    payload=root/'round_0012_payload'
    def remap(value):
        if isinstance(value,dict):return {k:remap(v) for k,v in value.items()}
        if isinstance(value,list):return [remap(v) for v in value]
        if isinstance(value,str):return value.replace(str(old),str(payload))
        return value
    student=remap(student);phase=remap(phase)
    demo=Path(phase['demo_manifest']['path']);atomic_json(demo,remap(json.loads(demo.read_text())))
    phase['demo_manifest']['sha256']=file_sha(demo)
    mpath=Path(student['learner']['path']);manifest=remap(m);atomic_json(mpath,manifest);student['learner']['sha256']=file_sha(mpath)
    spec=dict(round_index=12,source={'actor_sha256':'pi0'},continuation=previous,budgets={'max_physics':3500000,'max_wall_seconds':20000},
        student_reset_inventory=str(payload/'inventory.json'),series_id='same-campaign',locks={})
    atomic_json(payload/'production.json',spec);atomic_json(payload/'source_phase_result.json',phase)
    atomic_json(payload/'stages/round_contract.json',{'sha256':digest(spec),'contract':spec})
    atomic_json(payload/'stages/train_student.json',dict(input_sha256=digest({'round':digest(spec),'inputs':phase}),output_sha256=digest(student),result=student))
    worker=payload/'teacher_worker';teachers=json.loads((worker/'results.json').read_text());teachers=remap(teachers)
    for row in teachers.values():row['generator']=previous['generator']
    atomic_json(worker/'results.json',teachers);receipt=json.loads((worker/'completed.json').read_text());receipt['results_sha256']=file_sha(worker/'results.json');atomic_json(worker/'completed.json',receipt)
    costs=[dict(stage='teacher_worker',phase='completed',charged_interactions=10,worker_receipt_sha256=file_sha(worker/'completed.json')),
        dict(stage='student_student',phase='completed',charged_interactions=32000)]
    for role,count in [('source_suffix',4),('student_train',3)]:
        for i in range(count):
            path=payload/f'{role}_{i}.json';atomic_json(path,[]);result={'path':str(path),'sha256':file_sha(path)}
            atomic_json(payload/'stages'/f'eval_{role}_batch_{i:04d}.json',dict(result=result,output_sha256=digest(result)))
            if role=='student_train':costs.append(dict(stage=f'student_train_batch_{i:04d}',phase='completed',charged_interactions=20))
    costs.append(dict(stage='student_train_batch_0003',phase='stopped',charged_interactions=12800))
    charge=sum(c['charged_interactions'] for c in costs);atomic_json(payload/'costs.json',costs)
    atomic_json(payload/'status.json',dict(phase='stopped',stage='student_train_batch_0003',training_transitions=32000,charged_interactions=charge,wall_seconds=100,updated_unix=1200))
    atomic_json(payload/'started.json',dict(started_unix=1000))
    collection=payload/'collection';collection.mkdir();atomic_json(collection/'status.json',dict(phase='completed',charged_interactions=0))
    atomic_json(payload/'collection_spec.json',{});(collection/'prefixes.npz').write_bytes(b'prefix');(collection/'update_state.msgpack').write_bytes(b'state')
    atomic_json(root/'completed_rounds.json',journal);atomic_json(root/'current_source.json',previous)
    atomic_json(root/'status.json',dict(phase='stopped',current_round=12,completed_rounds=11,completed_student_transitions=384000))
    atomic_json(root/'started.json',{})
    for i in (0,5,10):
        out=root/f'teacher_monitor/G{i}';out.mkdir(parents=True);atomic_json(out/'status.json',dict(phase='completed',round=i,charged_interactions=7))
        atomic_json(out/'report.json',dict(phase='completed',round=i))
    declaration=json.loads((root/'declaration.json').read_text());declaration['monitor_initial']=str(root/'teacher_monitor/G0/report.json');atomic_json(root/'declaration.json',declaration)
    return root,student,journal,charge


def test_stopped_round12_preserves_lifetime_and_completed_three_batches(tmp_path,monkeypatch):
    from jit_dvgc.generative_bridge.recovery_campaign import verify_evaluation_boundary
    root,student,_,charge=stopped_fixture(tmp_path,monkeypatch)
    proof=verify_evaluation_boundary(root,round_index=12,stopped=True)
    assert proof['student']==student and proof['lifetime_training_transitions']==384000
    assert proof['charged_interactions']==charge and proof['failed_evaluation_upper']==12800
    assert proof['extra_actor_updates']==0
    p=Path(student['learner']['path']);m=json.loads(p.read_text());m['lifetime_transitions']=32000;atomic_json(p,m)
    with pytest.raises(ValueError):verify_evaluation_boundary(root,round_index=12,stopped=True)


def test_prepare_stopped_cursor12_keeps_prefix_and_pending_reservation(tmp_path,monkeypatch):
    from jit_dvgc.generative_bridge.recovery_campaign import prepare_stopped_resume,campaign_start_state,resume_round_spec
    import jit_dvgc.generative_bridge.production as production
    root,student,journal,charge=stopped_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(production,'implementation_identity',lambda repo:'canonical-commit')
    monkeypatch.setattr(production,'implementation_files',lambda repo:{})
    output=tmp_path/'successor';d=prepare_stopped_resume(root,output,tmp_path/'repo',port=6039)
    source,done,next_round=campaign_start_state(d,output)
    assert next_round==12 and done==journal and source['round']==11
    assert (output/'round_0011').is_symlink() and not (output/'round_0012').exists()
    info=d['engineering_recovery'];assert info['round_index']==12 and info['completed_student']['student']==student
    assert len(info['stages'])==8 and 'eval_student_train_batch_0003' not in info['stages']
    assert info['maximum_additional_round_interactions']==128000
    assert sum(c['charged_interactions'] for c in info['carried_round_costs'])==charge
    spec=resume_round_spec(d)
    assert spec['round_index']==12 and spec['output']==str(output/'round_0012')
    assert spec['recovery_active_wall_seconds']==200
    assert [r['round'] for r in d['campaign_resume']['teacher_monitors']]==[0,5,10]
    # Repeated explicit stops at the same pending stage retain both reservations.
    rd=output/'round_0012';(rd/'stages').mkdir()
    atomic_json(rd/'stages/round_contract.json',dict(contract=spec,sha256=digest(spec)))
    phase=json.loads(Path(info['completed_student']['phase']['path']).read_text())
    atomic_json(rd/'stages/train_student.json',dict(input_sha256=digest({'round':digest(spec),'inputs':phase}),output_sha256=digest(student),result=student))
    for name,ref in info['stages'].items():
        if name!='train_student':atomic_json(rd/'stages'/f'{name}.json',json.loads(Path(ref['path']).read_text()))
    costs=json.loads((rd/'costs.json').read_text());costs.append(dict(stage='student_train_batch_0003',phase='stopped',charged_interactions=12800))
    atomic_json(rd/'costs.json',costs);atomic_json(rd/'status.json',dict(phase='stopped',stage='student_train_batch_0003',training_transitions=32000,
        charged_interactions=charge+12800,wall_seconds=250,updated_unix=1400))
    atomic_json(rd/'started.json',dict(started_unix=1000));atomic_json(output/'started.json',{})
    atomic_json(output/'status.json',dict(phase='stopped',current_round=12,completed_rounds=11))
    # Collection reuse remains pinned to the original round's immutable files.
    chain=prepare_stopped_resume(output,tmp_path/'third',tmp_path/'repo')
    assert chain['engineering_recovery']['completed_student']['failed_evaluation_upper']==25600
    assert sum(c['charged_interactions'] for c in chain['engineering_recovery']['carried_round_costs'])==charge+12800
    assert chain['campaign_resume']['next_round']==12
