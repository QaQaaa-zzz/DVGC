"""Post-learning repair must consume the already serialized student exactly once."""
import json
from pathlib import Path
import pytest
from jit_dvgc.generative_bridge.protocol import atomic_json
from jit_dvgc.generative_bridge.contracts import digest,file_sha
from .test_teacher_recovery_storage import completed_fixture


def failed_fixture(tmp_path,monkeypatch):
    root,bundle=completed_fixture(tmp_path,monkeypatch);rd=root/'round_0001'
    (rd/'current_source.json').unlink()
    training=rd/'training';training.mkdir()
    atomic_json(training/'status.json',dict(status='completed'))
    atomic_json(training/'formal_report.json',dict(status='completed',completed_training_transitions=32000,train_panel_interactions=0))
    atomic_json(training/'checkpoint_action_probes.json',{'0':{},'16000':{},'32000':{}})
    bank=rd/'bank.json';atomic_json(bank,{})
    student=dict(bundle['actor'],learner=bundle['learner'],training=str(training),bank=str(bank))
    phase=dict(support_path=str(rd/'support.json'),panel_support_path=str(rd/'seed.json'),
        demo_manifest={'path':str(rd/'demo.json'),'sha256':'later'},retention_ref={})
    for filename in ('support.json','seed.json','inventory.json'):atomic_json(rd/filename,{})
    tape=rd/'demo.npz';tape.write_bytes(b'demo');atomic_json(rd/'demo.json',dict(path=str(tape),sha256=file_sha(tape)))
    phase['demo_manifest']['sha256']=file_sha(rd/'demo.json');atomic_json(rd/'source_phase_result.json',phase)
    spec=dict(round_index=1,source={'actor_sha256':'pi0'},continuation={'generator':{'checkpoint_manifest_sha256':'G'}},
        budgets={'max_physics':3500000},student_reset_inventory=str(rd/'inventory.json'))
    atomic_json(rd/'production.json',spec);(rd/'stages').mkdir()
    atomic_json(rd/'stages/round_contract.json',{'sha256':digest(spec),'contract':spec})
    atomic_json(rd/'stages/train_student.json',dict(input_sha256=digest({'round':digest(spec),'inputs':phase}),
        output_sha256=digest(student),result=student))
    worker=rd/'teacher_worker';worker.mkdir();proposal=worker/'proposals.npz';proposal.write_bytes(b'proposal')
    roots={str(i):dict(root_id=str(i),source_actor_sha256='pi0',generator=spec['continuation']['generator'],
        teacher_status='searched_no_solution',proposal_path=str(proposal),proposal_sha256=file_sha(proposal)) for i in range(128)}
    atomic_json(worker/'results.json',roots);atomic_json(worker/'request.json',{})
    atomic_json(worker/'completed.json',dict(phase='completed',roots=128,results_sha256=file_sha(worker/'results.json'),
        request_sha256=file_sha(worker/'request.json'),charged_interactions=10))
    atomic_json(rd/'panels.json',{'new_roots':[{'root_id':str(i)} for i in range(128)]})
    atomic_json(rd/'costs.json',[dict(stage='student_student',phase='completed',charged_interactions=32000),
        dict(stage='teacher_worker',phase='completed',charged_interactions=10,worker_receipt_sha256=file_sha(worker/'completed.json')),
        dict(stage='student_train_batch_0000',phase='failed',charged_interactions=12800)])
    atomic_json(rd/'status.json',dict(phase='failed',error="RuntimeError('student_train_batch_0000: failed')",charged_interactions=44810))
    return root,student,phase


def test_evaluation_boundary_retains_completed_student_and_failed_upper(tmp_path,monkeypatch):
    from jit_dvgc.generative_bridge.recovery_campaign import verify_evaluation_boundary
    root,student,phase=failed_fixture(tmp_path,monkeypatch)
    proof=verify_evaluation_boundary(root)
    assert proof['student']==student and proof['completed_training_transitions']==32000
    assert proof['extra_actor_updates']==0 and proof['charged_interactions']==44810
    assert proof['teacher_roots']==128 and proof['phase']['path']==str(root/'round_0001/source_phase_result.json')
    atomic_json(Path(student['training'])/'status.json',{'status':'running'})
    with pytest.raises(ValueError,match='completed'):verify_evaluation_boundary(root)


def test_completed_phase_reuse_requires_original_input_digest_and_inventory(tmp_path,monkeypatch):
    from jit_dvgc.generative_bridge.recovery_campaign import completed_student_phase,verify_evaluation_boundary
    root,student,phase=failed_fixture(tmp_path,monkeypatch);proof=verify_evaluation_boundary(root)
    runner=type('Runner',(),{'spec':{'recovery_completed_student':proof},'root':tmp_path/'new'})()
    runner.root.mkdir()
    assert completed_student_phase(runner)==phase
    assert runner.spec['student_reset_inventory']==str(root/'round_0001/inventory.json')
    p=Path(proof['phase']['path']);atomic_json(p,{**phase,'support_path':'different'})
    with pytest.raises(ValueError,match='changed'):completed_student_phase(runner)


def test_inherited_student_journal_never_calls_training(tmp_path,monkeypatch):
    from jit_dvgc.generative_bridge.recovery_campaign import verify_evaluation_boundary
    from jit_dvgc.generative_bridge.recovery import RecoveryJournal
    root,student,phase=failed_fixture(tmp_path,monkeypatch);proof=verify_evaluation_boundary(root)
    contract=json.loads((root/'round_0001/stages/round_contract.json').read_text())
    journal=RecoveryJournal(tmp_path/'successor/stages',{'recovery':dict(previous_contract_identity=contract['sha256'],
        stages={'train_student':proof['student_stage']})})
    def forbidden():raise AssertionError('completed student must never train again')
    assert journal.stage('train_student',phase,forbidden)==student


def test_recovered_checkpoint_probes_skip_original_writer_and_reject_drift(tmp_path,monkeypatch):
    from jit_dvgc.generative_bridge.recovery_campaign import verify_evaluation_boundary,run_round
    from jit_dvgc.generative_bridge.closed_loop import ClosedLoopRound
    root,student,phase=failed_fixture(tmp_path,monkeypatch);proof=verify_evaluation_boundary(root)
    old=Path(student['training'])/'checkpoint_action_probes.json';original=old.read_bytes()
    assert proof['checkpoint_probes']==dict(path=str(old),sha256=file_sha(old))
    assert proof['locks'][str(old)]==file_sha(old)
    successor=tmp_path/'probe_reuse';successor.mkdir();spec_path=successor/'production.json'
    atomic_json(spec_path,dict(output=str(successor),recovery_collection={},recovery_completed_student=proof))
    def initialize(self,spec):self.spec=spec;self.root=Path(spec['output'])
    def forbidden(*args,**kwargs):raise AssertionError('original probe writer must not run')
    monkeypatch.setattr(ClosedLoopRound,'__init__',initialize)
    monkeypatch.setattr(ClosedLoopRound,'checkpoint_probes',forbidden)
    monkeypatch.setattr(ClosedLoopRound,'run',lambda self:self.checkpoint_probes(student,phase,None))
    assert run_round(spec_path)==old
    assert old.read_bytes()==original
    assert json.loads((successor/'checkpoint_probe_reuse.json').read_text())['source']==proof['checkpoint_probes']
    atomic_json(old,{'tampered':True})
    with pytest.raises(ValueError,match='checkpoint probes changed'):run_round(spec_path)


def test_sibling_prepare_spec_carries_all_fees_without_student_update(tmp_path,monkeypatch):
    from jit_dvgc.generative_bridge.recovery_campaign import prepare_evaluation_recovery,resume_round_spec
    import jit_dvgc.generative_bridge.production as production
    root,student,phase=failed_fixture(tmp_path,monkeypatch);rd=root/'round_0001'
    collection=rd/'collection';collection.mkdir();atomic_json(collection/'status.json',{'phase':'completed','charged_interactions':2})
    atomic_json(rd/'collection_spec.json',{});collection_ref=dict(output=str(collection),previous_round=str(rd),
        files={str(collection/'status.json'):file_sha(collection/'status.json')},charged_interactions=2,spec_path=str(rd/'collection_spec.json'))
    spec=json.loads((rd/'production.json').read_text());spec.update(series_id='original-series',locks={},
        recovery_collection=collection_ref)
    atomic_json(rd/'production.json',spec);contract=digest(spec)
    atomic_json(rd/'stages/round_contract.json',{'sha256':contract,'contract':spec})
    stage=json.loads((rd/'stages/train_student.json').read_text());stage['input_sha256']=digest({'round':contract,'inputs':phase})
    atomic_json(rd/'stages/train_student.json',stage)
    for i in range(4):
        path=rd/f'source_{i}.json';atomic_json(path,[]);result=dict(path=str(path),sha256=file_sha(path))
        atomic_json(rd/'stages'/f'eval_source_suffix_batch_{i:04d}.json',dict(result=result,output_sha256=digest(result)))
    atomic_json(root/'status.json',dict(phase='failed',completed_rounds=0));atomic_json(root/'started.json',{})
    atomic_json(rd/'started.json',{'started_unix':123})
    monkeypatch.setattr(production,'implementation_identity',lambda repo:'newcommit')
    monkeypatch.setattr(production,'implementation_files',lambda repo:{})
    successor=tmp_path/'sibling';d=prepare_evaluation_recovery(root,successor,tmp_path/'repo')
    info=d['engineering_recovery'];assert info['completed_student']['student']==student
    assert info['cumulative_round_upper']==44810+166400
    assert d['seed']==8113001 and d['series_id']=='original-series'
    recovered=resume_round_spec(d)
    assert recovered['recovery']['stages']['train_student']==info['completed_student']['student_stage']
    assert recovered['recovery_completed_student']['inventory']['path']==str(rd/'inventory.json')
    assert sum(c['charged_interactions'] for c in json.loads((successor/'round_0001/costs.json').read_text()))==44810
    assert not (successor/'round_0001/students').exists()
    assert json.loads((rd/'stages/train_student.json').read_text())['result']==student
    # A later explicit evaluation-only repair must keep the original stage identity.
    next_rd=successor/'round_0001';next_contract=digest(recovered)
    (next_rd/'stages').mkdir()
    atomic_json(next_rd/'stages/round_contract.json',{'sha256':next_contract,'contract':recovered})
    atomic_json(next_rd/'stages/train_student.json',dict(input_sha256=digest({'round':next_contract,'inputs':phase}),
        output_sha256=digest(student),result=student))
    costs=json.loads((next_rd/'costs.json').read_text())
    costs.append(dict(stage='student_train_batch_0000',phase='failed',charged_interactions=12800))
    atomic_json(next_rd/'costs.json',costs)
    atomic_json(next_rd/'status.json',dict(phase='failed',error='student_train_batch_0000: failed',charged_interactions=57610))
    atomic_json(next_rd/'started.json',{'started_unix':123});atomic_json(successor/'started.json',{})
    atomic_json(successor/'status.json',dict(phase='failed',completed_rounds=0))
    third=prepare_evaluation_recovery(successor,tmp_path/'third',tmp_path/'repo')
    chain=third['engineering_recovery']
    assert chain['completed_student']['charged_interactions']==57610
    assert chain['completed_student']['failed_evaluation_upper']==25600
    assert chain['previous_contract_identity']==contract
    assert chain['stages']==info['stages']
    assert chain['completed_student']['student']==student
    assert sum(c['charged_interactions'] for c in chain['carried_round_costs'])==57610
    costs.append(dict(stage='explorer_update',phase='completed',charged_interactions=0))
    atomic_json(next_rd/'costs.json',costs)
    with pytest.raises(ValueError,match='only failed student evaluation'):
        prepare_evaluation_recovery(successor,tmp_path/'fourth',tmp_path/'repo')
