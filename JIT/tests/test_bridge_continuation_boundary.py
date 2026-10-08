import json
from pathlib import Path
import pytest
from jit_dvgc.generative_bridge.contracts import digest, file_sha


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {'path': str(path), 'sha256': file_sha(path)}


def parent(tmp_path, monkeypatch):
    from jit_dvgc.generative_bridge import continuation_boundary as m
    root=tmp_path/'parent'; rd=root/'round_0002'
    policy={'actor_sha256':'actor-C','normalizer_sha256':'normalizer-C','critic_sha256':'critic-C'}
    actor={'frozen_policy':str(root/'frozen.json'),'policy':policy,**policy}
    write(root/'frozen.json', {'policy':policy})
    monkeypatch.setattr(m, '_validate_actor', lambda actor: policy)
    monkeypatch.setattr(m, '_validate_semantics', lambda bundle, spec: None)
    g=write(root/'g/manifest.json', {'state_sha256':write(root/'g/state.msgpack', {})['sha256']})
    bundle={'round':2,'actor':actor,'evaluated_student':{'actor_sha256':'rejected-student'},
        'generator':{'checkpoint_manifest':g['path'],'checkpoint_manifest_sha256':g['sha256']},
        'corpus':write(root/'corpus.json',{}),'demo':write(root/'demo.json',{}),
        'explorer':str(root/'e.msgpack'),'explorer_sha256':write(root/'e.msgpack',{})['sha256'],
        'support':str(root/'support.json'),'tail_lineage':[],
        'dev_fixture':str(root/'dev.npz'),'dev_fixture_sha256':write(root/'dev.npz',{})['sha256']}
    write(root/'support.json',{})
    spec={'round_index':2,'continuation':{'generator':bundle['generator']}}
    write(rd/'production.json',spec);write(rd/'stages/round_contract.json',{'contract':spec,'sha256':digest(spec)})
    phase={};write(rd/'source_phase_result.json',phase)
    for name, inputs, result in [('train_student',phase,bundle['evaluated_student']),
        ('generator_selection',{'corpus':bundle['corpus'],'incumbent':bundle['generator']},bundle['generator'])]:
        write(rd/f'stages/{name}.json',{'input_sha256':digest({'round':digest(spec),'inputs':inputs}),
            'output_sha256':digest(result),'result':result})
    write(rd/'source_rows.json',{'rows':[{}]})
    for family in ('source_suffix','student_train'):
        result=write(rd/f'{family}.json',[{}])
        write(rd/f'evaluations/{family}/candidates.json',[{}])
        write(rd/f'evaluations/{family}/spec.json',{'bank':'bank','proposer':'actor'})
        inputs={'rows':[{}],'bank':'bank','actor':'actor','prefixes_sha256':None}
        write(rd/f'stages/eval_{family}.json',{'result':result,'output_sha256':digest(result),
            'input_sha256':digest({'round':digest(spec),'inputs':inputs})})
    for child in ('collection','explorer_update'):write(rd/child/'status.json',{'phase':'completed'})
    write(rd/'status.json',{'phase':'completed'});write(rd/'current_source.json',bundle)
    write(root/'current_source.json',bundle);write(root/'completed_rounds.json',[str(rd)])
    write(root/'plan.json',{'rounds':1,'round_offset':1,'prior_physics_charged':50})
    write(root/'status.json',{'phase':'completed','completed_rounds':1,'lifetime_physics_charged':80})
    write(rd/'costs.json',[{'charged_interactions':30,'charged_updates':2}])
    return m, root, bundle


def test_completed_published_actor_preserved(tmp_path,monkeypatch):
    m,root,bundle=parent(tmp_path,monkeypatch)
    result=m.resolve_completed_boundary(root)
    assert result['bundle']==bundle
    assert result['prior_physics_charged']==80
    assert result['migration']['legacy_optimizer_bootstrap'] is True


def test_stopped_empty_parent_follows_explicit_link_keeps_cost(tmp_path,monkeypatch):
    m,root,bundle=parent(tmp_path,monkeypatch); child=tmp_path/'stopped'
    write(child/'status.json',{'phase':'cancelled','completed_rounds':0})
    write(child/'plan.json',{'rounds':2,'continuation_parent':str(root),'prior_physics_charged':80})
    write(child/'round_0003/costs.json',[{'charged_interactions':12,'charged_updates':7}])
    with pytest.raises(ValueError):m.resolve_completed_boundary(child)
    result=m.resolve_completed_boundary(child,True)
    assert result['bundle']==bundle
    assert result['prior_physics_charged']==92
    assert result['prior_optimization_costs']['supervised_updates']==9


@pytest.mark.parametrize('damage',['running','partial','hash','stage'])
def test_reject_bad_boundary(tmp_path,monkeypatch,damage):
    m,root,bundle=parent(tmp_path,monkeypatch)
    if damage=='running':write(root/'status.json',{'phase':'running','completed_rounds':1})
    elif damage=='partial':write(root/'round_0002/status.json',{'phase':'failed'})
    elif damage=='hash':write(root/'e.msgpack',{'changed':True})
    else:write(root/'round_0002/stages/train_student.json',{})
    with pytest.raises((ValueError,KeyError)):m.resolve_completed_boundary(root,True)


def test_cancelled_status_cannot_hide_live_pid(tmp_path,monkeypatch):
    import subprocess,sys
    m,root,_=parent(tmp_path,monkeypatch)
    proc=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)',str(root),'run_bridge_closed_loop.py'])
    try:
        write(root/'launch.json',{'pid':proc.pid})
        with pytest.raises(ValueError,match='still running'):m.resolve_completed_boundary(root,True)
    finally:
        proc.terminate();proc.wait()


def test_learner_state_hash_is_checked(tmp_path,monkeypatch):
    m,root,bundle=parent(tmp_path,monkeypatch)
    state=write(root/'learner/state.bin',{})
    metadata={**bundle['actor']['policy'],'state_path':state['path'],'state_sha256':state['sha256'],'local_transitions':128000}
    bundle['learner']=write(root/'learner/manifest.json',metadata)
    write(root/'current_source.json',bundle);write(root/'round_0002/current_source.json',bundle)
    result=m.resolve_completed_boundary(root)
    assert result['migration']['legacy_optimizer_bootstrap'] is False
    write(root/'learner/state.bin',{'modified':True})
    with pytest.raises(ValueError,match='learner payload changed'):m.resolve_completed_boundary(root)


def test_no_second_legacy_bootstrap(tmp_path,monkeypatch):
    m,root,_=parent(tmp_path,monkeypatch)
    write(root/'plan.json',{'rounds':1,'round_offset':1,'prior_physics_charged':50,'continuous_learning':True})
    with pytest.raises(ValueError,match='bootstrap'):m.resolve_completed_boundary(root)


def test_prepare_from_parent_is_not_a_live_training_worker(tmp_path,monkeypatch):
    import subprocess,sys
    m,root,_=parent(tmp_path,monkeypatch)
    proc=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)',
        'run_bridge_closed_loop.py','--continue-from',str(root),'--output',str(tmp_path/'new')])
    try:
        assert m.resolve_completed_boundary(root)['bundle']['actor']['actor_sha256']=='actor-C'
    finally:
        proc.terminate();proc.wait()


def test_removed_required_evaluation_is_partial(tmp_path,monkeypatch):
    m,root,_=parent(tmp_path,monkeypatch)
    (root/'round_0002/stages/eval_student_train.json').unlink()
    with pytest.raises(ValueError,match='missing required evaluation'):m.resolve_completed_boundary(root)


def test_profile_generator_stage_identity_continues(tmp_path,monkeypatch):
    m,root,bundle=parent(tmp_path,monkeypatch);rd=root/'round_0002'
    spec=json.loads((rd/'production.json').read_text());spec['generator_update_policy']='last_valid'
    write(rd/'production.json',spec);write(rd/'stages/round_contract.json',{'contract':spec,'sha256':digest(spec)})
    inputs_by_name={'train_student':{},'generator_selection':{'corpus':bundle['corpus'],
        'incumbent':bundle['generator'],'generator_update_policy':'last_valid'}}
    for family in ('source_suffix','student_train'):
        inputs_by_name['eval_'+family]={'rows':[{}],'bank':'bank','actor':'actor','prefixes_sha256':None}
    for name,inputs in inputs_by_name.items():
        path=rd/f'stages/{name}.json';record=json.loads(path.read_text())
        record['input_sha256']=digest({'round':digest(spec),'inputs':inputs});write(path,record)
    assert m.resolve_completed_boundary(root)['bundle']==bundle


def test_orphan_ppo_child_blocks_stopped_parent(tmp_path,monkeypatch):
    import subprocess,sys
    m,root,_=parent(tmp_path,monkeypatch)
    script=tmp_path/'train_unified_from_pi0.py';script.write_text('import time; time.sleep(30)')
    proc=subprocess.Popen([sys.executable,str(script),'--config',str(root/'round_0003/config.json')])
    try:
        with pytest.raises(ValueError,match='worker is still running'):m.resolve_completed_boundary(root,True)
    finally:
        proc.terminate();proc.wait()
