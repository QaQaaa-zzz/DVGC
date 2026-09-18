import pytest
from jit_dvgc import current_policy_iteration as iteration


def test_full_nominal_rollout_is_explicitly_zero_residual_only():
    from jit_dvgc.pulse_exploration import pulse_delay
    spec=dict(nominal_source_rollout=True,pulse_start_schedule=[0],pulse_steps=400,
              horizon=400,num_envs=1,delta_limit=[0.,0.,0.,0.])
    assert pulse_delay(spec,0)==0
    with pytest.raises(ValueError):pulse_delay({**spec,'delta_limit':[.15]*4},0)
    with pytest.raises(ValueError):pulse_delay({**spec,'nominal_source_rollout':False},0)


def test_initial_bank_rejects_historical_helpers():
    with pytest.raises(ValueError):
        iteration.validate_initial_bank({'members':[{'name':'pi_0'}, {'name':'pi_1'}]}, 'pi_0')
    iteration.validate_initial_bank({'members':[{'name':'pi_0'}]}, 'pi_0')


def test_retention_panel_uses_available_distinct_phase_entries():
    support={'entries':[dict(key=str(i),phase=phase) for phase,n in [('upstream',3),('downstream',1)] for i in range(n)]}
    rows=iteration.retention_candidates(support,{'snapshot':'start'},8)
    assert len(rows)==5
    assert len({(r['phase'],r['key']) for r in rows[1:]})==4


def test_reuse_rejects_scientific_drift_but_allows_new_code_locks():
    iteration.verify_stage_reuse({'seed':1,'source_locks':{'old':'hash'}},
                                 {'seed':1,'source_locks':{'new':'hash'},'resume_stage_root':'old'})
    with pytest.raises(ValueError):iteration.verify_stage_reuse({'seed':1},{'seed':2})


def test_reuse_allows_new_stop_horizon_and_resource_gate_only():
    old={'rounds':150,'maximum_interactions':999999,'gate':{'kind':'user_requested_immediate'},'horizon':400,'seed':5}
    new={**old,'rounds':100,'maximum_interactions':666666,'gate':{'kind':'gpu_idle','minimum_free_mib':20000}}
    iteration.verify_stage_reuse(old,new)
    with pytest.raises(ValueError):iteration.verify_stage_reuse(old,{**new,'horizon':200})


def test_promotion_requires_new_success_and_old_retention_and_start():
    def row(a,b):
        return dict(attempts=[dict(policy='old',label=a),dict(policy='new',label=b)])
    rows=[row(1,1),row(1,1),row(1,0)]
    assert not iteration.promotion_decision(rows,'old','new',1,.9)['promote']
    assert iteration.promotion_decision(rows,'old','new',1,.5)['promote']
    assert not iteration.promotion_decision([row(1,0),row(1,1)],'old','new',1,.5)['promote']
    assert not iteration.promotion_decision([row(1,1),row(1,1)],'old','new',0,.5)['promote']
    assert not iteration.promotion_decision([row(1,1),row(1,None)],'old','new',1,.5)['promote']


def test_current_budget_reserves_reseeding_and_uses_one_evaluator():
    from jit_dvgc.pulse_exploration import budget_contract
    spec=dict(rounds=12,num_envs=128,horizon=400,pulse_steps=3,policy_steps=128000,
              pulse_start_schedule=[5,10,15,20,25,0],iteration_mode='current_policy_only_v1',
              retention_samples_per_phase=8,minimum_retention=.9)
    budget=budget_contract(spec,1)
    assert budget['bank_suffixes']==12*128*400
    assert budget['nominal_support']==13*400*401
    assert budget['retention_evaluation']==12*2*17*400
    assert budget['maximum_interactions']==sum(v for k,v in budget.items() if k!='maximum_interactions')


def test_failed_nominal_candidate_is_a_negative_result_not_pipeline_error(tmp_path,monkeypatch):
    import numpy as np
    from jit_dvgc import pulse_exploration_runtime as runtime
    from jit_dvgc.jump_evidence_validation import write,read
    def collect(spec,out):
        out.mkdir(parents=True)
        np.savez(out/'prefixes.npz',prefix_mask=np.ones((2,1),bool),
            **{'snap/down/recovery_success':np.zeros((2,1),bool),'physical_failure':np.array([[False],[True]])})
        write(out/'status.json',{'charged_interactions':400})
    monkeypatch.setattr(runtime,'collect',collect)
    monkeypatch.setattr(runtime,'networks',lambda spec:(None,None,None,None,None,None))
    spec={'horizon':400,'success_criterion':'stable_forward_recovery','allow_nominal_failure':True}
    iteration.seed_support(spec,tmp_path/'candidate')
    result=read(tmp_path/'candidate/status.json')
    assert result['phase']=='completed' and result['support_ready'] is False
    assert result['charged_interactions']==400
    assert not (tmp_path/'candidate/support.json').exists()
    with pytest.raises(ValueError,match='current source'):
        iteration.seed_support({**spec,'allow_nominal_failure':False},tmp_path/'initial')

def test_stage_reuse_allows_relocated_code_but_not_changed_sampling():
    iteration.verify_stage_reuse({'repo':'/old/code','num_envs':1024},
                                 {'repo':'/new/code','num_envs':1024})
    with pytest.raises(ValueError):
        iteration.verify_stage_reuse({'repo':'/old/code','num_envs':1024},
                                     {'repo':'/new/code','num_envs':128})


def test_nested_training_config_resolves_original_bootstrap(tmp_path):
    import json
    from jit_dvgc.iterative_probe_training import resolve_bootstrap_config
    original=tmp_path/'original.json';original.write_text(json.dumps({'schema':'jit_unified_formal_v1'}))
    derived=tmp_path/'derived.json';derived.write_text(json.dumps({'bootstrap_formal_config':'original.json'}))
    successor=tmp_path/'successor.json';successor.write_text(json.dumps({'bootstrap_formal_config':str(derived)}))
    assert resolve_bootstrap_config(successor)==original.resolve()
    original.write_text(json.dumps({'bootstrap_formal_config':str(successor)}))
    with pytest.raises(ValueError,match='cycle'):resolve_bootstrap_config(successor)

def test_relocated_neighborhood_map_reuses_verified_content(tmp_path):
    import json,hashlib
    from jit_dvgc.current_policy_iteration import verify_stage_reuse
    data=json.dumps({'source_actor_sha256':'source','rows':[]}).encode();sha=hashlib.sha256(data).hexdigest()
    a=tmp_path/'a.json';b=tmp_path/'b.json';a.write_bytes(data);b.write_bytes(data)
    verify_stage_reuse({'neighborhood_map':str(a),'neighborhood_map_sha256':sha},
                       {'neighborhood_map':str(b),'neighborhood_map_sha256':sha})
    b.write_text('{}')
    import pytest
    with pytest.raises(ValueError,match='map'):
        verify_stage_reuse({'neighborhood_map':str(a),'neighborhood_map_sha256':sha},
                           {'neighborhood_map':str(b),'neighborhood_map_sha256':sha})


def test_successor_name_avoids_inherited_and_local_names():
    from jit_dvgc.current_policy_iteration import successor_name
    bank={'members':[{'name':'lineage_repair_0010'},{'name':'lineage_repair_0010_new_0001'}]}
    assert successor_name('lineage',10,bank)=='lineage_repair_0010_new_0002'
    assert successor_name('lineage',9,bank)=='lineage_repair_0009'


def test_code_commit_is_provenance_not_scientific_reuse_identity():
    from jit_dvgc.current_policy_iteration import verify_stage_reuse
    verify_stage_reuse({'code_commit':'old','seed':1},{'code_commit':'fixed','seed':1})
    import pytest
    with pytest.raises(ValueError):
        verify_stage_reuse({'code_commit':'old','seed':1},{'code_commit':'fixed','seed':2})


def test_recovery_resolves_ancestor_artifacts_and_rejects_cycles(tmp_path):
    from jit_dvgc.jump_evidence_validation import write
    original=tmp_path/'original'; original.mkdir()
    resumed=tmp_path/'resumed'; resumed.mkdir()
    write(original/'outcomes.json', [{'label':1}])
    write(resumed/'recovery.json', {'previous':str(original)})
    assert iteration.lineage_artifact(resumed,'outcomes.json') == original/'outcomes.json'
    write(resumed/'outcomes.json',[{'label':0}])
    assert iteration.lineage_artifact(resumed,'outcomes.json') == resumed/'outcomes.json'
    write(original/'recovery.json', {'previous':str(resumed)})
    with pytest.raises(ValueError,match='cycle'):
        iteration.lineage_artifact(resumed,'missing.json')


def test_recovery_keeps_incomplete_stage_maximum_charged():
    status={'inherited_interactions':10,'charged_interactions':40,'costs':[
        {'accounting':'actual','charged_interactions':10,'maximum_interactions':20},
        {'accounting':'reserved_after_incomplete_child','charged_interactions':20,'maximum_interactions':20}]}
    assert iteration.recovery_charge(status) == 40
    status['costs'][1]['charged_interactions']=19
    with pytest.raises(ValueError): iteration.recovery_charge(status)
