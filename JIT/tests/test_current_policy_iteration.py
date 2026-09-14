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
