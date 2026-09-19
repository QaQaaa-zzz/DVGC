import numpy as np
import pytest
from jit_dvgc.pulse_exploration import (pulse_feedback, budget_contract,
    round_delta_limit, reserve_interactions, InteractionBudgetExhausted)


def test_fixed_evaluator_failures_train_but_unknown_and_missing_stage_do_not():
    rows=[dict(cell=str(i),label=v,learning_attempted=False) for i,v in enumerate([1,0,None])]
    rows.append(dict(cell='missing',label=None,learning_attempted=False,stage_reached=False))
    rewards,mask,cells,parts=pulse_feedback(rows,[],dict(novelty=.25,success=1.,failure=1.),quality_mode='current_policy')
    np.testing.assert_allclose(rewards,[1.25,-.75,0,0])
    assert mask.tolist()==[True,True,False,False]
    assert 'missing' not in cells
    assert pulse_feedback(rows,[],dict(novelty=.25,success=1.,failure=1.))[1].tolist()==[True,False,False,False]


def test_current_quality_uses_pre_training_label_and_novelty_only_removes_quality():
    rows=[dict(cell='x',label=1,initial_label=0,learning_attempted=True)]
    weights=dict(novelty=.25,success=1.,failure=1.)
    assert pulse_feedback(rows,[],weights,quality_mode='current_policy')[0][0]==-.75
    assert pulse_feedback(rows,[],weights,quality_mode='delayed')[0][0]==1.25
    assert pulse_feedback(rows,[],weights,quality_mode='novelty_only')[0][0]==.25


def test_fixed_policy_budget_excludes_supplementation_and_uses_event_wait_bound():
    spec=dict(rounds=24,num_envs=128,horizon=400,pulse_steps=3,policy_steps=128000,
              iteration_mode='current_policy_only_v1',retention_samples_per_phase=8,
              minimum_retention=.95,training_mode='fixed_policy',
              pulse_event_schedule=['start','liftoff','apex','descent'])
    b=budget_contract(spec,1)
    assert b['prefixes']==24*128*400
    assert b['learning_and_reevaluation']==0 and b['retention_evaluation']==0
    assert b['nominal_support']==400+200*400
    assert b['maximum_interactions']==sum(v for k,v in b.items() if k!='maximum_interactions')


def test_actual_cost_reservation_never_exceeds_cap():
    reserve_interactions(10,20,30)
    with pytest.raises(InteractionBudgetExhausted):reserve_interactions(11,20,30)
    with pytest.raises(ValueError):reserve_interactions(0,-1,30)


def test_amplitude_schedule_covers_cartesian_cycle_and_rejects_nonfinite():
    s={'delta_limit':[.1]*4,'delta_limit_schedule':[[.1]*4]*4+[[.15]*4]*4}
    assert [round_delta_limit(s,i)[0] for i in range(8)]==[.1]*4+[.15]*4
    with pytest.raises(ValueError):round_delta_limit({'delta_limit':[float('nan')]*4},0)


def test_empty_successful_set_is_completed_evidence_not_runtime_failure(tmp_path):
    import json
    from jit_dvgc.pulse_exploration import plot_completed_rounds
    plot_completed_rounds(tmp_path,[{'label':None},{'label':0}])
    assert json.loads((tmp_path/'analysis/tube_xz/status.json').read_text())['phase']=='completed'
