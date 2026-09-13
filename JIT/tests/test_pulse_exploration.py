import numpy as np
from jit_dvgc.pulse_exploration import pulse_feedback, budget_contract


def test_failure_only_penalized_after_learning_and_unknown_excluded():
    rows=[dict(cell='a',label=0,learning_attempted=False),dict(cell='b',label=0,learning_attempted=True),dict(cell='c',label=1,learning_attempted=False)]
    rewards,mask,ledger,parts=pulse_feedback(rows,[],dict(novelty=.25,success=1.,failure=1.))
    assert mask.tolist()==[False,True,True]
    np.testing.assert_allclose(rewards,[0,-.75,1.25])
    assert set(ledger)=={'a','b','c'}


def test_duplicate_cells_share_novelty_but_keep_context_quality():
    rows=[dict(cell='a',label=1,learning_attempted=False),dict(cell='a',label=0,learning_attempted=True)]
    r,_,_,_=pulse_feedback(rows,[],dict(novelty=.25,success=1.,failure=1.))
    np.testing.assert_allclose(r,[1.125,-.875])
    r,_,_,_=pulse_feedback(rows,['a'],dict(novelty=.25,success=1.,failure=1.))
    np.testing.assert_allclose(r,[1,-1])


def test_budget_includes_growing_bank_new_policy_and_panels():
    s=dict(rounds=2,num_envs=4,pulse_steps=3,horizon=400,policy_steps=128000)
    b=budget_contract(s,7)
    assert b['maximum_interactions']==7*400+2*4*3+(7+8)*4*400+2*(128000+1600+4*400)


def test_conflict_remains_unknown_but_another_policy_can_witness():
    from jit_dvgc.pulse_exploration_runtime import suffix_label,aggregate_labels
    label,_=suffix_label(True,True,False,True,False)
    assert label is None
    attempts=[dict(policy='pi_0',label=label),dict(policy='pi_1',label=0)]
    assert aggregate_labels(attempts,['pi_0','pi_1']) is None
    attempts[1]['label']=1
    assert aggregate_labels(attempts,['pi_0','pi_1'])==1
    assert aggregate_labels([dict(policy='pi_0',label=0)],['pi_0','pi_1']) is None


def test_unknown_after_learning_never_receives_failure_penalty():
    rewards,mask,ledger,_=pulse_feedback([dict(cell='u',label=None,learning_attempted=True)],[],dict(novelty=.25,success=1.,failure=1.))
    assert rewards.tolist()==[0.] and mask.tolist()==[False] and ledger==['u']


def test_delayed_pulses_charge_unperturbed_prefix_and_cycle_locations():
    from jit_dvgc.pulse_exploration import pulse_delay
    s=dict(rounds=4,num_envs=8,pulse_steps=3,horizon=400,policy_steps=128000,pulse_start_schedule=[0,10])
    assert [pulse_delay(s,i) for i in range(4)]==[0,10,0,10]
    assert budget_contract(s,7)['prefixes']==8*(3+13+3+13)


def test_terminal_prefix_quality_preserves_failure_success_and_conflict():
    from jit_dvgc.pulse_exploration_runtime import terminal_prefix_label
    assert terminal_prefix_label(False,True)[0]==0
    assert terminal_prefix_label(True,False)[0]==1
    assert terminal_prefix_label(True,True)[0] is None
    assert terminal_prefix_label(True,None)[0] is None
    assert terminal_prefix_label(False,None)[0]==0
    rows=[dict(cell=str(i),label=label,learning_attempted=False,prefix_terminal=True)
          for i,label in enumerate([0,1,None])]
    r,mask,ledger,_=pulse_feedback(rows,[],dict(novelty=.25,success=1.,failure=1.))
    np.testing.assert_allclose(r,[-.75,1.25,0]);assert mask.tolist()==[True,True,False]
    assert len(ledger)==3
