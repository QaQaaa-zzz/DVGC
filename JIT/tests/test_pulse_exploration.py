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
