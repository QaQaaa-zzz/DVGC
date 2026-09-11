import numpy as np
from jit_dvgc.exploration_training import episode_advantages

def test_terminal_credit_propagates_but_padding_does_not():
    reward=np.array([[0.],[2.],[900.]])
    value=np.zeros_like(reward);mask=np.array([[1.],[1.],[0.]])
    adv,ret=episode_advantages(reward,value,mask,gamma=.5,lam=1.)
    np.testing.assert_allclose(ret[:,0],[1.,2.,0.])
    np.testing.assert_allclose(adv,ret)

def test_complete_episode_does_not_bootstrap_value_beyond_terminal():
    adv,ret=episode_advantages(np.array([[3.],[0.]]),np.array([[1.],[999.]]),np.array([[1.],[0.]]),gamma=.99,lam=.95)
    assert adv[0,0]==2 and ret[0,0]==3
