import numpy as np
import pytest

def test_symmetric_initialization_has_no_state_or_channel_direction_bias():
 from jit_dvgc.rsl_pulse import initialize,infer
 import jax.numpy as jp
 spec=dict(seed=113,learning_rate=.001,explorer_initialization={'mode':'symmetric','latent_std':.8})
 state=initialize(spec,np.zeros(106),np.ones(106))
 obs=jp.asarray(np.random.default_rng(9).normal(size=(9,106)),dtype=jp.float32)
 mu,sd,_=infer(state,obs)
 np.testing.assert_allclose(mu,0,atol=1e-7)
 np.testing.assert_allclose(sd,.8,atol=1e-6)

def test_invalid_initial_std_rejected():
 from jit_dvgc.rsl_pulse import initialize
 with pytest.raises(ValueError,match='initial'):
  initialize(dict(seed=4,learning_rate=.001,explorer_initialization={'mode':'symmetric','latent_std':0}),np.zeros(106),np.ones(106))
