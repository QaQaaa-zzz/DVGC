import numpy as np
from dual_stress_protocol import compose

def test_existing_residual_mapping_saturation():
 base=np.array([[.95,-.95,0.,.1]],np.float32)
 latent=np.array([[20.,-20.,0.,1.]],np.float32)
 action,request,effective=compose(base,latent)
 np.testing.assert_allclose(request,np.tanh(latent)*.25,atol=1e-7)
 np.testing.assert_allclose(action,np.clip(base+request,-1,1),atol=1e-7)
 np.testing.assert_allclose(effective,action-base,atol=1e-7)
 assert abs(request).max()<=.25

def test_frozen_existing_explorer_identity():
 from dual_stress_protocol import load_protocol
 spec,state,index=load_protocol()
 assert len(state['normalizer_mean'])==386
 assert spec['pulse_steps']==3 and spec['delta_limit']==[.25]*4
 assert state['backend']=='rsl_rl_3.2.0'
