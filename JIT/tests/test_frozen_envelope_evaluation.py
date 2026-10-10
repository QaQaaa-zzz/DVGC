import numpy as np
import pytest
from jit_dvgc.pulse_schedule import collection_steps

def test_frozen_explorer_can_continue_after_pulse():
 s=dict(full_episode_rollout=True,controller_mode='learned_residual',frozen_explorer_evaluation=True,horizon=400,pulse_steps=3)
 assert collection_steps(s,np.array([0,15]),None)==400

def test_training_explorer_still_cannot_use_full_evaluation():
 with pytest.raises(ValueError):collection_steps(dict(full_episode_rollout=True,controller_mode='learned_residual',horizon=400),np.array([0]),None)

def test_shared_bank_shape_and_quaternion_validation(tmp_path):
 from jit_dvgc.evaluation_initial_state import load_initial_state_bank
 p=tmp_path/'bank.npz';q=np.zeros((2,8));q[:,3]=1;v=np.zeros((2,7));np.savez(p,qpos=q,qvel=v)
 a,b=load_initial_state_bank(p,count=2,nq=8,nv=7,root_qpos=0);np.testing.assert_array_equal(a,q)
 with pytest.raises(ValueError):load_initial_state_bank(p,count=3,nq=8,nv=7,root_qpos=0)
 q[1,3]=0;np.savez(p,qpos=q,qvel=v)
 with pytest.raises(ValueError):load_initial_state_bank(p,count=2,nq=8,nv=7,root_qpos=0)

def test_shared_bank_rejects_nonfinite(tmp_path):
 from jit_dvgc.evaluation_initial_state import load_initial_state_bank
 p=tmp_path/'bank.npz';q=np.zeros((1,7));q[:,3]=1;v=np.zeros((1,6));v[0,0]=np.nan;np.savez(p,qpos=q,qvel=v)
 with pytest.raises(ValueError):load_initial_state_bank(p,count=1,nq=7,nv=6,root_qpos=0)


def test_frozen_evaluation_refuses_optimizer_update(tmp_path):
 from jit_dvgc.rsl_pulse import update_runtime
 with pytest.raises(ValueError,match="cannot update"):
  update_runtime(dict(frozen_explorer_evaluation=True),tmp_path/'uncreated')
 assert not (tmp_path/'uncreated').exists()
