import numpy as np
import pytest

def test_fixed_request_loader_requires_locked_identity_and_one_pulse(tmp_path):
    from jit_dvgc.retention_repair import load_requests, sha
    p=tmp_path/'r.npz';a=np.zeros((20,2,4),np.float32);a[5:8,0]=.25;a[10:13,1]=-.2
    np.savez(p,requested=a,onsets=np.array([5,10]))
    spec={'frozen_request_table':{'path':str(p),'sha256':sha(p)},'num_envs':2,'horizon':20,'pulse_steps':3,'frozen_explorer_evaluation':True,'controller_mode':'fixed_random'}
    out,onsets=load_requests(spec);np.testing.assert_array_equal(out,a);assert onsets.tolist()==[5,10]
    a[1,0]=.1;np.savez(p,requested=a,onsets=np.array([5,10]));spec['frozen_request_table']['sha256']=sha(p)
    with pytest.raises(ValueError,match='window'):load_requests(spec)

def test_relative_labels_keep_unknown_and_retention_separate():
    from jit_dvgc.retention_repair import relative_label
    assert relative_label(True,False,True)=='retention_debt'
    assert relative_label(False,True,True)=='baseline_failure_teacher_recoverable_learned'
    assert relative_label(None,True,True)=='unknown'
    assert relative_label(False,False,True)=='baseline_failure_teacher_recoverable_pending'

def test_only_d0_can_execute_and_padding_is_charged():
    from jit_dvgc.retention_repair import validate_budget
    p={'stage':'D1','batches':[{'capacity':250,'horizon':400}],'budget':{'physical_transitions':1500000,'wall_seconds':14400},'training_transitions':0}
    with pytest.raises(ValueError):validate_budget(p)
    p['stage']='D0';assert validate_budget(p)==100000
    p['budget']['physical_transitions']=99999
    with pytest.raises(ValueError):validate_budget(p)

def test_actor_normalized_export_uses_full_normalizer_observation_tree():
    import jax.numpy as jp
    from brax.training.acme import running_statistics
    from jit_dvgc.retention_repair_report import normalized_actor_observations
    stats=running_statistics.init_state({'state':jp.zeros(76),'privileged_state':jp.zeros(106)})
    assert normalized_actor_observations(stats,np.zeros((2,76)),np.zeros((2,106))).shape==(2,76)

def test_liftoff_diagnostic_does_not_call_initial_suspension_a_jump():
    from jit_dvgc.retention_repair_report import liftoff_tick
    assert liftoff_tick(np.array([.03,.03,0,0,.02,.03,.04]),np.array([.03,.03,0,0,.02,.03,.04]))==4
    assert liftoff_tick(np.array([.03]*8),np.array([.03]*8)) is None
