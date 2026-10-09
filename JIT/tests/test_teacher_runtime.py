import numpy as np
from jit_dvgc.generative_bridge.teacher_runtime import TeacherEvaluationSession,kernel_identity


def test_dynamic_actions_reused_kernel_and_aba():
    import jax.numpy as jp
    s=TeacherEvaluationSession()
    f=lambda state,actions:state+actions
    a=jp.array([1.,2.]);b=jp.array([3.,4.]);z=jp.zeros(2)
    first=s.execute('fixed',f,z,a)
    s.execute('fixed',f,z,b)
    np.testing.assert_array_equal(s.execute('fixed',f,z,a),first)
    assert s.compile_count==1
    s.close();assert not s.kernels


def test_rng_and_actor_changes_do_not_reuse_kernel():
    spec=dict(horizon=400);policy=dict(actor_sha256='a',normalizer_sha256='n')
    key=kernel_identity(spec,policy,17,True,None,True)
    assert key!=kernel_identity(spec,{**policy,'actor_sha256':'b'},17,True,None,True)
    assert key!=kernel_identity({**spec,'suffix_rng_indices':list(reversed(range(17)))},policy,17,True,None,True)
