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


def test_unverified_persistent_worker_cannot_be_adopted():
    import pytest
    from jit_dvgc.generative_bridge.teacher_worker import validate_teacher_acceptance
    with pytest.raises(ValueError,match='physical acceptance'):
        validate_teacher_acceptance({'implementation_commit':'example'})


def test_explicit_trial_is_bound_to_one_output_round_and_code(tmp_path):
    import json,pytest
    from jit_dvgc.generative_bridge.teacher_worker import validate_teacher_acceptance
    from jit_dvgc.generative_bridge.contracts import file_sha
    report=tmp_path/'trial.json';out=tmp_path/'round'
    report.write_text(json.dumps(dict(schema='jit_b1_experimental_trial_v1',implementation_commit='test',output=str(out),round_index=31,user_authorized=True,formal_acceptance=False,max_physics=1500000,known_limitations=['prior label mismatch','physical trajectories not identical'])))
    spec=dict(implementation_commit='test',output=str(out),round_index=31,budgets={'max_physics':1500000},teacher_experimental_trial={'path':str(report),'sha256':file_sha(report)})
    validate_teacher_acceptance(spec)
    for field,value in [('output',str(tmp_path/'other')),('round_index',32),('implementation_commit','other')]:
        with pytest.raises(ValueError,match='trial'):
            validate_teacher_acceptance({**spec,field:value})
    report.write_text('{}')
    with pytest.raises(ValueError,match='trial'):validate_teacher_acceptance(spec)


def test_trial_multiround_rejected_before_source_loading():
    import pytest
    from jit_dvgc.generative_bridge.continuation_profile import inspect_plan
    with pytest.raises(ValueError,match='exactly one round'):
        inspect_plan(dict(teacher_experimental_trial={'path':'unused'},rounds=2,automatic_extension=False))
