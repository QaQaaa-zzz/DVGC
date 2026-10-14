import numpy as np
import pytest
from jit_dvgc.recovery_curriculum import remaining_horizon, curriculum_probabilities, validate_teacher_entries


def test_remaining_budget_has_no_extra_episode():
    assert remaining_horizon(400, 3, 16) == 381
    with pytest.raises(ValueError): remaining_horizon(400, 400, 0)


def test_root_equal_and_weights_dynamic():
    rows=[dict(root_id='a',stage=16),dict(root_id='a',stage=16),dict(root_id='b',stage=16),dict(root_id='a',stage=0),dict(root_id='b',stage=0)]
    p=curriculum_probabilities(rows,[1,0,0,0,0])
    np.testing.assert_allclose(p,[.25,.25,.5,0,0])
    q=curriculum_probabilities(rows,[0,0,0,0,1])
    np.testing.assert_allclose(q,[0,0,0,.5,.5])


def test_reject_dev_and_unverified():
    good=dict(root_id='a',role='train',inherited_split='generator_train',complete=True,full_success=True,teacher_status='verified_solution',teacher_repeat_labels=[1,1,1])
    assert validate_teacher_entries({'entries':[good]}) == [good]
    for change in ({'role':'DEV'},{'inherited_split':'generator_dev'},{'full_success':False},{'teacher_repeat_labels':[1,0,1]}):
        with pytest.raises(ValueError): validate_teacher_entries({'entries':[{**good,**change}]})


def _mock_reset():
    import jax
    import jax.numpy as jp
    from flax import struct
    from jit_dvgc.recovery_curriculum import RecoveryCurriculumReset
    @struct.dataclass
    class State:
        data: object
        obs: object
        reward: object
        done: object
        metrics: object
        info: object
    reset=RecoveryCurriculumReset.__new__(RecoveryCurriculumReset)
    reset.entries=[dict(stage=16,root_id='a',control_step=398,remaining_horizon=2),dict(stage=0,root_id='b',control_step=397,remaining_horizon=3)]
    reset.weights=jp.array([1.,0,0,0,0])
    reset.stage_basis=jp.array([[1.,0],[0,0],[0,0],[0,0],[0,1.]])
    flags=('terminated','truncated','success','physical_failure','roll_limit','pitch_limit','jump_zone_missed','stuck','yaw_limit','timeout')
    info={k:jp.array([False,False]) for k in flags}
    info.update(rng=jax.random.split(jax.random.key(42),2),recovery_reset_rng=jax.random.split(jax.random.key(13),2),expert_switching_used=jp.array([False,True]),episode_step=jp.array([398,397]),
                recovery_curriculum_weights=jp.tile(reset.weights,(2,1)),recovery_curriculum_index=jp.array([0,1]),
                recovery_curriculum_stage=jp.array([16,0]),recovery_remaining_horizon=jp.array([2,3]),
                time_out=jp.zeros(2),end_code=jp.zeros(2,jp.int32))
    reset.pool=State(data=jp.array([7.,9.]),obs=jp.array([[1.],[2.]]),reward=jp.zeros(2),done=jp.zeros(2),metrics={'x':jp.zeros(2)},info=info)
    def step(s,a):
        return s.replace(data=s.data+1,reward=jp.array(1.),done=jp.array(0.),info={**s.info,'episode_step':s.info['episode_step']+1})
    reset.original_step=step
    class Env:
        observation_size=1
        action_size=1
        @property
        def unwrapped(self): return self
    reset.env=Env()
    reset.install()
    return reset


def test_jitted_dynamic_weights_and_terminal_reset_remaining_horizon():
    import jax
    import jax.numpy as jp
    r=_mock_reset()
    sample=jax.jit(r.reset_with_weights)
    s=sample(jax.random.key(1),jp.array([1.,0,0,0,0]))
    assert int(s.info['recovery_curriculum_stage'])==16
    s=r.set_state_weights(s,jp.array([0.,0,0,0,1.]))
    step=jax.jit(r.step)
    s=step(s,jp.zeros(1)); assert not bool(s.done)
    s=step(s,jp.zeros(1)); assert bool(s.done)
    assert int(s.info['recovery_curriculum_stage'])==0
    assert int(s.info['episode_step'])==397
    assert int(s.info['recovery_remaining_horizon'])==3
    s=step(s.replace(done=jp.array(0.)),jp.zeros(1)); assert not bool(s.done)
    assert int(s.info['recovery_remaining_horizon'])==2


def test_training_wrapper_preserves_absolute_budget_and_next_episode():
    import jax
    import jax.numpy as jp
    from jit_dvgc.recovery_curriculum import wrap_recovery_for_training
    r=_mock_reset(); env=wrap_recovery_for_training(r.env)
    s=jax.jit(env.reset)(jax.random.split(jax.random.key(0),2))
    step=jax.jit(env.step)
    s=step(s,jp.zeros((2,1)))
    np.testing.assert_array_equal(s.info['steps'],[399,399])
    s=step(s,jp.zeros((2,1)))
    np.testing.assert_array_equal(s.done,[1,1])
    np.testing.assert_array_equal(s.info['steps'],[400,400])
    s=step(s,jp.zeros((2,1)))
    np.testing.assert_array_equal(s.done,[0,0])
    np.testing.assert_array_equal(s.info['steps'],[399,399])
    np.testing.assert_array_equal(s.info['episode_metrics']['length'],[1,1])


def test_versioned_snapshot_preserves_time_expert_and_context(tmp_path):
    from jit_dvgc.unified_envelope_snapshot import (UnifiedEnvelopeSnapshot,
        UP_EVENT_FIELDS,DOWN_EVENT_FIELDS,save_unified_envelope_snapshot,
        load_unified_envelope_snapshot,snapshot_context_sha256)
    import dataclasses,json
    values={f.name:0 for f in dataclasses.fields(UnifiedEnvelopeSnapshot)}
    for name in ('qpos','qvel','ctrl','last_action'): values[name]=np.zeros(4,np.float32)
    values.update(observation_fifo=np.arange(75,dtype=np.float32).reshape(3,25),observation=np.arange(76,dtype=np.float32),
                  rng=np.array([42,0],np.uint32),up_events={k:np.array(0) for k in UP_EVENT_FIELDS},
                  down_events={k:np.array(0) for k in DOWN_EVENT_FIELDS},parent_trajectory='actual_trace.npz',
                  compatibility_identity={'original':True},snapshot_time=.38,expert_switching_used=True)
    for name in ('parent_state_sha256','config_sha256','xml_sha256','policy_actor_sha256','policy_payload_sha256'): values[name]='a'*64
    s=UnifiedEnvelopeSnapshot(**values);save_unified_envelope_snapshot(tmp_path/'snapshot',s)
    restored=load_unified_envelope_snapshot(tmp_path/'snapshot')
    assert restored.snapshot_time==.38 and restored.expert_switching_used is True
    assert snapshot_context_sha256(s)==snapshot_context_sha256(restored)
    np.testing.assert_array_equal(restored.observation_fifo,s.observation_fifo)
    original=snapshot_context_sha256(s);s.snapshot_time=.4
    assert snapshot_context_sha256(s)!=original
    sidecar=json.loads((tmp_path/'snapshot'/'identity.json').read_text());sidecar['expert_switching_used']=False
    (tmp_path/'snapshot'/'identity.json').write_text(json.dumps(sidecar))
    with pytest.raises(ValueError,match='provenance mismatch'):load_unified_envelope_snapshot(tmp_path/'snapshot')
