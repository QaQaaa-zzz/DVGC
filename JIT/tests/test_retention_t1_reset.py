"""FourPoolReset through the actual PPO wrappers, with zero-physics states."""
import jax
import jax.numpy as jp
import numpy as np
from flax import struct
from mujoco_playground._src import mjx_env

from jit_dvgc.continuation.device_rollout import stack_worlds
from jit_dvgc.descent_semantics import DescentEventState
from jit_dvgc.observation import HistoryState
from jit_dvgc.ppo import wrap_for_jit_training
from jit_dvgc.retention_reset import FourPoolReset
from jit_dvgc.semantics import EventState


@struct.dataclass
class SavedData:
    qpos: object
    qvel: object
    time: object


def saved_state(group):
    integer = lambda x: jp.asarray(x, jp.int32)
    scalar = lambda x: jp.asarray(x, jp.float32)
    flag = jp.asarray(group >= 2)
    return mjx_env.State(
        data=SavedData(jp.full(7, group + 10.), jp.full(6, group + 11.), scalar(group + .125)),
        obs={'state': jp.full(76, group + 20.), 'privileged_state': jp.full(106, group + 30.)},
        reward=scalar(0), done=scalar(0), metrics={'retention/reset_group': scalar(group)},
        info={'rng': jax.random.key(900 + group),
              'history': HistoryState(jp.full((3, 25), group + 40.), integer(3)),
              'up_events': EventState(flag, flag, flag, flag, flag, flag, scalar(group + 50.),
                                     integer(group + 51), jp.asarray(False), integer(group + 70)),
              'down_events': DescentEventState(flag, flag, scalar(group + 52.), integer(group + 90), jp.asarray(False)),
              'episode_step': integer(group + 60), 'phase_episode_step': integer(group + 80),
              'active_phase': integer(group % 2), 'last_action': jp.full(4, group + 100.),
              'retention_reset_group': integer(group), 'retention_reset_index': integer(group)})


class NoPhysicsEnv:
    def step(self, state, action):
        # Deliberately make the just-ended context distinct from every pool.
        info = {k: v for k, v in state.info.items()
                if k not in ('retention_reset_group', 'retention_reset_index')}
        info.update(history=state.info['history'].replace(frames=state.info['history'].frames + 1000),
                    episode_step=state.info['episode_step'] + 1,
                    phase_episode_step=state.info['phase_episode_step'] + 1,
                    last_action=action,
                    up_events=state.info['up_events'].replace(episode_step=state.info['up_events'].episode_step + 1),
                    down_events=state.info['down_events'].replace(post_contact_ticks=state.info['down_events'].post_contact_ticks + 1))
        return state.replace(data=state.data.replace(time=state.data.time + .02, qpos=state.data.qpos + 1000),
                             obs=jax.tree.map(lambda x: x + 1000, state.obs), info=info,
                             reward=jp.asarray(7.), done=(action[0] > .5).astype(jp.float32))


def test_full_ppo_reset_restores_selected_snapshot_context_and_keeps_lane_rng():
    base = NoPhysicsEnv()
    resetter = FourPoolReset.__new__(FourPoolReset)
    resetter.env = base
    resetter.original_step = base.step
    resetter.starts = jp.arange(4)
    resetter.lengths = jp.ones(4, jp.int32)
    resetter.pool = stack_worlds([saved_state(i) for i in range(4)])
    wrapped = wrap_for_jit_training(resetter.install(), episode_length=400)
    state = jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(17), 64))
    original_group = int(state.info['retention_reset_group'][0])
    original_time = float(state.data.time[0])
    actions = jp.ones((64, 4)).at[0].set(0)
    result = jax.jit(wrapped.step)(state, actions)
    groups = np.asarray(result.info['retention_reset_group'])
    assert set(groups[1:]) == {0, 1, 2, 3}
    np.testing.assert_array_equal(result.done, [0.] + [1.] * 63)
    np.testing.assert_array_equal(result.info['episode_done'], result.done)
    np.testing.assert_array_equal(result.info['AutoResetWrapper_done_count'], [0] + [1] * 63)
    assert int(groups[0]) == original_group
    np.testing.assert_allclose(result.data.time[0], original_time + .02, atol=1e-6)
    np.testing.assert_allclose(result.obs['state'][0], original_group + 1020.)
    # Each terminal lane must use the complete newly selected pool context.
    for lane, group in enumerate(groups[1:], 1):
        np.testing.assert_array_equal(result.obs['state'][lane], group + 20.)
        np.testing.assert_array_equal(result.obs['privileged_state'][lane], group + 30.)
        np.testing.assert_array_equal(result.data.qpos[lane], group + 10.)
        np.testing.assert_array_equal(result.data.qvel[lane], group + 11.)
        assert float(result.data.time[lane]) == group + .125
        np.testing.assert_array_equal(result.info['history'].frames[lane], group + 40.)
        assert int(result.info['history'].valid_count[lane]) == 3
        assert int(result.info['episode_step'][lane]) == group + 60
        assert int(result.info['phase_episode_step'][lane]) == group + 80
        assert int(result.info['up_events'].episode_step[lane]) == group + 70
        assert int(result.info['down_events'].post_contact_ticks[lane]) == group + 90
        np.testing.assert_array_equal(result.info['last_action'][lane], group + 100.)
        assert int(result.info['retention_reset_index'][lane]) == group
        actual_rng = np.asarray(jax.random.key_data(result.info['rng'][lane]))
        saved_rng = np.asarray(jax.random.key_data(jax.random.key(900 + int(group))))
        if group >= 2:
            np.testing.assert_array_equal(actual_rng, saved_rng)
        else:
            assert not np.array_equal(actual_rng, saved_rng)


def test_t1_capacity_guard_survives_terminal_reset():
    import pytest
    from jit_dvgc.retention_t1_runtime import wrap_t1_for_training
    class OverflowEnv(NoPhysicsEnv):
        def step(self,state,action):
            result=super().step(state,action)
            return result.replace(metrics={**result.metrics,'T1/capacity_exceeded':jp.asarray(1.)})
    base=OverflowEnv()
    resetter=FourPoolReset.__new__(FourPoolReset)
    resetter.env=base;resetter.original_step=base.step
    resetter.starts=jp.arange(4);resetter.lengths=jp.ones(4,jp.int32)
    resetter.pool=stack_worlds([saved_state(i).replace(metrics={**saved_state(i).metrics,'T1/capacity_exceeded':jp.asarray(0.)}) for i in range(4)])
    wrapped=wrap_t1_for_training(resetter.install())
    initial=jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(17),4))
    with pytest.raises(Exception,match='scratch saturation'):
        result=jax.jit(wrapped.step)(initial,jp.ones((4,4)))
        jax.block_until_ready(result)
