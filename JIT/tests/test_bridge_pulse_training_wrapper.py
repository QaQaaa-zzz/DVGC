import jax
import jax.numpy as jp
import numpy as np
from flax import struct

from jit_dvgc.iterative_probe_training import initialize_training_action_pulse, apply_training_action_pulse
from jit_dvgc.ppo import wrap_for_jit_training


@struct.dataclass
class State:
    data: object
    obs: object
    reward: object
    done: object
    metrics: object
    info: object


class FakeEnv:
    def __init__(self, slots, full_start=True):
        self.full_start = full_start
        self._training_action_pulse = dict(pulse_scope='full_task_start_only', onsets=[0],
            steps=3, probability=1., delta_limit=[.25]*4,
            logical_episode_rng=dict(master_seed=3, role='student_ppo', round=2, slot_ids=slots))

    def reset(self, rng):
        info = initialize_training_action_pulse(dict(reset_from_jump_start=jp.asarray(self.full_start)),
                    rng, self._training_action_pulse, full_task_start=self.full_start)
        return State(jp.zeros(1), {'state':jp.zeros(1)}, jp.asarray(0.), jp.asarray(0.), {}, info)

    def step(self, state, action):
        _, info, _ = apply_training_action_pulse(state.info, action, self._training_action_pulse)
        return state.replace(info=info, done=(action[0] > .5).astype(jp.float32))


def test_training_reset_identity_is_slot_counter_and_only_done_lanes_advance():
    env = wrap_for_jit_training(FakeEnv([19, 2**32-1]), episode_length=100)
    state = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(10), 2))
    np.testing.assert_array_equal(state.info['training_episode_id_low'], [19, 2**32-1])
    np.testing.assert_array_equal(state.info['training_episode_id_high'], [0,0])
    initial_draws = np.asarray(state.info['training_pulse_draws'])
    step = jax.jit(env.step)
    state = step(state, jp.asarray([[1.,0,0,0],[0.,0,0,0]]))
    np.testing.assert_array_equal(state.info['training_episode_id_high'], [1,0])
    assert not np.array_equal(initial_draws[0], state.info['training_pulse_draws'][0])
    np.testing.assert_array_equal(initial_draws[1], state.info['training_pulse_draws'][1])
    assert int(state.info['training_pulse_tick'][0]) == 0
    state = step(state, jp.ones((2,4)))
    np.testing.assert_array_equal(state.info['training_episode_id_high'], [2,1])
    assert len({np.asarray(k).tobytes() for k in state.info['training_pulse_key']}) == 2


def test_training_slot_requests_are_partition_invariant_and_snapshots_disabled():
    def reset(slots, full=True):
        env = wrap_for_jit_training(FakeEnv(slots, full), episode_length=100)
        return env, jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(90),len(slots)))
    _, whole = reset([2,7,13])
    _, first = reset([2]); _, rest = reset([7,13])
    np.testing.assert_array_equal(whole.info['training_pulse_draws'],
        np.concatenate([first.info['training_pulse_draws'],rest.info['training_pulse_draws']]))
    env, snapshot = reset([2], False)
    for _ in range(3):
        snapshot = jax.jit(env.step)(snapshot, jp.ones((1,4)))
        assert not np.asarray(snapshot.info['training_pulse_enabled']).any()
        np.testing.assert_array_equal(snapshot.info['training_pulse_request'], 0.)


def test_ppo_key_matches_declared_lineage_and_rejects_duplicate_slots():
    import pytest
    from jit_dvgc.generative_bridge.pulse_protocol import logical_episode_key
    env = wrap_for_jit_training(FakeEnv([7]), episode_length=100)
    state = env.reset(jax.random.split(jax.random.PRNGKey(0),1))
    state = jax.jit(env.step)(state, jp.ones((1,4)))
    expected = logical_episode_key(3, 'student_ppo', 2, (1 << 32) | 7, 'pulse')
    np.testing.assert_array_equal(state.info['training_pulse_key'][0], expected)
    assert not np.array_equal(expected, logical_episode_key(3, 'train', 2, (1 << 32) | 7, 'pulse'))
    with pytest.raises(ValueError, match='duplicate'):
        wrap_for_jit_training(FakeEnv([7,7]), episode_length=100)
    env = wrap_for_jit_training(FakeEnv([7,8]), episode_length=100)
    with pytest.raises(ValueError, match='slot_ids'):
        env.reset(jax.random.split(jax.random.PRNGKey(0),1))


def test_brax_evaluator_uses_distinct_eight_lane_identity_without_mutating_training():
    from jit_dvgc.ppo import logical_pulse_evaluation_env
    original=FakeEnv(list(range(128)))
    evaluation=logical_pulse_evaluation_env(original,8)
    train=wrap_for_jit_training(original,episode_length=100)
    evaluate=wrap_for_jit_training(evaluation,episode_length=100)
    training_state=jax.jit(train.reset)(jax.random.split(jax.random.PRNGKey(0),128))
    eval_state=jax.jit(evaluate.reset)(jax.random.split(jax.random.PRNGKey(1),8))
    assert training_state.done.shape==(128,) and eval_state.done.shape==(8,)
    assert original._training_action_pulse['logical_episode_rng']['slot_ids']==list(range(128))
    assert original._training_action_pulse['logical_episode_rng']['role']=='student_ppo'
    assert not set(map(tuple,np.asarray(training_state.info['training_pulse_key']))) & set(map(tuple,np.asarray(eval_state.info['training_pulse_key'])))
