"""CPU checks for bounded generator pretraining with a real scalar optimizer."""
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jit_dvgc.generative_bridge import diffusion
from jit_dvgc.generative_bridge.feedback_data import realized_mix


def _state():
    return diffusion.create_train_state(
        {'w': jnp.array(0.1)}, jax.random.PRNGKey(7),
        {'mean': jnp.zeros(76), 'std': jnp.ones(76)})


def _trace():
    n = 17
    obs = np.zeros((n + 1, 76), np.float32)
    return {'metadata': {'root_episode_id': 'real_train_ancestor'}, 'arrays': {
        'actor_observation_before': obs[:-1], 'actor_observation_after': obs[1:],
        'normalized_action_executed': np.full((n, 4), 0.2, np.float32),
        'valid_mask': np.ones(n, bool), 'done': np.arange(n) == n - 1,
        'success': np.arange(n) == n - 1, 'failure': np.zeros(n, bool),
        'timeout': np.zeros(n, bool), 'phase_before': np.zeros(n, int),
        'action_origin': np.array(['actor_only'] * n)}}


def _corpus():
    groups = {'history': [_trace()], 'teacher_new': [], 'actor_new': []}
    return {'groups': groups, 'requested_mix': {'history': .5, 'teacher_new': .25, 'actor_new': .25},
            'realized_mix': realized_mix({k: len(v) for k, v in groups.items()}), 'new_data': False}


def _fixture():
    return (np.zeros((2, 76), np.float32), np.full((2, 16, 4), .2, np.float32),
            np.array([0, 99], np.int32), np.zeros((2, 16, 4), np.float32))


def _predict(params, x, obs, k):
    return jnp.ones_like(x) * params['w']


def test_pretrain_warmup_and_cosine_reach_floor_without_resetting_adam():
    schedule = diffusion.pretrain_learning_rate(max_updates=6, warmup_updates=2,
                                                 learning_rate=1e-4, min_learning_rate=1e-5)
    assert [schedule(i) for i in (0, 1, 2, 4, 5)] == pytest.approx(
        [5e-5, 1e-4, 1e-4, 3.25e-5, 1e-5])
    step = diffusion.make_train_step(_predict, learning_rate=schedule)
    state = _state()
    obs = jnp.zeros((2, 76)); actions = jnp.full((2, 16, 4), .2)
    first, _ = step(state, obs, actions)
    second, _ = step(first, obs, actions)
    assert int(first['optimizer'][1].count) == 1
    assert int(second['optimizer'][1].count) == 2
    assert float(first['params']['w']) != float(state['params']['w'])
    assert float(second['params']['w']) != float(first['params']['w'])


def test_pretrain_selects_fixed_dev_checkpoint_and_restores_full_state(tmp_path):
    selected, report = diffusion.train_pretrain(
        _state(), _corpus(), predict=_predict, dev_fixture=_fixture(),
        output=tmp_path/'pretrain', identity={'source': 'fixture'}, updates=3,
        max_updates=4, warmup_updates=1, validation_every=2,
        batch_size=2, max_wall_seconds=60)
    assert report['status'] == 'completed' and report['updates'] == 3
    assert report['environment_interactions'] == 0
    assert [entry['update'] for entry in report['metrics']] == [1, 2, 3]
    assert [name for _, name in report['scores']] == ['initial', 'update_0002', 'update_0003']
    assert report['selected'] == 'update_0003'
    restored = diffusion.restore_state(tmp_path/'pretrain'/report['selected'], _state(), {'source': 'fixture'})
    for a, b in zip(jax.tree.leaves(selected), jax.tree.leaves(restored)):
        np.testing.assert_array_equal(a, b)
    assert np.array_equal(selected['normalizer']['std'], _state()['normalizer']['std'])
    assert json.loads((tmp_path/'pretrain/cost_progress.json').read_text())['charged_updates'] == 3


def test_pretrain_refuses_over_budget_and_ambiguous_replay(tmp_path):
    kwargs = dict(predict=_predict, dev_fixture=_fixture(), output=tmp_path/'pretrain',
                  identity={'source': 'fixture'}, max_updates=4, warmup_updates=1,
                  validation_every=2, batch_size=2, max_wall_seconds=60)
    with pytest.raises(ValueError, match='budget'):
        diffusion.train_pretrain(_state(), _corpus(), updates=5, **kwargs)
    assert not (tmp_path/'pretrain').exists()
    diffusion.train_pretrain(_state(), _corpus(), updates=1, **kwargs)
    with pytest.raises(FileExistsError):
        diffusion.train_pretrain(_state(), _corpus(), updates=1, **kwargs)


def test_pretrain_failure_charges_attempt_and_keeps_checkpoint(tmp_path):
    def nonfinite_predict(params, x, obs, k):
        return jnp.ones_like(x) * jnp.where(jnp.any(x < 0), jnp.nan, params['w'])
    with pytest.raises(FloatingPointError):
        diffusion.train_pretrain(
            _state(), _corpus(), predict=nonfinite_predict, dev_fixture=_fixture(),
            output=tmp_path/'failed', identity={'source': 'fixture'}, updates=2,
            max_updates=4, warmup_updates=1, validation_every=2,
            batch_size=2, max_wall_seconds=60)
    receipt = json.loads((tmp_path/'failed/failure.json').read_text())
    assert receipt['status'] == 'failed'
    assert receipt['charged_updates'] == 1
    assert receipt['completed_updates'] == 0
    assert (tmp_path/'failed/initial/state.msgpack').exists()
