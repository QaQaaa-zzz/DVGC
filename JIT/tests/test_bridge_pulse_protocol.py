import jax
import jax.numpy as jp
import numpy as np
import pytest

from jit_dvgc.generative_bridge.pulse_protocol import (
    logical_episode_key, requested_draws, EpisodeKeyRegistry, validate_ancestor_splits,
)
from jit_dvgc.iterative_probe_training import (
    initialize_training_action_pulse, apply_training_action_pulse, validate_training_action_pulse,
)

PULSE = dict(pulse_scope='full_task_start_only', onsets=[0], steps=3,
             delta_limit=[.25]*4, probability=1., training_conditions=[
                 {'probability': .2, 'amplitude': 0.},
                 {'probability': .4, 'amplitude': .1},
                 {'probability': .4, 'amplitude': .25}])


def test_logical_draws_are_partition_invariant_and_separate_all_components():
    def draw(i, **kw):
        return requested_draws(logical_episode_key(9281201, 'train', 2, i,
                                                  kw.pop('purpose', 'pulse'), **kw), .25)
    ids = [0, 1, 2**32, 2**32+1, 2**64-1]
    full = np.stack([draw(i) for i in ids])
    partitioned = np.concatenate([np.stack([draw(i) for i in part])
                                  for part in [ids[:2], ids[2:]]])
    np.testing.assert_array_equal(full, partitioned)
    assert full.shape == (5, 3, 4)
    assert len({x.tobytes() for x in full}) == 5
    keys = [logical_episode_key(*args) for args in [
        (1, 'train', 0, 0, 'pulse'), (2, 'train', 0, 0, 'pulse'),
        (1, 'student_dev', 0, 0, 'pulse'), (1, 'train', 1, 0, 'pulse'),
        (1, 'train', 0, 1, 'pulse'), (1, 'train', 0, 0, 'policy')]]
    assert len({np.asarray(k).tobytes() for k in keys}) == 6


def test_registry_detects_duplicate_keys_and_split_leakage():
    registry = EpisodeKeyRegistry()
    registry.claim(1, 'train', 0, 7, 'pulse')
    with pytest.raises(ValueError, match='duplicate'):
        registry.claim(1, 'train', 0, 7, 'pulse')
    registry.claim(1, 'student_dev', 0, 7, 'pulse')
    validate_ancestor_splits({'train': ['root1', 'root1'], 'student_dev': ['root2']})
    with pytest.raises(ValueError, match='ancestor'):
        validate_ancestor_splits({'train': ['root1'], 'student_dev': ['root1']})


def test_scope_no_second_pulse_and_exact_first_three_steps():
    cfg = {**PULSE, 'training_conditions': [{'probability': 1., 'amplitude': .25}]}
    for full_start in [False, True]:
        info = initialize_training_action_pulse({}, jax.random.PRNGKey(5), cfg,
                                               full_task_start=full_start)
        requests = []
        for _ in range(7):
            _, info, _ = apply_training_action_pulse(info, jp.ones(4), cfg)
            requests.append(np.asarray(info['training_pulse_request']))
        requests = np.asarray(requests)
        assert np.flatnonzero(np.any(requests, axis=1)).tolist() == ([0,1,2] if full_start else [])
        np.testing.assert_array_equal(requests[3:], 0.)


def test_scope_requires_explicit_reset_provenance_and_honors_logical_key():
    with pytest.raises(ValueError, match='full_task_start'):
        initialize_training_action_pulse({}, jax.random.PRNGKey(1), PULSE)
    cfg = {**PULSE, 'training_conditions': [{'probability': 1., 'amplitude': .25}]}
    key = logical_episode_key(3, 'train', 0, 123, 'pulse')
    a = initialize_training_action_pulse({}, jax.random.PRNGKey(1), cfg,
                                       full_task_start=True, logical_key=key)
    b = initialize_training_action_pulse({}, jax.random.PRNGKey(9), cfg,
                                       full_task_start=True, logical_key=key)
    np.testing.assert_array_equal(a['training_pulse_draws'], b['training_pulse_draws'])


def test_condition_mixture_and_jitted_scope():
    keys = jax.random.split(jax.random.PRNGKey(42), 1000)
    initialize = jax.jit(jax.vmap(lambda key: initialize_training_action_pulse(
        {}, key, PULSE, full_task_start=True)))
    info = initialize(keys)
    counts = np.unique(np.round(np.asarray(info['training_pulse_amplitude']), 2), return_counts=True)
    np.testing.assert_allclose(counts[0], [0., .1, .25])
    np.testing.assert_allclose(counts[1] / 1000., [.2, .4, .4], atol=.05)


@pytest.mark.parametrize('change', [dict(pulse_scope='typo'), dict(onsets=[5]),
    dict(steps=4), dict(training_conditions=[{'probability': .8, 'amplitude': .1}]),
    dict(training_conditions=[{'probability': 1., 'amplitude': .3}])])
def test_invalid_v12_scope_contract(change):
    with pytest.raises(ValueError):
        validate_training_action_pulse({**PULSE, **change})


def test_collection_draws_receipts_and_duplicate_across_shards():
    from jit_dvgc.generative_bridge.pulse_protocol import collection_draws
    def spec(ids):
        return dict(controller_mode='fixed_random', pulse_steps=3, num_envs=len(ids),
                    pulse_protocol_v1_2=dict(master_seed=12, role='train', round=0, episode_ids=ids))
    all_draws, receipt = collection_draws(spec([12, 13, 2**32]))
    registry = EpisodeKeyRegistry()
    a, ra = collection_draws(spec([12]), registry=registry)
    b, rb = collection_draws(spec([13, 2**32]), registry=registry)
    np.testing.assert_array_equal(all_draws, np.concatenate([a, b]))
    assert receipt['episodes'] == ra['episodes'] + rb['episodes']
    with pytest.raises(ValueError, match='duplicate'):
        collection_draws(spec([12]), registry=registry)
    with pytest.raises(ValueError, match='duplicate'):
        collection_draws(spec([12, 12]))
    final = spec([1]); final['pulse_protocol_v1_2']['role'] = 'test'
    with pytest.raises(ValueError, match='TEST'):
        collection_draws(final)


def test_collection_rejects_duplicate_ids_before_simulator_or_files(tmp_path):
    from jit_dvgc.pulse_exploration_runtime import collect
    spec = dict(controller_mode='fixed_random', pulse_steps=3, num_envs=2,
                pulse_protocol_v1_2=dict(master_seed=12, role='train', round=0, episode_ids=[1,1]))
    with pytest.raises(ValueError, match='duplicate'):
        collect(spec, tmp_path / 'must_not_create')
    assert not (tmp_path / 'must_not_create').exists()


def test_collection_step_selects_lane_draw_and_zeros_dead_or_finished_lanes():
    from jit_dvgc.generative_bridge.pulse_protocol import pulse_delta_for_step
    draws = jp.arange(24, dtype=jp.float32).reshape(2,3,4) / 24
    select = jax.jit(pulse_delta_for_step)
    np.testing.assert_array_equal(select(draws, jp.array([0,2]), jp.array([True,True])),
                                  np.stack([draws[0,0],draws[1,2]]))
    np.testing.assert_array_equal(select(draws, jp.array([3,1]), jp.array([False,False])), 0.)
