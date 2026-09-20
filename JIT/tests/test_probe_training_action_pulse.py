import numpy as np
import pytest
import jax
import jax.numpy as jp
from jit_dvgc.iterative_probe_training import (
    validate_training_action_pulse, initialize_training_action_pulse,
    apply_training_action_pulse,
)

PULSE = dict(onsets=[0, 5, 10, 15, 25], steps=3, delta_limit=[.25]*4, probability=.5)


def test_absent_is_identity():
    info = {'rng': jax.random.PRNGKey(3)}
    action = jp.ones(4)
    assert initialize_training_action_pulse(info, info['rng'], None) is info
    actual, next_info, metrics = apply_training_action_pulse(info, action, None)
    assert actual is action and next_info is info and metrics == {}


def test_exact_window_clipping_and_independent_draws():
    cfg = {**PULSE, 'onsets': [5], 'probability': 1.}
    physical_rng = jax.random.PRNGKey(92)
    initial = initialize_training_action_pulse({'rng': physical_rng}, jax.random.PRNGKey(2), cfg)
    info = initial
    requests = []
    for tick in range(30):
        actual, info, metrics = apply_training_action_pulse(info, jp.ones(4), cfg)
        request = np.asarray(info['training_pulse_request'])
        requests.append(request)
        assert bool(metrics['pulse/active']) == (5 <= tick < 8)
        np.testing.assert_array_equal(info['rng'], physical_rng)
        np.testing.assert_allclose(actual, np.clip(1 + request, -1, 1))
        np.testing.assert_allclose(info['training_pulse_effective'], actual - 1)
    requests = np.asarray(requests)
    assert np.flatnonzero(np.any(requests != 0, axis=1)).tolist() == [5, 6, 7]
    assert len(np.unique(requests[5:8])) == 12
    assert np.max(np.abs(requests)) <= .25
    again = initialize_training_action_pulse({'rng': physical_rng}, jax.random.PRNGKey(2), cfg)
    for key in initial:
        np.testing.assert_array_equal(initial[key], again[key])


def test_vectorized_reset_randomizes_schedule_and_episode_rng():
    reset = jax.jit(jax.vmap(lambda key: initialize_training_action_pulse({}, key, PULSE)))
    infos = reset(jax.random.split(jax.random.PRNGKey(1), 256))
    assert set(np.asarray(infos['training_pulse_onset']).tolist()) == set(PULSE['onsets'])
    assert 80 < np.asarray(infos['training_pulse_enabled']).sum() < 180
    cfg = {**PULSE, 'probability': 0.}
    info = initialize_training_action_pulse({}, jax.random.PRNGKey(4), cfg)
    actual, _, metrics = apply_training_action_pulse(info, jp.zeros(4), cfg)
    np.testing.assert_array_equal(actual, np.zeros(4))
    assert not metrics['pulse/active']


@pytest.mark.parametrize('change', [dict(onsets=[]), dict(onsets=[-1]), dict(onsets=[399]),
    dict(steps=0), dict(delta_limit=[.2]*3), dict(delta_limit=[float('nan')]*4),
    dict(probability=1.1), dict(onsets=[True])])
def test_invalid_config_rejected(change):
    with pytest.raises(ValueError):
        validate_training_action_pulse({**PULSE, **change}, horizon=400)


def test_evaluation_runtime_disables_training_augmentation_before_rollout(monkeypatch):
    from types import SimpleNamespace
    from jit_dvgc import pulse_exploration_runtime as runtime, unified_formal
    env = SimpleNamespace(_training_action_pulse=PULSE)
    cfg = SimpleNamespace(raw={'success_criterion': 'stable_forward_recovery'})
    monkeypatch.setattr(runtime, 'controller_mode', lambda spec: 'fixed_random')
    monkeypatch.setattr(runtime, 'selected_event', lambda spec: None)
    monkeypatch.setattr(runtime, 'descent_clearance', lambda spec: 0.)
    monkeypatch.setattr(runtime, 'load_probe_bank', lambda path: {'members': [
        {'name': 'student', 'policy': {'formal_config': 'student.json'}}]})
    monkeypatch.setattr(unified_formal, 'build_unified_formal_environment', lambda path: (cfg, None, env))
    # Stop before checkpoint loading; exercise the actual common runtime builder
    # used by nominal, fixed-window, and teacher-label continuations.
    with pytest.raises(ValueError, match='endpoint differ'):
        runtime.networks({'bank': 'bank.json', 'proposer': 'student',
                          'success_criterion': 'first_valid_landing'})
    assert env._training_action_pulse is None
    # A fixed-start evaluation state need not contain any augmentation keys.
    action = jp.asarray([.1, .2, .3, .4])
    actual, info, metrics = apply_training_action_pulse({}, action, env._training_action_pulse)
    assert actual is action and info == {} and metrics == {}
