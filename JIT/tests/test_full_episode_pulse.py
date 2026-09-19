import numpy as np
import pytest
from jit_dvgc.pulse_schedule import collection_steps, pulse_activity


def test_full_episode_retains_horizon_after_three_pulse_steps():
    spec = dict(full_episode_rollout=True, controller_mode='fixed_random', horizon=400, pulse_steps=3)
    assert collection_steps(spec, np.array([0, 25]), None) == 400
    trigger = np.array([-1]); applied = np.array([0]); masks = []
    for tick in range(8):
        trigger, mask = pulse_activity(trigger, applied, tick >= 2, np.array([True]), tick, 3)
        masks.append(bool(mask[0])); applied = applied + mask
    assert masks == [False, False, True, True, True, False, False, False]


def test_legacy_prefix_length_unchanged_and_learning_cannot_use_eval_mode():
    spec = dict(horizon=400, pulse_steps=3)
    assert collection_steps(spec, np.array([0, 25]), None) == 28
    assert collection_steps(spec, np.array([0]), 'liftoff') == 400
    with pytest.raises(ValueError):
        collection_steps({**spec, 'full_episode_rollout': True}, np.array([0]), None)


def test_phase_policy_full_episode_cutoff_needs_no_continuation_identity():
    from jit_dvgc.pulse_exploration_runtime import completed_trace_endpoint
    # Phase policies deliberately have neither formal_config_sha256 nor iteration.
    phase_policy = {'source_checkpoint': 'phase_u/transition_14991360'}
    arrays = {'data/qpos': np.zeros(7), 'data/qvel': np.zeros(6)}
    assert 'formal_config_sha256' not in phase_policy
    endpoint = completed_trace_endpoint(arrays=arrays, terminal=False,
        stage_reached=True, full_episode=True, valid=False, failure=False,
        prefix_sha='trace', lane=198, tick=399)
    assert endpoint['snapshot'] is None
    assert endpoint['endpoint_kind'] == 'horizon_trace'
    assert endpoint['terminal_reason'] == 'horizon_exhausted'
    assert endpoint['prefix_label'] is None
    assert completed_trace_endpoint(arrays=arrays, terminal=False,
        stage_reached=True, full_episode=False, valid=False, failure=False,
        prefix_sha='trace', lane=198, tick=2) is None
    failure = completed_trace_endpoint(arrays=arrays, terminal=True,
        stage_reached=True, full_episode=True, valid=False, failure=True,
        prefix_sha='trace', lane=198, tick=399)
    assert failure['endpoint_kind'] == 'terminal_trace'
    assert failure['prefix_label'] == 0
