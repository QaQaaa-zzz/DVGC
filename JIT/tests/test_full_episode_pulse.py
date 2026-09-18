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
