from __future__ import annotations

import math

import jax
from jax import numpy as jp
import pytest

from jit_dvgc.config import load_config
from jit_dvgc.constants import REWARD_COMPONENT_KEYS
from jit_dvgc.rewards import RewardInputs, RewardState, phase_u_reward


@pytest.fixture
def config(jit_root):
    return load_config(jit_root / "configs" / "phase_u_smoke.json")


@pytest.fixture
def v4_config(jit_root):
    return load_config(jit_root / "configs" / "phase_u_continuation_smoke.json")


def _state(**overrides) -> RewardState:
    values = dict(
        x=jp.array(2.8),
        y=jp.array(0.0),
        z=jp.array(0.3),
        roll=jp.array(0.0),
        pitch=jp.array(0.0),
        yaw=jp.array(0.0),
        forward_velocity=jp.array(3.5),
        lateral_velocity=jp.array(0.0),
        vertical_velocity=jp.array(0.2),
        roll_rate=jp.array(0.0),
        pitch_rate=jp.array(0.0),
        yaw_rate=jp.array(0.0),
        hip_velocity=jp.array(0.0),
        knee_velocity=jp.array(0.0),
        hip_force=jp.array(0.0),
        knee_force=jp.array(0.0),
    )
    values.update(overrides)
    return RewardState(**values)


def _inputs(**overrides) -> RewardInputs:
    values = dict(
        current=_state(),
        action=jp.zeros(4),
        last_action=jp.zeros(4),
        jump_signal=jp.array(True),
        first_apex_success=jp.array(False),
        illegal_contact=jp.array(False),
        physical_failure_transition=jp.array(False),
        stuck_transition=jp.array(False),
        yaw_limit_transition=jp.array(False),
        timeout_transition=jp.array(False),
    )
    values.update(overrides)
    return RewardInputs(**values)


def _reward(config, **overrides):
    return phase_u_reward(_inputs(**overrides), config.reward, config.physical_limits)


@pytest.mark.parametrize(("degrees", "expected"), [(0.0, 3.0), (5.0, 0.0), (15.0, -3.0)])
def test_roll_component_matches_reference_piecewise_values(config, degrees, expected):
    result = _reward(config, current=_state(roll=jp.array(math.radians(degrees))))
    assert float(result.components.roll) == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize(
    ("degrees", "expected"),
    [(0.0, 1.0), (3.0, 0.9), (8.0, 0.5), (10.0, 0.0), (20.0, -1.0)],
)
def test_pitch_component_matches_reference_piecewise_values(config, degrees, expected):
    result = _reward(config, current=_state(pitch=jp.array(math.radians(degrees))))
    assert float(result.components.pitch) == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize(
    ("degrees", "expected"),
    [(0.0, 0.3), (8.0, 0.15), (15.0, 0.06), (25.0, 0.015), (50.0, 0.0)],
)
def test_yaw_component_matches_reference_piecewise_values(config, degrees, expected):
    result = _reward(config, current=_state(yaw=jp.array(math.radians(degrees))))
    assert float(result.components.yaw) == pytest.approx(expected, abs=1e-5)


def test_speed_and_survival_match_reference_values(config):
    exact = _reward(config, current=_state(forward_velocity=jp.array(3.5)))
    half_meter_error = _reward(config, current=_state(forward_velocity=jp.array(3.0)))
    assert float(exact.components.speed) == pytest.approx(0.2)
    assert float(half_meter_error.components.speed) == pytest.approx(0.2 * math.exp(-0.5))
    assert float(exact.components.survival) == pytest.approx(1.5)


@pytest.mark.parametrize(
    ("z", "expected"),
    [(0.34, 0.0), (0.35, 20.0), (0.50, 30.0), (0.80, 12.0), (0.81, 8.0)],
)
def test_height_component_matches_reference_curve(config, z, expected):
    result = _reward(config, current=_state(z=jp.array(z)))
    assert float(result.components.height) == pytest.approx(expected, abs=1e-4)


def test_height_is_exactly_zero_when_current_jump_signal_is_zero(config):
    result = _reward(
        config, current=_state(z=jp.array(0.5)), jump_signal=jp.array(False)
    )
    assert float(result.components.height) == 0.0


def test_v4_height_component_uses_40x_raw_height_only_while_signaled(v4_config):
    signaled = _reward(
        v4_config, current=_state(z=jp.array(0.5)), jump_signal=jp.array(True)
    )
    unsignaled = _reward(
        v4_config, current=_state(z=jp.array(0.5)), jump_signal=jp.array(False)
    )

    assert float(signaled.components.height) == pytest.approx(60.0)
    assert float(unsignaled.components.height) == 0.0


@pytest.mark.parametrize(
    ("z", "jump_signal", "expected"),
    [
        (0.15, True, -3.0),
        (0.25, True, -1.5),
        (0.35, True, 0.0),
        (0.15, False, 0.0),
    ],
)
def test_v4_low_height_penalty_provides_dense_progress_below_reward_threshold(
    v4_config, z, jump_signal, expected
):
    result = _reward(
        v4_config,
        current=_state(z=jp.array(z)),
        jump_signal=jp.array(jump_signal),
    )

    assert float(result.components.low_height) == pytest.approx(expected, abs=1e-5)


def test_action_rate_and_joint_energy_costs_match_reference(config):
    result = _reward(
        config,
        current=_state(
            pitch_rate=jp.array(2.0),
            hip_velocity=jp.array(3.0),
            knee_velocity=jp.array(-4.0),
            hip_force=jp.array(10.0),
            knee_force=jp.array(-5.0),
        ),
        action=jp.ones(4),
        last_action=jp.zeros(4),
    )
    assert float(result.components.action_smoothness) == pytest.approx(-0.0006)
    assert float(result.components.action_magnitude) == pytest.approx(-0.6)
    assert float(result.components.pitch_rate) == pytest.approx(-0.075)
    assert float(result.components.roll_rate) == 0.0
    assert float(result.components.yaw_rate) == 0.0
    assert float(result.components.joint_energy) == pytest.approx(-2.0)


def test_terminal_components_and_total_clipping_are_separate(config):
    success = _reward(
        config,
        current=_state(z=jp.array(0.5)),
        first_apex_success=jp.array(True),
    )
    failure = _reward(
        config,
        jump_signal=jp.array(False),
        illegal_contact=jp.array(True),
        physical_failure_transition=jp.array(True),
    )
    assert float(success.components.apex_success) == 50.0
    assert float(success.unclipped_total) == pytest.approx(86.0)
    assert float(success.total) == pytest.approx(50.0)
    assert float(failure.components.illegal_contact) == -30.0
    assert float(failure.components.physical_failure) == -30.0
    assert float(failure.unclipped_total) == pytest.approx(-54.0)
    assert float(failure.total) == pytest.approx(-50.0)


def test_v4_task_terminal_penalties_are_large_distinct_and_not_double_counted(
    v4_config,
):
    stuck = _reward(v4_config, stuck_transition=jp.array(True))
    yaw = _reward(v4_config, yaw_limit_transition=jp.array(True))

    assert float(stuck.components.stuck) == -40.0
    assert float(stuck.components.physical_failure) == 0.0
    assert float(yaw.components.yaw_limit) == -40.0
    assert float(yaw.components.physical_failure) == 0.0


def test_v4_overlapping_terminal_inputs_apply_only_the_highest_priority_penalty(
    v4_config,
):
    physical = _reward(
        v4_config,
        physical_failure_transition=jp.array(True),
        stuck_transition=jp.array(True),
        yaw_limit_transition=jp.array(True),
    )
    stuck = _reward(
        v4_config,
        stuck_transition=jp.array(True),
        yaw_limit_transition=jp.array(True),
    )

    assert float(physical.components.physical_failure) == -30.0
    assert float(physical.components.stuck) == 0.0
    assert float(physical.components.yaw_limit) == 0.0
    assert float(stuck.components.physical_failure) == 0.0
    assert float(stuck.components.stuck) == -40.0
    assert float(stuck.components.yaw_limit) == 0.0


def test_component_keys_exclude_old_shaping_and_target_terms(config):
    result = _reward(config)
    assert tuple(result.components) == REWARD_COMPONENT_KEYS
    for forbidden in (
        "drive",
        "window",
        "liftoff",
        "stable_airborne",
        "ascent",
        "clearance",
        "apex_progress",
        "target",
        "position",
        "direction",
    ):
        assert forbidden not in result.components


def test_reward_is_jittable_finite_and_bounded(config):
    base = _inputs()
    inputs = jax.tree.map(lambda value: jp.stack((value, value * 1e3)), base)
    results = jax.jit(
        jax.vmap(lambda item: phase_u_reward(item, config.reward, config.physical_limits))
    )(inputs)
    assert bool(jp.isfinite(results.total).all())
    assert bool((results.total >= -50.0).all())
    assert bool((results.total <= 50.0).all())
