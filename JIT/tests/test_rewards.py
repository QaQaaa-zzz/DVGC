from __future__ import annotations

import jax
from jax import numpy as jp
import pytest

from jit_dvgc.config import load_config
from jit_dvgc.constants import REWARD_COMPONENT_KEYS
from jit_dvgc.rewards import RewardInputs, RewardState, phase_u_reward


@pytest.fixture
def config(jit_root):
    return load_config(jit_root / "configs" / "phase_u_smoke.json")


def _state(**overrides) -> RewardState:
    values = dict(
        x=jp.array(2.5),
        z=jp.array(0.3),
        clearance=jp.array(0.03),
        roll=jp.array(0.0),
        pitch=jp.array(0.0),
        angular_speed=jp.array(0.1),
        vertical_velocity=jp.array(0.5),
        forward_velocity=jp.array(4.0),
        obstacle_relative_x=jp.array(0.7),
    )
    values.update(overrides)
    return RewardState(**values)


def _inputs(**overrides) -> RewardInputs:
    values = dict(
        previous=_state(),
        current=_state(x=jp.array(2.52), z=jp.array(0.32), clearance=jp.array(0.05)),
        action=jp.zeros(4),
        last_action=jp.zeros(4),
        window_latched=jp.array(True),
        first_window_entry=jp.array(False),
        first_liftoff=jp.array(False),
        first_stable_airborne=jp.array(False),
        stable_airborne=jp.array(True),
        ascending_seen=jp.array(True),
        first_apex_success=jp.array(False),
        illegal_contact=jp.array(False),
        physical_failure_transition=jp.array(False),
        timeout_transition=jp.array(False),
    )
    values.update(overrides)
    return RewardInputs(**values)


def test_every_jump_positive_is_exactly_zero_before_window(config):
    result = phase_u_reward(
        _inputs(
            window_latched=jp.array(False),
            first_liftoff=jp.array(True),
            first_stable_airborne=jp.array(True),
            stable_airborne=jp.array(True),
            ascending_seen=jp.array(True),
            first_apex_success=jp.array(True),
        ),
        config.reward,
        config.apex,
        config.physical_limits,
    )

    for key in (
        "liftoff",
        "stable_airborne",
        "ascent",
        "clearance",
        "apex_progress",
        "apex_success",
    ):
        assert float(result.components[key]) == 0.0


def test_early_airborne_is_not_a_penalty_or_success(config):
    result = phase_u_reward(
        _inputs(
            window_latched=jp.array(False),
            stable_airborne=jp.array(True),
            first_liftoff=jp.array(True),
        ),
        config.reward,
        config.apex,
        config.physical_limits,
    )

    assert float(result.components["physical_failure"]) == 0.0
    assert float(result.components["apex_success"]) == 0.0
    assert float(result.components["illegal_contact"]) == 0.0


def test_high_rotation_suppresses_motion_credit_and_adds_rate_penalty(config):
    low = phase_u_reward(
        _inputs(current=_state(z=jp.array(0.32), clearance=jp.array(0.05), angular_speed=jp.array(0.0))),
        config.reward,
        config.apex,
        config.physical_limits,
    )
    high = phase_u_reward(
        _inputs(current=_state(z=jp.array(0.32), clearance=jp.array(0.05), angular_speed=jp.array(20.0))),
        config.reward,
        config.apex,
        config.physical_limits,
    )

    assert float(high.components["ascent"]) < float(low.components["ascent"]) * 1e-8
    assert float(high.components["rate"]) < 0.0


def test_total_is_finite_bounded_and_has_stable_component_keys(config):
    base = _inputs()
    inputs = jax.tree.map(lambda value: jp.stack((value, value * 1e6)), base)
    results = jax.jit(
        jax.vmap(
            lambda item: phase_u_reward(
                item,
                config.reward,
                config.apex,
                config.physical_limits,
            )
        )
    )(inputs)

    assert tuple(results.components) == REWARD_COMPONENT_KEYS
    assert bool(jp.isfinite(results.total).all())
    assert bool((results.total >= -50.0).all())
    assert bool((results.total <= 50.0).all())


def test_support_reward_does_not_accumulate_history(config):
    result = phase_u_reward(
        _inputs(), config.reward, config.apex, config.physical_limits
    )
    assert "recovery_hold" not in result.components
