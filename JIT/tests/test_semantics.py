from __future__ import annotations

from jax import numpy as jp
import pytest

from jit_dvgc.config import load_config
from jit_dvgc.constants import END_ONGOING, END_PITCH_LIMIT, END_TIMEOUT
from jit_dvgc.semantics import (
    PhaseUSignals,
    TerminalInputs,
    advance_events,
    classify_terminal,
    initial_event_state,
)


@pytest.fixture
def config(jit_root):
    return load_config(jit_root / "configs" / "phase_u_smoke.json")


@pytest.fixture
def v4_config(jit_root):
    return load_config(jit_root / "configs" / "phase_u_continuation_smoke.json")


def _signals(**overrides) -> PhaseUSignals:
    values = dict(
        x=jp.array(2.0),
        z=jp.array(0.15),
        vertical_velocity=jp.array(0.0),
        physical_failure=jp.array(False),
    )
    values.update(overrides)
    return PhaseUSignals(**values)


def test_reset_initializes_one_shot_jump_signal_from_position(config):
    below = initial_event_state(jp.array(2.4), config)
    inside = initial_event_state(jp.array(2.8), config)
    above = initial_event_state(jp.array(3.2), config)

    assert not bool(below.jump_signal)
    assert not bool(below.jump_zone_seen)
    assert not bool(below.jump_zone_consumed)
    assert bool(inside.jump_signal)
    assert bool(inside.jump_zone_seen)
    assert not bool(inside.jump_zone_consumed)
    assert not bool(inside.ascending_seen)
    assert not bool(inside.height_seen)
    assert not bool(above.jump_signal)
    assert not bool(above.jump_zone_seen)
    assert bool(above.jump_zone_consumed)


def test_jump_signal_is_inclusive_then_closes_permanently(config):
    event = initial_event_state(jp.array(2.4), config)
    at_start = advance_events(event, _signals(x=jp.array(2.5)), config)
    at_end = advance_events(at_start, _signals(x=jp.array(3.1)), config)
    left = advance_events(at_end, _signals(x=jp.array(3.2)), config)
    reentered = advance_events(left, _signals(x=jp.array(2.8)), config)

    assert bool(at_start.jump_signal)
    assert bool(at_end.jump_signal)
    assert not bool(left.jump_signal)
    assert bool(left.jump_zone_consumed)
    assert not bool(reentered.jump_signal)
    assert bool(reentered.jump_zone_consumed)


def test_v4_jump_signal_stays_live_to_3_4_then_closes_permanently(v4_config):
    event = initial_event_state(jp.array(2.4), v4_config)
    signal_at_x_3_25 = advance_events(
        event, _signals(x=jp.array(3.25)), v4_config
    )
    signal_after_x_3_41 = advance_events(
        signal_at_x_3_25, _signals(x=jp.array(3.41)), v4_config
    )
    signal_after_return_to_x_3_0 = advance_events(
        signal_after_x_3_41, _signals(x=jp.array(3.0)), v4_config
    )

    assert bool(signal_at_x_3_25.jump_signal) is True
    assert bool(signal_after_x_3_41.jump_signal) is False
    assert bool(signal_after_return_to_x_3_0.jump_signal) is False


def test_apex_requires_legal_zone_height_prior_ascent_and_descent(config):
    event = initial_event_state(jp.array(2.4), config)
    ascending = advance_events(
        event,
        _signals(x=jp.array(2.8), z=jp.array(0.45), vertical_velocity=jp.array(0.05)),
        config,
    )
    high = advance_events(
        ascending,
        _signals(x=jp.array(3.2), z=jp.array(0.5), vertical_velocity=jp.array(0.1)),
        config,
    )
    noise = advance_events(
        high,
        _signals(x=jp.array(3.3), z=jp.array(0.51), vertical_velocity=jp.array(-0.049)),
        config,
    )
    apex = advance_events(
        noise,
        _signals(x=jp.array(3.3), z=jp.array(0.51), vertical_velocity=jp.array(-0.05)),
        config,
    )

    assert bool(ascending.ascending_seen)
    assert not bool(ascending.height_seen)
    assert bool(high.height_seen)
    assert not bool(noise.apex_seen)
    assert bool(apex.apex_seen)


def test_high_rsi_state_cannot_succeed_without_observed_ascent(config):
    event = initial_event_state(jp.array(2.8), config)
    falling = advance_events(
        event,
        _signals(x=jp.array(2.81), z=jp.array(2.0), vertical_velocity=jp.array(-0.2)),
        config,
    )
    assert bool(falling.height_seen)
    assert not bool(falling.ascending_seen)
    assert not bool(falling.apex_seen)


def test_physical_failure_blocks_apex(config):
    event = initial_event_state(jp.array(2.8), config)
    ascending = advance_events(
        event,
        _signals(x=jp.array(2.9), z=jp.array(0.6), vertical_velocity=jp.array(0.2)),
        config,
    )
    failed = advance_events(
        ascending,
        _signals(
            x=jp.array(3.0),
            z=jp.array(0.6),
            vertical_velocity=jp.array(-0.2),
            physical_failure=jp.array(True),
        ),
        config,
    )
    assert not bool(failed.apex_seen)


def _terminal_inputs(**overrides) -> TerminalInputs:
    values = dict(
        episode_step=jp.array(0),
        nonfinite=jp.array(False),
        roll=jp.array(0.0),
        pitch=jp.array(0.0),
        illegal_contact=jp.array(False),
        backward_exit=jp.array(False),
    )
    values.update(overrides)
    return TerminalInputs(**values)


def test_horizon_is_truncated_not_terminated(config):
    terminal = classify_terminal(
        _terminal_inputs(episode_step=jp.array(config.ppo.episode_horizon - 1)), config
    )
    assert not bool(terminal.terminated)
    assert bool(terminal.truncated)
    assert int(terminal.end_code) == END_TIMEOUT


def test_apex_event_does_not_terminate_or_report_terminal_success(config):
    event = initial_event_state(jp.array(2.8), config)
    ascending = advance_events(
        event,
        _signals(x=jp.array(2.9), z=jp.array(0.6), vertical_velocity=jp.array(0.2)),
        config,
    )
    apex = advance_events(
        ascending,
        _signals(x=jp.array(3.0), z=jp.array(0.6), vertical_velocity=jp.array(-0.2)),
        config,
    )
    terminal = classify_terminal(_terminal_inputs(), config)

    assert bool(apex.apex_seen)
    assert not bool(terminal.terminated)
    assert not bool(terminal.truncated)
    assert not bool(terminal.success)
    assert int(terminal.end_code) == END_ONGOING


def test_physical_failure_still_terminates_after_an_apex_event(config):
    terminal = classify_terminal(_terminal_inputs(pitch=jp.array(2.0)), config)
    assert bool(terminal.terminated)
    assert bool(terminal.physical_failure)
    assert not bool(terminal.success)
    assert int(terminal.end_code) == END_PITCH_LIMIT


def test_ordinary_state_is_ongoing(config):
    terminal = classify_terminal(_terminal_inputs(), config)
    assert not bool(terminal.terminated)
    assert not bool(terminal.truncated)
    assert int(terminal.end_code) == END_ONGOING
