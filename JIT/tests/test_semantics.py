from __future__ import annotations

from dataclasses import replace

from jax import numpy as jp
import pytest

from jit_dvgc.config import load_config
from jit_dvgc.constants import END_APEX_SUCCESS, END_ONGOING, END_PITCH_LIMIT, END_TIMEOUT
from jit_dvgc.semantics import (
    APEX_GATE_FIELDS,
    ApexSignals,
    TerminalInputs,
    advance_events,
    apex_membership,
    classify_terminal,
    initial_event_state,
)


@pytest.fixture
def config(jit_root):
    return load_config(jit_root / "configs" / "phase_u_smoke.json")


def _valid_apex(**overrides) -> ApexSignals:
    values = dict(
        stable_airborne=jp.array(True),
        vertical_velocity=jp.array(0.0),
        clearance=jp.array(0.2),
        roll=jp.array(0.0),
        pitch=jp.array(0.0),
        angular_speed=jp.array(0.1),
        forward_velocity=jp.array(4.0),
        obstacle_relative_x=jp.array(0.25),
        illegal_contact=jp.array(False),
        physical_failure=jp.array(False),
    )
    values.update(overrides)
    return ApexSignals(**values)


def test_window_latch_is_monotonic(config):
    event = advance_events(
        initial_event_state(),
        _valid_apex(obstacle_relative_x=jp.array(1.0), stable_airborne=jp.array(False)),
        config,
        supported=jp.array(True),
    )
    assert bool(event.window_latched)

    later = advance_events(
        event,
        _valid_apex(obstacle_relative_x=jp.array(2.0), stable_airborne=jp.array(False)),
        config,
        supported=jp.array(True),
    )
    assert bool(later.window_latched)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stable_airborne", False),
        ("vertical_velocity", 0.08),
        ("clearance", 0.10),
        ("roll", 0.08),
        ("pitch", 0.06),
        ("angular_speed", 1.3),
        ("forward_velocity", 3.0),
        ("obstacle_relative_x", 0.10),
        ("illegal_contact", True),
        ("physical_failure", True),
    ],
)
def test_apex_requires_every_declared_gate(config, field, value):
    assert bool(apex_membership(_valid_apex(), config.apex))
    assert not bool(apex_membership(replace(_valid_apex(), **{field: jp.array(value)}), config.apex))
    assert field in APEX_GATE_FIELDS


def _terminal_inputs(**overrides) -> TerminalInputs:
    values = dict(
        episode_step=jp.array(0),
        nonfinite=jp.array(False),
        roll=jp.array(0.0),
        pitch=jp.array(0.0),
        illegal_contact=jp.array(False),
        illegal_wheel_contact=jp.array(False),
        backward_exit=jp.array(False),
        platform_overrun=jp.array(False),
        apex_success=jp.array(False),
    )
    values.update(overrides)
    return TerminalInputs(**values)


def test_horizon_is_truncated_not_terminated(config):
    terminal = classify_terminal(
        _terminal_inputs(episode_step=jp.array(config.ppo.episode_horizon - 1)),
        config,
    )
    assert not bool(terminal.terminated)
    assert bool(terminal.truncated)
    assert int(terminal.end_code) == END_TIMEOUT


def test_apex_success_terminates_without_timeout(config):
    terminal = classify_terminal(_terminal_inputs(apex_success=jp.array(True)), config)
    assert bool(terminal.terminated)
    assert not bool(terminal.truncated)
    assert bool(terminal.success)
    assert int(terminal.end_code) == END_APEX_SUCCESS


def test_physical_failure_has_precedence_over_apex_success(config):
    terminal = classify_terminal(
        _terminal_inputs(pitch=jp.array(2.0), apex_success=jp.array(True)),
        config,
    )
    assert bool(terminal.terminated)
    assert bool(terminal.physical_failure)
    assert not bool(terminal.success)
    assert int(terminal.end_code) == END_PITCH_LIMIT


def test_early_airborne_without_apex_is_ongoing(config):
    terminal = classify_terminal(_terminal_inputs(), config)
    assert not bool(terminal.terminated)
    assert not bool(terminal.truncated)
    assert int(terminal.end_code) == END_ONGOING
