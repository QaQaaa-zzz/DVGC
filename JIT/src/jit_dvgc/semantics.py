"""Pure JAX one-shot jump signal, height/descent Apex, and terminal outcomes."""

from __future__ import annotations

from flax import struct
import jax
from jax import numpy as jp

from .config import ResolvedConfig
from .constants import (
    END_BACKWARD_EXIT,
    END_NONFINITE,
    END_ONGOING,
    END_PITCH_LIMIT,
    END_PROHIBITED_CONTACT,
    END_ROLL_LIMIT,
    END_TIMEOUT,
)


@struct.dataclass
class PhaseUSignals:
    x: jax.Array
    z: jax.Array
    vertical_velocity: jax.Array
    physical_failure: jax.Array


@struct.dataclass
class EventState:
    jump_signal: jax.Array
    jump_zone_seen: jax.Array
    jump_zone_consumed: jax.Array
    ascending_seen: jax.Array
    height_seen: jax.Array
    apex_seen: jax.Array
    episode_step: jax.Array


@struct.dataclass
class TerminalInputs:
    episode_step: jax.Array
    nonfinite: jax.Array
    roll: jax.Array
    pitch: jax.Array
    illegal_contact: jax.Array
    backward_exit: jax.Array


@struct.dataclass
class TerminalState:
    terminated: jax.Array
    truncated: jax.Array
    success: jax.Array
    physical_failure: jax.Array
    timeout: jax.Array
    end_code: jax.Array


def _inside_jump_zone(root_x: jax.Array, config: ResolvedConfig) -> jax.Array:
    return (root_x >= config.events.jump_zone_x_min) & (
        root_x <= config.events.jump_zone_x_max
    )


def initial_event_state(root_x: jax.Array, config: ResolvedConfig) -> EventState:
    inside = _inside_jump_zone(root_x, config)
    passed = root_x > config.events.jump_zone_x_max
    false = jp.asarray(False)
    return EventState(
        jump_signal=inside,
        jump_zone_seen=inside,
        jump_zone_consumed=passed,
        ascending_seen=false,
        height_seen=false,
        apex_seen=false,
        episode_step=jp.asarray(0, jp.int32),
    )


def advance_events(
    previous: EventState,
    signals: PhaseUSignals,
    config: ResolvedConfig,
) -> EventState:
    inside = _inside_jump_zone(signals.x, config)
    exited = jp.asarray(previous.jump_signal, dtype=bool) & ~inside
    consumed = (
        jp.asarray(previous.jump_zone_consumed, dtype=bool)
        | exited
        | (signals.x > config.events.jump_zone_x_max)
    )
    jump_signal = inside & ~jp.asarray(previous.jump_zone_consumed, dtype=bool)
    zone_seen = jp.asarray(previous.jump_zone_seen, dtype=bool) | jump_signal
    ascending_seen = jp.asarray(previous.ascending_seen, dtype=bool) | (
        zone_seen & (signals.vertical_velocity >= config.events.min_ascent_velocity)
    )
    height_seen = jp.asarray(previous.height_seen, dtype=bool) | (
        zone_seen & (signals.z >= config.events.apex_height)
    )
    apex_now = (
        zone_seen
        & ascending_seen
        & height_seen
        & (signals.vertical_velocity <= -config.events.min_descent_velocity)
        & ~jp.asarray(signals.physical_failure, dtype=bool)
    )
    return EventState(
        jump_signal=jump_signal,
        jump_zone_seen=zone_seen,
        jump_zone_consumed=consumed,
        ascending_seen=ascending_seen,
        height_seen=height_seen,
        apex_seen=jp.asarray(previous.apex_seen, dtype=bool) | apex_now,
        episode_step=previous.episode_step + 1,
    )


def classify_terminal(inputs: TerminalInputs, config: ResolvedConfig) -> TerminalState:
    roll_failure = jp.abs(inputs.roll) > config.physical_limits.max_abs_roll
    pitch_failure = jp.abs(inputs.pitch) > config.physical_limits.max_abs_pitch
    failures = (
        jp.asarray(inputs.nonfinite, dtype=bool),
        roll_failure,
        pitch_failure,
        jp.asarray(inputs.illegal_contact, dtype=bool),
        jp.asarray(inputs.backward_exit, dtype=bool),
    )
    failure_codes = (
        END_NONFINITE,
        END_ROLL_LIMIT,
        END_PITCH_LIMIT,
        END_PROHIBITED_CONTACT,
        END_BACKWARD_EXIT,
    )
    physical_failure = jp.asarray(False)
    failure_code = jp.asarray(END_ONGOING, jp.int32)
    for condition, code in reversed(tuple(zip(failures, failure_codes, strict=True))):
        failure_code = jp.where(condition, jp.asarray(code, jp.int32), failure_code)
        physical_failure = physical_failure | condition
    success = jp.asarray(False)
    horizon = inputs.episode_step >= config.ppo.episode_horizon - 1
    timeout = horizon & ~physical_failure
    end_code = jp.where(timeout, END_TIMEOUT, END_ONGOING)
    end_code = jp.where(physical_failure, failure_code, end_code).astype(jp.int32)
    return TerminalState(
        terminated=physical_failure,
        truncated=timeout,
        success=success,
        physical_failure=physical_failure,
        timeout=timeout,
        end_code=end_code,
    )
