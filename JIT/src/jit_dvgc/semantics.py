"""Pure JAX Propulsion-Ascent events, Apex membership, and terminal outcomes."""

from __future__ import annotations

from flax import struct
import jax
from jax import numpy as jp

from .config import ApexConfig, ResolvedConfig
from .constants import (
    END_APEX_SUCCESS,
    END_BACKWARD_EXIT,
    END_ILLEGAL_WHEEL_CONTACT,
    END_NONFINITE,
    END_ONGOING,
    END_PITCH_LIMIT,
    END_PLATFORM_OVERRUN,
    END_PROHIBITED_CONTACT,
    END_ROLL_LIMIT,
    END_TIMEOUT,
)


APEX_GATE_FIELDS = (
    "stable_airborne",
    "vertical_velocity",
    "clearance",
    "roll",
    "pitch",
    "angular_speed",
    "forward_velocity",
    "obstacle_relative_x",
    "illegal_contact",
    "physical_failure",
)


@struct.dataclass
class ApexSignals:
    stable_airborne: jax.Array
    vertical_velocity: jax.Array
    clearance: jax.Array
    roll: jax.Array
    pitch: jax.Array
    angular_speed: jax.Array
    forward_velocity: jax.Array
    obstacle_relative_x: jax.Array
    illegal_contact: jax.Array
    physical_failure: jax.Array


@struct.dataclass
class EventState:
    window_latched: jax.Array
    liftoff_seen: jax.Array
    stable_airborne_seen: jax.Array
    ascending_seen: jax.Array
    apex_seen: jax.Array
    stable_airborne_current: jax.Array
    airborne_count: jax.Array
    episode_step: jax.Array


@struct.dataclass
class TerminalInputs:
    episode_step: jax.Array
    nonfinite: jax.Array
    roll: jax.Array
    pitch: jax.Array
    illegal_contact: jax.Array
    illegal_wheel_contact: jax.Array
    backward_exit: jax.Array
    platform_overrun: jax.Array
    apex_success: jax.Array


@struct.dataclass
class TerminalState:
    terminated: jax.Array
    truncated: jax.Array
    success: jax.Array
    physical_failure: jax.Array
    timeout: jax.Array
    end_code: jax.Array


def initial_event_state() -> EventState:
    false = jp.asarray(False)
    return EventState(
        window_latched=false,
        liftoff_seen=false,
        stable_airborne_seen=false,
        ascending_seen=false,
        apex_seen=false,
        stable_airborne_current=false,
        airborne_count=jp.asarray(0, jp.int32),
        episode_step=jp.asarray(0, jp.int32),
    )


def apex_components(signals: ApexSignals, config: ApexConfig) -> dict[str, jax.Array]:
    return {
        "stable_airborne": jp.asarray(signals.stable_airborne, dtype=bool),
        "vertical_velocity": jp.abs(signals.vertical_velocity)
        <= config.max_abs_vertical_velocity,
        "clearance": signals.clearance >= config.min_clearance,
        "roll": jp.abs(signals.roll) <= config.max_abs_roll,
        "pitch": jp.abs(signals.pitch) <= config.max_abs_pitch,
        "angular_speed": (signals.angular_speed >= 0.0)
        & (signals.angular_speed <= config.max_angular_speed),
        "forward_velocity": signals.forward_velocity >= config.min_forward_velocity,
        "obstacle_relative_x": (signals.obstacle_relative_x >= config.relative_x_min)
        & (signals.obstacle_relative_x <= config.relative_x_max),
        "illegal_contact": ~jp.asarray(signals.illegal_contact, dtype=bool),
        "physical_failure": ~jp.asarray(signals.physical_failure, dtype=bool),
    }


def apex_membership(signals: ApexSignals, config: ApexConfig) -> jax.Array:
    result = jp.asarray(True)
    for value in apex_components(signals, config).values():
        result = result & value
    return result


def advance_events(
    previous: EventState,
    signals: ApexSignals,
    config: ResolvedConfig,
    *,
    supported: jax.Array,
) -> EventState:
    inside_window = (
        signals.obstacle_relative_x >= config.events.window_relative_x_min
    ) & (signals.obstacle_relative_x <= config.events.window_relative_x_max)
    window_latched = previous.window_latched | inside_window
    airborne_count = jp.where(
        ~jp.asarray(supported, dtype=bool),
        previous.airborne_count + 1,
        jp.asarray(0, jp.int32),
    )
    confirmed_airborne = airborne_count >= config.events.airborne_confirm_ticks
    stable_airborne = confirmed_airborne & (
        signals.clearance >= config.events.stable_airborne_min_clearance
    )
    liftoff_seen = previous.liftoff_seen | (window_latched & confirmed_airborne)
    stable_seen = previous.stable_airborne_seen | (window_latched & stable_airborne)
    ascending_seen = previous.ascending_seen | (
        window_latched
        & stable_airborne
        & (signals.vertical_velocity >= config.events.ascending_min_vertical_velocity)
    )
    apex_now = apex_membership(
        signals.replace(stable_airborne=stable_airborne), config.apex
    )
    return EventState(
        window_latched=window_latched,
        liftoff_seen=liftoff_seen,
        stable_airborne_seen=stable_seen,
        ascending_seen=ascending_seen,
        apex_seen=previous.apex_seen | apex_now,
        stable_airborne_current=stable_airborne,
        airborne_count=airborne_count,
        episode_step=previous.episode_step + 1,
    )


def classify_terminal(
    inputs: TerminalInputs, config: ResolvedConfig
) -> TerminalState:
    roll_failure = jp.abs(inputs.roll) > config.physical_limits.max_abs_roll
    pitch_failure = jp.abs(inputs.pitch) > config.physical_limits.max_abs_pitch
    failures = (
        jp.asarray(inputs.nonfinite, dtype=bool),
        roll_failure,
        pitch_failure,
        jp.asarray(inputs.illegal_contact, dtype=bool),
        jp.asarray(inputs.illegal_wheel_contact, dtype=bool),
        jp.asarray(inputs.backward_exit, dtype=bool),
        jp.asarray(inputs.platform_overrun, dtype=bool),
    )
    failure_codes = (
        END_NONFINITE,
        END_ROLL_LIMIT,
        END_PITCH_LIMIT,
        END_PROHIBITED_CONTACT,
        END_ILLEGAL_WHEEL_CONTACT,
        END_BACKWARD_EXIT,
        END_PLATFORM_OVERRUN,
    )
    physical_failure = jp.asarray(False)
    failure_code = jp.asarray(END_ONGOING, jp.int32)
    for condition, code in reversed(tuple(zip(failures, failure_codes, strict=True))):
        failure_code = jp.where(condition, jp.asarray(code, jp.int32), failure_code)
        physical_failure = physical_failure | condition
    success = jp.asarray(inputs.apex_success, dtype=bool) & ~physical_failure
    horizon = inputs.episode_step >= config.ppo.episode_horizon - 1
    timeout = horizon & ~physical_failure & ~success
    end_code = jp.where(success, END_APEX_SUCCESS, END_ONGOING)
    end_code = jp.where(timeout, END_TIMEOUT, end_code)
    end_code = jp.where(physical_failure, failure_code, end_code).astype(jp.int32)
    return TerminalState(
        terminated=physical_failure | success,
        truncated=timeout,
        success=success,
        physical_failure=physical_failure,
        timeout=timeout,
        end_code=end_code,
    )
