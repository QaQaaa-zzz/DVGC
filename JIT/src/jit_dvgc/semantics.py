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
    END_JUMP_ZONE_MISSED,
    END_PITCH_LIMIT,
    END_PROHIBITED_CONTACT,
    END_ROLL_LIMIT,
    END_STUCK,
    END_TIMEOUT,
    END_YAW_LIMIT,
)


@struct.dataclass
class PhaseUSignals:
    x: jax.Array
    z: jax.Array
    forward_velocity: jax.Array
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
    stuck_anchor_x: jax.Array
    stuck_ticks: jax.Array
    stuck: jax.Array
    episode_step: jax.Array


@struct.dataclass
class TerminalInputs:
    episode_step: jax.Array
    nonfinite: jax.Array
    roll: jax.Array
    pitch: jax.Array
    illegal_contact: jax.Array
    backward_exit: jax.Array
    stuck: jax.Array
    yaw: jax.Array
    jump_zone_seen: jax.Array


@struct.dataclass
class TerminalState:
    terminated: jax.Array
    truncated: jax.Array
    success: jax.Array
    physical_failure: jax.Array
    roll_limit: jax.Array
    pitch_limit: jax.Array
    jump_zone_missed: jax.Array
    stuck: jax.Array
    yaw_limit: jax.Array
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
        stuck_anchor_x=jp.asarray(root_x),
        stuck_ticks=jp.asarray(0, jp.int32),
        stuck=false,
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
    if config.events.stuck_forward_velocity_threshold is not None:
        monitor_stuck = jump_signal & ~height_seen
        stuck_anchor_x = signals.x
        stuck_ticks = jp.asarray(0, jp.int32)
        stuck = jp.asarray(previous.stuck, dtype=bool) | (
            monitor_stuck
            & (
                signals.forward_velocity
                <= config.events.stuck_forward_velocity_threshold
            )
        )
    elif config.events.stuck_window_steps is None:
        stuck_anchor_x = signals.x
        stuck_ticks = jp.asarray(0, jp.int32)
        stuck = jp.asarray(False)
    else:
        monitor_stuck = (
            jump_signal
            & ~height_seen
            & (signals.z < config.reward.jump_reward_min_height)
        )
        made_progress = (
            signals.x - previous.stuck_anchor_x
        ) >= config.events.stuck_min_progress
        reset_window = ~monitor_stuck | made_progress
        stuck_anchor_x = jp.where(reset_window, signals.x, previous.stuck_anchor_x)
        stuck_ticks = jp.where(
            reset_window,
            jp.asarray(0, jp.int32),
            previous.stuck_ticks + 1,
        )
        stuck = jp.asarray(previous.stuck, dtype=bool) | (
            monitor_stuck & (stuck_ticks >= config.events.stuck_window_steps)
        )
    return EventState(
        jump_signal=jump_signal,
        jump_zone_seen=zone_seen,
        jump_zone_consumed=consumed,
        ascending_seen=ascending_seen,
        height_seen=height_seen,
        apex_seen=jp.asarray(previous.apex_seen, dtype=bool) | apex_now,
        stuck_anchor_x=stuck_anchor_x,
        stuck_ticks=stuck_ticks,
        stuck=stuck,
        episode_step=previous.episode_step + 1,
    )


def classify_terminal(inputs: TerminalInputs, config: ResolvedConfig) -> TerminalState:
    roll_failure = jp.abs(inputs.roll) > config.physical_limits.max_abs_roll
    pitch_failure = jp.abs(inputs.pitch) > config.physical_limits.max_abs_pitch
    physical_failures = (
        jp.asarray(inputs.nonfinite, dtype=bool),
        roll_failure,
        pitch_failure,
        jp.asarray(inputs.illegal_contact, dtype=bool)
        & jp.asarray(config.physical_limits.terminate_on_prohibited_contact),
        jp.asarray(inputs.backward_exit, dtype=bool),
    )
    raw_yaw_limit = (
        jp.asarray(False)
        if config.physical_limits.max_abs_yaw is None
        else jp.abs(inputs.yaw) > config.physical_limits.max_abs_yaw
    )
    raw_stuck = (
        jp.asarray(False)
        if (
            config.events.stuck_window_steps is None
            and config.events.stuck_forward_velocity_threshold is None
        )
        else jp.asarray(inputs.stuck, dtype=bool)
    )
    horizon = inputs.episode_step >= config.ppo.episode_horizon - 1
    physical_failure = jp.asarray(False)
    for condition in physical_failures:
        physical_failure = physical_failure | condition
    jump_zone_missed = jp.asarray(config.schema.endswith("_v4"), dtype=bool) & (
        ~jp.asarray(inputs.jump_zone_seen, dtype=bool) & (physical_failure | horizon)
    )
    stuck = raw_stuck & ~physical_failure
    yaw_limit = raw_yaw_limit & ~physical_failure & ~stuck
    failures = physical_failures + (jump_zone_missed, stuck, yaw_limit)
    failure_codes = (
        END_NONFINITE,
        END_ROLL_LIMIT,
        END_PITCH_LIMIT,
        END_PROHIBITED_CONTACT,
        END_BACKWARD_EXIT,
        END_JUMP_ZONE_MISSED,
        END_STUCK,
        END_YAW_LIMIT,
    )
    terminated = physical_failure | jump_zone_missed | stuck | yaw_limit
    failure_code = jp.asarray(END_ONGOING, jp.int32)
    for condition, code in reversed(tuple(zip(failures, failure_codes, strict=True))):
        failure_code = jp.where(condition, jp.asarray(code, jp.int32), failure_code)
    success = jp.asarray(False)
    timeout = horizon & ~terminated
    end_code = jp.where(timeout, END_TIMEOUT, END_ONGOING)
    end_code = jp.where(terminated, failure_code, end_code).astype(jp.int32)
    return TerminalState(
        terminated=terminated,
        truncated=timeout,
        success=success,
        physical_failure=physical_failure,
        roll_limit=roll_failure,
        pitch_limit=pitch_failure,
        jump_zone_missed=jump_zone_missed,
        stuck=stuck,
        yaw_limit=yaw_limit,
        timeout=timeout,
        end_code=end_code,
    )
