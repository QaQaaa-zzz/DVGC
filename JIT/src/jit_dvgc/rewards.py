"""Bounded, component-wise Propulsion-Ascent reward."""

from __future__ import annotations

from flax import struct
import jax
from jax import numpy as jp

from .config import ApexConfig, PhysicalLimits, RewardConfig
from .constants import CTRL_DT, REWARD_COMPONENT_KEYS


@struct.dataclass
class RewardState:
    x: jax.Array
    z: jax.Array
    clearance: jax.Array
    roll: jax.Array
    pitch: jax.Array
    angular_speed: jax.Array
    vertical_velocity: jax.Array
    forward_velocity: jax.Array
    obstacle_relative_x: jax.Array


@struct.dataclass
class RewardInputs:
    previous: RewardState
    current: RewardState
    action: jax.Array
    last_action: jax.Array
    window_latched: jax.Array
    first_window_entry: jax.Array
    first_liftoff: jax.Array
    first_stable_airborne: jax.Array
    stable_airborne: jax.Array
    ascending_seen: jax.Array
    first_apex_success: jax.Array
    illegal_contact: jax.Array
    physical_failure_transition: jax.Array
    timeout_transition: jax.Array


@struct.dataclass
class RewardComponents:
    drive: jax.Array
    window: jax.Array
    liftoff: jax.Array
    stable_airborne: jax.Array
    ascent: jax.Array
    clearance: jax.Array
    apex_progress: jax.Array
    apex_success: jax.Array
    attitude: jax.Array
    rate: jax.Array
    smoothness: jax.Array
    action_magnitude: jax.Array
    illegal_contact: jax.Array
    physical_failure: jax.Array
    timeout: jax.Array

    def __getitem__(self, key: str) -> jax.Array:
        if key not in REWARD_COMPONENT_KEYS:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self):
        return iter(REWARD_COMPONENT_KEYS)

    def values(self) -> tuple[jax.Array, ...]:
        return tuple(getattr(self, key) for key in REWARD_COMPONENT_KEYS)

    def as_dict(self) -> dict[str, jax.Array]:
        return {key: getattr(self, key) for key in REWARD_COMPONENT_KEYS}


@struct.dataclass
class RewardResult:
    total: jax.Array
    components: RewardComponents


def _quality(state: RewardState) -> jax.Array:
    pose = jp.exp(-jp.square(state.roll / 0.25) - jp.square(state.pitch / 0.35))
    rate = jp.exp(-jp.square(state.angular_speed / 4.0))
    return pose * rate


def _apex_score(state: RewardState, apex: ApexConfig) -> jax.Array:
    vertical = jp.clip(
        1.0 - jp.abs(state.vertical_velocity) / apex.max_abs_vertical_velocity,
        0.0,
        1.0,
    )
    clearance = jp.clip(state.clearance / apex.min_clearance, 0.0, 1.0)
    roll = jp.clip(1.0 - jp.abs(state.roll) / apex.max_abs_roll, 0.0, 1.0)
    pitch = jp.clip(1.0 - jp.abs(state.pitch) / apex.max_abs_pitch, 0.0, 1.0)
    angular = jp.clip(
        1.0 - state.angular_speed / apex.max_angular_speed, 0.0, 1.0
    )
    forward = jp.clip(state.forward_velocity / apex.min_forward_velocity, 0.0, 1.0)
    center = 0.5 * (apex.relative_x_min + apex.relative_x_max)
    half_width = 0.5 * (apex.relative_x_max - apex.relative_x_min)
    relative_x = jp.clip(
        1.0 - jp.abs(state.obstacle_relative_x - center) / half_width,
        0.0,
        1.0,
    )
    return jp.mean(
        jp.stack((vertical, clearance, roll, pitch, angular, forward, relative_x))
    )


def phase_u_reward(
    inputs: RewardInputs,
    config: RewardConfig,
    apex: ApexConfig,
    physical_limits: PhysicalLimits,
) -> RewardResult:
    current = inputs.current
    previous = inputs.previous
    window = jp.asarray(inputs.window_latched, dtype=bool)
    stable = window & jp.asarray(inputs.stable_airborne, dtype=bool)
    motion_quality = _quality(current)

    drive = config.drive_weight * jp.clip(
        (current.x - previous.x) / (config.target_forward_velocity * CTRL_DT),
        0.0,
        1.0,
    )
    window_reward = config.window_bonus * jp.asarray(
        inputs.first_window_entry, dtype=jp.float32
    )
    liftoff = (
        config.liftoff_bonus
        * jp.asarray(window & jp.asarray(inputs.first_liftoff, dtype=bool), jp.float32)
        * motion_quality
    )
    stable_airborne = (
        config.stable_airborne_bonus
        * jp.asarray(
            window & jp.asarray(inputs.first_stable_airborne, dtype=bool),
            jp.float32,
        )
        * motion_quality
    )
    ascent = (
        config.ascent_weight
        * jp.asarray(stable, jp.float32)
        * jp.clip((current.z - previous.z) / 0.02, 0.0, 1.0)
        * motion_quality
    )
    clearance = (
        config.clearance_weight
        * jp.asarray(stable, jp.float32)
        * jp.clip((current.clearance - previous.clearance) / 0.02, 0.0, 1.0)
        * motion_quality
    )
    apex_progress = (
        config.apex_progress_weight
        * jp.asarray(
            stable & jp.asarray(inputs.ascending_seen, dtype=bool), jp.float32
        )
        * jp.clip(_apex_score(current, apex) - _apex_score(previous, apex), 0.0, 1.0)
    )
    apex_success = (
        config.apex_success_bonus
        * jp.asarray(
            window & jp.asarray(inputs.first_apex_success, dtype=bool), jp.float32
        )
    )

    attitude = -config.attitude_penalty_weight * jp.clip(
        jp.square(jp.abs(current.roll) / physical_limits.max_abs_roll)
        + jp.square(jp.abs(current.pitch) / physical_limits.max_abs_pitch),
        0.0,
        8.0,
    )
    rate = -config.rate_penalty_weight * jp.clip(
        jp.square(current.angular_speed / apex.max_angular_speed), 0.0, 16.0
    )
    smoothness = -config.action_smoothness_weight * jp.mean(
        jp.square(inputs.action - inputs.last_action)
    )
    action_magnitude = -config.action_magnitude_weight * jp.mean(
        jp.square(inputs.action)
    )
    illegal_contact = -config.illegal_contact_penalty * jp.asarray(
        inputs.illegal_contact, jp.float32
    )
    physical_failure = -config.physical_failure_penalty * jp.asarray(
        inputs.physical_failure_transition, jp.float32
    )
    timeout = -config.timeout_penalty * jp.asarray(
        inputs.timeout_transition, jp.float32
    )

    components = RewardComponents(
        drive=drive,
        window=window_reward,
        liftoff=liftoff,
        stable_airborne=stable_airborne,
        ascent=ascent,
        clearance=clearance,
        apex_progress=apex_progress,
        apex_success=apex_success,
        attitude=attitude,
        rate=rate,
        smoothness=smoothness,
        action_magnitude=action_magnitude,
        illegal_contact=illegal_contact,
        physical_failure=physical_failure,
        timeout=timeout,
    )
    total = jp.clip(
        sum(components.values(), start=jp.asarray(0.0, jp.float32)),
        config.total_min,
        config.total_max,
    )
    return RewardResult(total=total, components=components)
