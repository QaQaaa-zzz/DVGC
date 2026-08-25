"""Target-free reference reward for the Propulsion-Ascent task."""

from __future__ import annotations

from flax import struct
import jax
from jax import numpy as jp

from .config import PhysicalLimits, RewardConfig
from .constants import CTRL_DT, REWARD_COMPONENT_KEYS


@struct.dataclass
class RewardState:
    x: jax.Array
    y: jax.Array
    z: jax.Array
    roll: jax.Array
    pitch: jax.Array
    yaw: jax.Array
    forward_velocity: jax.Array
    lateral_velocity: jax.Array
    vertical_velocity: jax.Array
    roll_rate: jax.Array
    pitch_rate: jax.Array
    yaw_rate: jax.Array
    hip_velocity: jax.Array
    knee_velocity: jax.Array
    hip_force: jax.Array
    knee_force: jax.Array


@struct.dataclass
class RewardInputs:
    current: RewardState
    action: jax.Array
    last_action: jax.Array
    jump_signal: jax.Array
    first_apex_success: jax.Array
    illegal_contact: jax.Array
    physical_failure_transition: jax.Array
    timeout_transition: jax.Array


@struct.dataclass
class RewardComponents:
    roll: jax.Array
    pitch: jax.Array
    yaw: jax.Array
    speed: jax.Array
    survival: jax.Array
    height: jax.Array
    action_smoothness: jax.Array
    action_magnitude: jax.Array
    roll_rate: jax.Array
    pitch_rate: jax.Array
    yaw_rate: jax.Array
    joint_energy: jax.Array
    apex_success: jax.Array
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
    unclipped_total: jax.Array
    components: RewardComponents


def _roll_raw(degrees: jax.Array) -> jax.Array:
    return jp.where(degrees <= 5.0, 1.0 - degrees / 5.0, -0.1 * (degrees - 5.0))


def _pitch_raw(degrees: jax.Array) -> jax.Array:
    below_three = 1.0 - (degrees / 3.0) * 0.1
    below_eight = 0.9 - ((degrees - 3.0) / 5.0) * 0.4
    below_ten = 0.5 - ((degrees - 8.0) / 2.0) * 0.5
    above_ten = -0.1 * (degrees - 10.0)
    return jp.where(
        degrees <= 3.0,
        below_three,
        jp.where(degrees <= 8.0, below_eight, jp.where(degrees <= 10.0, below_ten, above_ten)),
    )


def _yaw_raw(degrees: jax.Array) -> jax.Array:
    below_three = 1.0 - (degrees / 3.0) * 0.1
    below_eight = 0.9 - ((degrees - 3.0) / 5.0) * 0.4
    below_fifteen = 0.5 - ((degrees - 8.0) / 7.0) * 0.3
    below_twenty_five = 0.2 - ((degrees - 15.0) / 10.0) * 0.15
    above_twenty_five = jp.maximum(0.05 - (degrees - 25.0) * 0.002, 0.0)
    return jp.where(
        degrees <= 3.0,
        below_three,
        jp.where(
            degrees <= 8.0,
            below_eight,
            jp.where(
                degrees <= 15.0,
                below_fifteen,
                jp.where(degrees <= 25.0, below_twenty_five, above_twenty_five),
            ),
        ),
    )


def _height_raw(z: jax.Array, config: RewardConfig) -> jax.Array:
    rising = 1.0 + (
        (z - config.jump_reward_min_height)
        / (config.peak_reward_height - config.jump_reward_min_height)
    ) * 0.5
    excess_ratio = (z - config.peak_reward_height) / (
        config.max_beneficial_height - config.peak_reward_height
    )
    descending = 1.5 * (1.0 - excess_ratio * 0.6)
    shaped = jp.where(
        z <= config.peak_reward_height,
        rising,
        jp.where(z <= config.max_beneficial_height, descending, 0.4),
    )
    return jp.where(z >= config.jump_reward_min_height, shaped, 0.0)


def phase_u_reward(
    inputs: RewardInputs,
    config: RewardConfig,
    physical_limits: PhysicalLimits,
) -> RewardResult:
    del physical_limits
    state = inputs.current
    radians_to_degrees = 180.0 / jp.pi
    roll = config.roll_coeff * _roll_raw(jp.abs(state.roll) * radians_to_degrees)
    pitch = config.pitch_coeff * _pitch_raw(jp.abs(state.pitch) * radians_to_degrees)
    yaw = config.yaw_coeff * _yaw_raw(jp.abs(state.yaw) * radians_to_degrees)
    speed = config.speed_coeff * jp.exp(
        -0.5 * jp.square((state.forward_velocity - config.desired_velocity) / config.speed_sigma)
    )
    survival = jp.asarray(config.survival_reward, jp.float32)
    height = (
        config.height_coeff
        * _height_raw(state.z, config)
        * jp.asarray(inputs.jump_signal, jp.float32)
    )
    action_smoothness = -config.action_coeff * config.action_smoothness_scale * jp.sum(
        jp.square(inputs.action - inputs.last_action)
    )
    action_magnitude = -config.action_coeff * config.action_magnitude_scale * jp.sum(
        jp.power(jp.abs(inputs.action), 1.5)
    )
    zero = jp.asarray(0.0, jp.float32)
    pitch_rate = -config.pitch_angular_velocity_coeff * 0.125 * jp.square(
        state.pitch_rate
    )
    mechanical_energy = CTRL_DT * (
        jp.abs(state.hip_force * state.hip_velocity)
        + jp.abs(state.knee_force * state.knee_velocity)
    )
    joint_energy = -config.joint_energy_penalty_coeff * mechanical_energy
    apex_success = config.apex_success_bonus * jp.asarray(
        inputs.first_apex_success, jp.float32
    )
    illegal_contact = -config.illegal_contact_penalty * jp.asarray(
        inputs.illegal_contact, jp.float32
    )
    physical_failure = -config.physical_failure_penalty * jp.asarray(
        inputs.physical_failure_transition, jp.float32
    )
    timeout = -config.timeout_penalty * jp.asarray(inputs.timeout_transition, jp.float32)
    components = RewardComponents(
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        speed=speed,
        survival=survival,
        height=height,
        action_smoothness=action_smoothness,
        action_magnitude=action_magnitude,
        roll_rate=zero,
        pitch_rate=pitch_rate,
        yaw_rate=zero,
        joint_energy=joint_energy,
        apex_success=apex_success,
        illegal_contact=illegal_contact,
        physical_failure=physical_failure,
        timeout=timeout,
    )
    unclipped = sum(components.values(), start=jp.asarray(0.0, jp.float32))
    return RewardResult(
        total=jp.clip(unclipped, config.total_min, config.total_max),
        unclipped_total=unclipped,
        components=components,
    )
