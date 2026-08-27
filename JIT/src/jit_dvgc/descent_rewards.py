"""Independent Phase D reward calculation; no reference or learned-score terms."""
from __future__ import annotations
from flax import struct
import jax
from jax import numpy as jp
from .config import DescentConfig
from .rewards import roll_raw, pitch_raw

@struct.dataclass
class DescentRewardInputs:
    x_delta: jax.Array
    valid_contact: jax.Array
    previous_valid_contact: jax.Array
    post_contact: jax.Array
    recovery_success: jax.Array
    previous_recovery_success: jax.Array
    bad_contact: jax.Array
    physical_failure: jax.Array
    timeout: jax.Array
    roll: jax.Array
    pitch: jax.Array
    roll_rate: jax.Array
    pitch_rate: jax.Array
    action: jax.Array
    last_action: jax.Array

@struct.dataclass
class DescentRewardComponents:
    roll_posture: jax.Array
    pitch_posture: jax.Array
    roll_rate: jax.Array
    pitch_rate: jax.Array
    action_smoothness: jax.Array
    forward_progress: jax.Array
    contact: jax.Array
    recovery_tick: jax.Array
    success: jax.Array
    bad_contact: jax.Array
    failure: jax.Array
    timeout: jax.Array

@struct.dataclass
class DescentRewardResult:
    total: jax.Array
    components: DescentRewardComponents

def descent_recovery_reward(inputs: DescentRewardInputs, config: DescentConfig) -> DescentRewardResult:
    dtype = jp.float32
    progress = jp.maximum(jp.asarray(inputs.x_delta, dtype), 0.0) * config.reward_forward_progress
    contact = jp.asarray(inputs.valid_contact & ~inputs.previous_valid_contact, dtype).astype(dtype) * config.reward_contact
    tick = jp.asarray(inputs.post_contact & ~inputs.physical_failure, dtype).astype(dtype) * config.reward_recovery_tick
    success = jp.asarray(inputs.recovery_success & ~inputs.previous_recovery_success, dtype).astype(dtype) * config.reward_success
    bad = jp.asarray(inputs.bad_contact, dtype).astype(dtype) * -config.penalty_bad_contact
    failure = jp.asarray(inputs.physical_failure, dtype).astype(dtype) * -config.penalty_failure
    timeout = jp.asarray(inputs.timeout & ~inputs.physical_failure, dtype).astype(dtype) * -config.penalty_timeout
    radians_to_degrees = 180.0 / jp.pi
    roll_posture = config.roll_posture_coeff * roll_raw(jp.abs(inputs.roll) * radians_to_degrees)
    pitch_posture = config.pitch_posture_coeff * pitch_raw(jp.abs(inputs.pitch) * radians_to_degrees)
    roll_rate = -config.roll_rate_penalty_coeff * 0.125 * jp.square(inputs.roll_rate)
    pitch_rate = -config.pitch_rate_penalty_coeff * 0.125 * jp.square(inputs.pitch_rate)
    action_smoothness = -config.action_smoothness_penalty_coeff * jp.sum(jp.square(inputs.action - inputs.last_action))
    components = DescentRewardComponents(roll_posture, pitch_posture, roll_rate, pitch_rate, action_smoothness, progress, contact, tick, success, bad, failure, timeout)
    total = sum(components.__dict__.values(), jp.asarray(0., dtype))
    total = jp.where(jp.isfinite(total), total, jp.asarray(0., dtype))
    return DescentRewardResult(total=jp.clip(total, config.total_min, config.total_max), components=components)
