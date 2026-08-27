"""Independent Phase D reward calculation; no reference or learned-score terms."""
from __future__ import annotations
from flax import struct
import jax
from jax import numpy as jp
from .config import DescentConfig

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

@struct.dataclass
class DescentRewardComponents:
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
    components = DescentRewardComponents(progress, contact, tick, success, bad, failure, timeout)
    total = sum((components.forward_progress, components.contact, components.recovery_tick, components.success, components.bad_contact, components.failure, components.timeout), jp.asarray(0., dtype))
    return DescentRewardResult(total=jp.where(jp.isfinite(total), total, jp.asarray(0., dtype)), components=components)
