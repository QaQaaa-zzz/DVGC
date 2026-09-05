"""Pure JAX contact and short-recovery semantics for Phase D."""
from __future__ import annotations
from flax import struct
import jax
from jax import numpy as jp
from .config import DescentConfig, PhysicalLimits
from .constants import END_BACKWARD_EXIT, END_NONFINITE, END_PITCH_LIMIT, END_PROHIBITED_CONTACT, END_RECOVERY_SUCCESS, END_ROLL_LIMIT, END_TIMEOUT, END_ONGOING

@struct.dataclass
class DescentSignals:
    x: jax.Array; front_clearance: jax.Array; rear_clearance: jax.Array
    maximum_wheel_penetration: jax.Array; body_contact: jax.Array; finite: jax.Array
    roll: jax.Array; pitch: jax.Array; backward_exit: jax.Array

@struct.dataclass
class DescentEventState:
    airborne_seen: jax.Array; valid_contact_seen: jax.Array; contact_x: jax.Array
    post_contact_ticks: jax.Array; recovery_success: jax.Array

def initial_descent_events(x: jax.Array) -> DescentEventState:
    return DescentEventState(jp.asarray(False), jp.asarray(False), jp.asarray(x), jp.asarray(0, jp.int32), jp.asarray(False))

def advance_descent_events(previous: DescentEventState, signals: DescentSignals, config: DescentConfig) -> DescentEventState:
    airborne = previous.airborne_seen | ((signals.front_clearance > config.min_airborne_clearance) & (signals.rear_clearance > config.min_airborne_clearance))
    contact = (~previous.valid_contact_seen & airborne & ((signals.front_clearance <= config.contact_clearance_threshold) | (signals.rear_clearance <= config.contact_clearance_threshold)) & (signals.maximum_wheel_penetration <= config.max_wheel_penetration) & ~signals.body_contact & signals.finite)
    seen = previous.valid_contact_seen | contact
    contact_x = jp.where(contact, signals.x, previous.contact_x)
    ticks = jp.where(seen & signals.finite, previous.post_contact_ticks + 1, previous.post_contact_ticks)
    success = previous.recovery_success | (seen & (ticks >= config.recovery_ticks) & (signals.x - contact_x + jp.asarray(1e-6, jp.float32) >= config.min_post_contact_forward_progress))
    return DescentEventState(airborne, seen, contact_x, ticks, success)

@struct.dataclass
class DescentTerminal:
    terminated: jax.Array; truncated: jax.Array; success: jax.Array
    physical_failure: jax.Array; timeout: jax.Array; end_code: jax.Array

def classify_descent_terminal(signals: DescentSignals, events: DescentEventState, config: DescentConfig, limits: PhysicalLimits, episode_step: jax.Array, episode_horizon: int) -> DescentTerminal:
    nonfinite = ~signals.finite; roll = jp.abs(signals.roll) > limits.max_abs_roll; pitch = jp.abs(signals.pitch) > limits.max_abs_pitch
    body = signals.body_contact & config.terminate_on_body_contact
    physical = nonfinite | roll | pitch | body | signals.backward_exit
    timeout = (episode_step >= episode_horizon - 1) & ~physical
    success = events.recovery_success & ~physical & ~timeout
    code = jp.where(nonfinite, END_NONFINITE, jp.where(roll, END_ROLL_LIMIT, jp.where(pitch, END_PITCH_LIMIT, jp.where(body, END_PROHIBITED_CONTACT, END_BACKWARD_EXIT))))
    return DescentTerminal(physical | success, timeout, success, physical, timeout, jp.where(physical, code, jp.where(success, END_RECOVERY_SUCCESS, jp.where(timeout, END_TIMEOUT, END_ONGOING))))
