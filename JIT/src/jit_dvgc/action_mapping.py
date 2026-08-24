"""Pure JAX normalized-action to authoritative actuator-target mapping."""

from __future__ import annotations

from flax import struct
import jax
from jax import numpy as jp


@struct.dataclass
class ActionMapping:
    ctrl_min: jax.Array
    ctrl_max: jax.Array
    hip_initial: float = struct.field(pytree_node=False)
    base_rear_speed: float = struct.field(pytree_node=False)
    rear_speed_delta: float = struct.field(pytree_node=False)
    knee_target_delta: float = struct.field(pytree_node=False)


def map_action(
    action: jax.Array,
    knee_position: jax.Array,
    mapping: ActionMapping,
) -> jax.Array:
    """Map `[steer, drive, hip, knee]` to position/velocity targets."""
    normalized = jp.clip(jp.asarray(action), -1.0, 1.0)
    steer = normalized[0] * mapping.ctrl_max[0]
    rear = jp.clip(
        mapping.base_rear_speed + normalized[1] * mapping.rear_speed_delta,
        mapping.ctrl_min[1],
        mapping.ctrl_max[1],
    )
    hip = jp.where(
        normalized[2] >= 0.0,
        mapping.hip_initial
        + normalized[2] * (mapping.ctrl_max[2] - mapping.hip_initial),
        mapping.hip_initial
        + normalized[2] * (mapping.hip_initial - mapping.ctrl_min[2]),
    )
    knee = jp.clip(
        knee_position - normalized[3] * mapping.knee_target_delta,
        mapping.ctrl_min[3],
        mapping.ctrl_max[3],
    )
    ctrl = jp.stack((steer, rear, hip, knee))
    return jp.clip(ctrl, mapping.ctrl_min, mapping.ctrl_max)
