"""JAX-compatible full collision-structure geometry and contact signals."""

from __future__ import annotations

from flax import struct
import jax
from jax import numpy as jp
import mujoco
import numpy as np


GEOM_SPHERE = int(mujoco.mjtGeom.mjGEOM_SPHERE)
GEOM_CAPSULE = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
GEOM_ELLIPSOID = int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)
GEOM_CYLINDER = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
GEOM_BOX = int(mujoco.mjtGeom.mjGEOM_BOX)
SUPPORTED_GEOM_TYPES = {
    GEOM_SPHERE,
    GEOM_CAPSULE,
    GEOM_ELLIPSOID,
    GEOM_CYLINDER,
    GEOM_BOX,
}


@struct.dataclass
class CollisionSupportBounds:
    min_x: jax.Array
    max_x: jax.Array
    min_y: jax.Array
    max_y: jax.Array
    min_z: jax.Array
    max_z: jax.Array


@struct.dataclass
class StructureMetrics:
    robot_frontmost_x: jax.Array
    obstacle_relative_x: jax.Array
    full_structure_clearance: jax.Array


@struct.dataclass
class ContactSignals:
    front_wheel_terrain_clearance: jax.Array
    rear_wheel_terrain_clearance: jax.Array
    maximum_wheel_penetration: jax.Array
    prohibited_contact: jax.Array


@struct.dataclass
class GeometrySignals:
    robot_frontmost_x: jax.Array
    obstacle_relative_x: jax.Array
    structure_clearance: jax.Array
    front_wheel_terrain_clearance: jax.Array
    rear_wheel_terrain_clearance: jax.Array
    maximum_wheel_penetration: jax.Array
    prohibited_contact: jax.Array
    roll: jax.Array
    pitch: jax.Array
    yaw: jax.Array
    angular_speed: jax.Array
    forward_velocity: jax.Array
    vertical_velocity: jax.Array


@struct.dataclass
class GeometryContract:
    robot_geom_ids: jax.Array
    robot_geom_types: jax.Array
    robot_geom_sizes: jax.Array
    wheel_geom_ids: jax.Array
    body_geom_ids: jax.Array
    wheel_geom_positions: jax.Array
    body_geom_positions: jax.Array
    floor_geom_id: int = struct.field(pytree_node=False)
    obstacle_geom_id: int = struct.field(pytree_node=False)
    obstacle_front_x: float = struct.field(pytree_node=False)
    obstacle_back_x: float = struct.field(pytree_node=False)
    obstacle_top_z: float = struct.field(pytree_node=False)
    obstacle_half_width: float = struct.field(pytree_node=False)
    robot_geom_names: tuple[str, ...] = struct.field(pytree_node=False)


def _support_radius(
    local_direction: jax.Array, geom_type: jax.Array, size: jax.Array
) -> jax.Array:
    sphere = size[..., 0]
    capsule = size[..., 1] * jp.abs(local_direction[..., 2]) + size[..., 0]
    ellipsoid = jp.sqrt(jp.sum(jp.square(size * local_direction), axis=-1))
    radial = jp.sqrt(jp.sum(jp.square(local_direction[..., :2]), axis=-1))
    cylinder = size[..., 1] * jp.abs(local_direction[..., 2]) + size[..., 0] * radial
    box = jp.sum(size * jp.abs(local_direction), axis=-1)
    result = jp.where(geom_type == GEOM_SPHERE, sphere, jp.inf)
    result = jp.where(geom_type == GEOM_CAPSULE, capsule, result)
    result = jp.where(geom_type == GEOM_ELLIPSOID, ellipsoid, result)
    result = jp.where(geom_type == GEOM_CYLINDER, cylinder, result)
    return jp.where(geom_type == GEOM_BOX, box, result)


def collision_support_bounds(
    geom_xpos: jax.Array,
    geom_xmat: jax.Array,
    geom_types: jax.Array,
    geom_sizes: jax.Array,
) -> CollisionSupportBounds:
    positions = jp.asarray(geom_xpos)
    rotations = jp.asarray(geom_xmat).reshape(positions.shape[:-1] + (3, 3))
    local_x = rotations[..., 0, :]
    local_y = rotations[..., 1, :]
    local_z = rotations[..., 2, :]
    support_x = _support_radius(local_x, jp.asarray(geom_types), jp.asarray(geom_sizes))
    support_y = _support_radius(local_y, jp.asarray(geom_types), jp.asarray(geom_sizes))
    support_z = _support_radius(local_z, jp.asarray(geom_types), jp.asarray(geom_sizes))
    return CollisionSupportBounds(
        min_x=positions[..., 0] - support_x,
        max_x=positions[..., 0] + support_x,
        min_y=positions[..., 1] - support_y,
        max_y=positions[..., 1] + support_y,
        min_z=positions[..., 2] - support_z,
        max_z=positions[..., 2] + support_z,
    )


def full_structure_metrics(
    geom_xpos: jax.Array,
    geom_xmat: jax.Array,
    geom_types: jax.Array,
    geom_sizes: jax.Array,
    *,
    obstacle_front_x: float,
    obstacle_top_z: float,
) -> StructureMetrics:
    bounds = collision_support_bounds(geom_xpos, geom_xmat, geom_types, geom_sizes)
    frontmost = jp.max(bounds.max_x, axis=-1)
    lowest = jp.min(bounds.min_z, axis=-1)
    return StructureMetrics(
        robot_frontmost_x=frontmost,
        obstacle_relative_x=jp.asarray(obstacle_front_x) - frontmost,
        full_structure_clearance=lowest - jp.asarray(obstacle_top_z),
    )


def _name(model: mujoco.MjModel, object_type: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, object_type, index) or f"unnamed_{index}"


def build_geometry_contract(model: mujoco.MjModel) -> GeometryContract:
    collision = (np.asarray(model.geom_contype) != 0) | (
        np.asarray(model.geom_conaffinity) != 0
    )
    body_ids = np.asarray(model.geom_bodyid, dtype=np.int32)
    robot_ids = np.flatnonzero(collision & (body_ids != 0)).astype(np.int32)
    unsupported = [
        int(index)
        for index in robot_ids
        if int(model.geom_type[index]) not in SUPPORTED_GEOM_TYPES
    ]
    if unsupported:
        names = [_name(model, mujoco.mjtObj.mjOBJ_GEOM, index) for index in unsupported]
        raise ValueError(f"unsupported collision robot geoms: {names}")
    front = int(model.geom("frontwheel_collision").id)
    rear = int(model.geom("rearwheel_collision").id)
    wheel_ids = np.asarray((front, rear), dtype=np.int32)
    body_geom_ids = np.asarray(
        [index for index in robot_ids if int(index) not in {front, rear}],
        dtype=np.int32,
    )
    wheel_positions = np.asarray(
        [
            int(np.flatnonzero(robot_ids == front)[0]),
            int(np.flatnonzero(robot_ids == rear)[0]),
        ],
        dtype=np.int32,
    )
    body_positions = np.asarray(
        [
            position
            for position, geom_id in enumerate(robot_ids)
            if int(geom_id) not in {front, rear}
        ],
        dtype=np.int32,
    )
    obstacle_id = int(model.geom("step").id)
    obstacle_pos = np.asarray(model.geom_pos[obstacle_id], dtype=np.float64)
    obstacle_size = np.asarray(model.geom_size[obstacle_id], dtype=np.float64)
    names = tuple(
        _name(model, mujoco.mjtObj.mjOBJ_GEOM, int(index)) for index in robot_ids
    )
    return GeometryContract(
        robot_geom_ids=jp.asarray(robot_ids),
        robot_geom_types=jp.asarray(model.geom_type[robot_ids], dtype=jp.int32),
        robot_geom_sizes=jp.asarray(model.geom_size[robot_ids], dtype=jp.float32),
        wheel_geom_ids=jp.asarray(wheel_ids),
        body_geom_ids=jp.asarray(body_geom_ids),
        wheel_geom_positions=jp.asarray(wheel_positions),
        body_geom_positions=jp.asarray(body_positions),
        floor_geom_id=int(model.geom("floor").id),
        obstacle_geom_id=obstacle_id,
        obstacle_front_x=float(obstacle_pos[0] - obstacle_size[0]),
        obstacle_back_x=float(obstacle_pos[0] + obstacle_size[0]),
        obstacle_top_z=float(obstacle_pos[2] + obstacle_size[2]),
        obstacle_half_width=float(obstacle_size[1]),
        robot_geom_names=names,
    )


def geometric_penetration_signals(
    bounds: CollisionSupportBounds,
    contract: GeometryContract,
) -> ContactSignals:
    overlaps_obstacle = (
        (bounds.max_x >= contract.obstacle_front_x)
        & (bounds.min_x <= contract.obstacle_back_x)
        & (bounds.max_y >= -contract.obstacle_half_width)
        & (bounds.min_y <= contract.obstacle_half_width)
    )
    terrain_height = jp.where(overlaps_obstacle, contract.obstacle_top_z, 0.0)
    clearances = bounds.min_z - terrain_height
    wheel_clearances = jp.take(clearances, contract.wheel_geom_positions)
    body_clearances = jp.take(clearances, contract.body_geom_positions)
    prohibited = jp.any(body_clearances < -0.002)
    return ContactSignals(
        front_wheel_terrain_clearance=wheel_clearances[0],
        rear_wheel_terrain_clearance=wheel_clearances[1],
        maximum_wheel_penetration=jp.maximum(-jp.min(wheel_clearances), 0.0),
        prohibited_contact=prohibited,
    )


def quaternion_to_euler(quaternion: jax.Array) -> tuple[jax.Array, ...]:
    w, x, y, z = quaternion
    roll = jp.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = jp.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = jp.arcsin(sin_pitch)
    yaw = jp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def extract_geometry(data, contract: GeometryContract) -> GeometrySignals:
    ids = contract.robot_geom_ids
    positions = jp.take(data.geom_xpos, ids, axis=-2)
    rotations = data.geom_xmat.reshape((-1, 3, 3))
    rotations = jp.take(rotations, ids, axis=-3)
    bounds = collision_support_bounds(
        positions, rotations, contract.robot_geom_types, contract.robot_geom_sizes
    )
    frontmost = jp.max(bounds.max_x, axis=-1)
    structure = StructureMetrics(
        robot_frontmost_x=frontmost,
        obstacle_relative_x=contract.obstacle_front_x - frontmost,
        full_structure_clearance=jp.min(bounds.min_z, axis=-1)
        - contract.obstacle_top_z,
    )
    contacts = geometric_penetration_signals(bounds, contract)
    roll, pitch, yaw = quaternion_to_euler(data.qpos[3:7])
    return GeometrySignals(
        robot_frontmost_x=structure.robot_frontmost_x,
        obstacle_relative_x=structure.obstacle_relative_x,
        structure_clearance=structure.full_structure_clearance,
        front_wheel_terrain_clearance=contacts.front_wheel_terrain_clearance,
        rear_wheel_terrain_clearance=contacts.rear_wheel_terrain_clearance,
        maximum_wheel_penetration=contacts.maximum_wheel_penetration,
        prohibited_contact=contacts.prohibited_contact,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        angular_speed=jp.linalg.norm(data.qvel[3:6]),
        forward_velocity=data.qvel[0],
        vertical_velocity=data.qvel[2],
    )
