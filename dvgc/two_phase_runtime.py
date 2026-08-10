"""Pure-JAX physical signal extraction for the approved two-phase method.

The fixed horizontal sign convention is::

    obstacle_relative_x = obstacle_front_x - robot_frontmost_x

Positive values are before the obstacle front, zero is aligned, and negative
values have passed it.  All robot support bounds cover every collision-relevant
robot geom; root or CoM position is never substituted for structure geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
import hashlib
import json
from typing import Any, NamedTuple

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from .two_phase_semantics import (
    INTERNAL_EVENTS,
    ApexBandSignals,
    ApexBandThresholds,
    RecoverySignals,
    RecoveryThresholds,
    advance_recovery_hold_count,
    apex_band_membership,
    descent_recovery_success,
)


for _signal_type in (ApexBandSignals, RecoverySignals):
    try:
        jax.tree_util.register_dataclass(
            _signal_type,
            data_fields=[field.name for field in dataclass_fields(_signal_type)],
            meta_fields=[],
        )
    except ValueError:
        pass


_GEOM_FORMULAS = {
    int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere_support",
    int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule_support",
    int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid_support",
    int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder_support",
    int(mujoco.mjtGeom.mjGEOM_BOX): "box_support",
}
EVENT_NAMES = INTERNAL_EVENTS

# Stable dvgc.env terminal-code contract.  Successful recovery, timeout, and
# legacy handoff terminals are deliberately excluded.
# Prelaunch airborne (9) is telemetry only.  Takeoff task deadlines (10-13)
# are adapter-owned failures after the legal jump latch, not global physical
# failures.  Keep only immutable physical-safety terminals here.
_PHYSICAL_FAILURE_END_CODES = (2, 3, 4, 5, 6, 7, 15)


class CollisionSupportBounds(NamedTuple):
    min_x: Any
    max_x: Any
    min_y: Any
    max_y: Any
    min_z: Any
    max_z: Any


class StructureMetrics(NamedTuple):
    robot_frontmost_x: Any
    obstacle_relative_x: Any
    full_structure_clearance: Any


@dataclass(frozen=True)
class TwoPhaseThresholds:
    apex: ApexBandThresholds
    recovery: RecoveryThresholds


class TwoPhaseEventState(NamedTuple):
    jump_window_entered: Any
    liftoff_seen: Any
    stable_airborne: Any
    ascending: Any
    apex_band_entered: Any
    descending: Any
    pre_landing: Any
    first_valid_contact: Any
    impact_absorbing: Any
    stable_recovery: Any
    first_event_ticks: Any
    previous_com_vz: Any
    previous_stable_wheel_support: Any
    recovery_hold_count: Any
    apex_band_width: Any
    max_apex_band_width: Any


@dataclass(frozen=True)
class TwoPhaseGeometry:
    robot_geom_ids: np.ndarray
    robot_geom_types: Any
    robot_geom_sizes: Any
    robot_geom_body_ids: np.ndarray
    wheel_mask: Any
    body_mask: Any
    obstacle_geom_id: int
    obstacle_front_x: float
    obstacle_back_x: float
    obstacle_top_z: float
    obstacle_half_width: float
    landing_x_min: float
    landing_x_max: float
    landing_y_limit: float
    root_qpos_adr: int
    root_dof_adr: int
    airborne_confirm_steps: int
    support_tolerance: float
    body_penetration_tolerance: float


def _support_radius(local_direction: Any, geom_type: Any, size: Any) -> Any:
    """Return analytic support radius along one unit world direction."""
    sphere = size[..., 0]
    capsule = size[..., 1] * jp.abs(local_direction[..., 2]) + size[..., 0]
    ellipsoid = jp.sqrt(jp.sum(jp.square(size * local_direction), axis=-1))
    radial = jp.sqrt(jp.sum(jp.square(local_direction[..., :2]), axis=-1))
    cylinder = size[..., 1] * jp.abs(local_direction[..., 2]) + size[..., 0] * radial
    box = jp.sum(size * jp.abs(local_direction), axis=-1)
    result = jp.where(geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE), sphere, jp.inf)
    result = jp.where(geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE), capsule, result)
    result = jp.where(geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID), ellipsoid, result)
    result = jp.where(geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER), cylinder, result)
    return jp.where(geom_type == int(mujoco.mjtGeom.mjGEOM_BOX), box, result)


def collision_geom_support_bounds(
    geom_xpos: Any,
    geom_xmat: Any,
    geom_types: Any,
    geom_sizes: Any,
) -> CollisionSupportBounds:
    """Compute world x/y/z support bounds for supported collision primitives."""
    positions = jp.asarray(geom_xpos)
    rotations = jp.asarray(geom_xmat).reshape(positions.shape[:-1] + (3, 3))
    types = jp.asarray(geom_types)
    sizes = jp.asarray(geom_sizes)
    local_x = rotations[..., 0, :]
    local_y = rotations[..., 1, :]
    local_z = rotations[..., 2, :]
    support_x = _support_radius(local_x, types, sizes)
    support_y = _support_radius(local_y, types, sizes)
    support_z = _support_radius(local_z, types, sizes)
    return CollisionSupportBounds(
        min_x=positions[..., 0] - support_x,
        max_x=positions[..., 0] + support_x,
        min_y=positions[..., 1] - support_y,
        max_y=positions[..., 1] + support_y,
        min_z=positions[..., 2] - support_z,
        max_z=positions[..., 2] + support_z,
    )


def full_structure_metrics(
    geom_xpos: Any,
    geom_xmat: Any,
    geom_types: Any,
    geom_sizes: Any,
    *,
    obstacle_front_x: float,
    obstacle_top_z: float,
) -> StructureMetrics:
    """Return full collision-structure progress and vertical obstacle clearance."""
    bounds = collision_geom_support_bounds(
        geom_xpos, geom_xmat, geom_types, geom_sizes
    )
    frontmost = jp.max(bounds.max_x, axis=-1)
    lowest = jp.min(bounds.min_z, axis=-1)
    return StructureMetrics(
        robot_frontmost_x=frontmost,
        obstacle_relative_x=jp.asarray(obstacle_front_x) - frontmost,
        full_structure_clearance=lowest - jp.asarray(obstacle_top_z),
    )


def _robot_geom_state(state: Any, geometry: TwoPhaseGeometry) -> tuple[Any, Any]:
    ids = jp.asarray(geometry.robot_geom_ids, jp.int32)
    rotation_axis = (
        -3
        if state.data.geom_xmat.ndim >= 3
        and state.data.geom_xmat.shape[-2:] == (3, 3)
        else -2
    )
    return (
        jp.take(state.data.geom_xpos, ids, axis=-2),
        jp.take(state.data.geom_xmat, ids, axis=rotation_axis),
    )


def _root_pose_velocity(state: Any, geometry: TwoPhaseGeometry) -> tuple[Any, ...]:
    qpos = state.data.qpos[..., geometry.root_qpos_adr : geometry.root_qpos_adr + 7]
    qvel = state.data.qvel[..., geometry.root_dof_adr : geometry.root_dof_adr + 6]
    quat = qpos[..., 3:7]
    w, x, y, z = (quat[..., index] for index in range(4))
    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    roll = jp.arctan2(sin_roll, cos_roll)
    sin_pitch = jp.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = jp.arcsin(sin_pitch)
    return qpos, qvel, roll, pitch


def _terrain_clearances(
    geom_xpos: Any,
    bounds: CollisionSupportBounds,
    geometry: TwoPhaseGeometry,
) -> Any:
    over_obstacle = (
        (bounds.max_x >= geometry.obstacle_front_x)
        & (bounds.min_x <= geometry.obstacle_back_x)
        & (jp.abs(geom_xpos[..., 1]) <= geometry.obstacle_half_width)
    )
    terrain_z = jp.where(over_obstacle, geometry.obstacle_top_z, 0.0)
    return bounds.min_z - terrain_z


def _physical_geometry_values(state: Any, geometry: TwoPhaseGeometry) -> dict[str, Any]:
    geom_xpos, geom_xmat = _robot_geom_state(state, geometry)
    bounds = collision_geom_support_bounds(
        geom_xpos,
        geom_xmat,
        geometry.robot_geom_types,
        geometry.robot_geom_sizes,
    )
    structure = full_structure_metrics(
        geom_xpos,
        geom_xmat,
        geometry.robot_geom_types,
        geometry.robot_geom_sizes,
        obstacle_front_x=geometry.obstacle_front_x,
        obstacle_top_z=geometry.obstacle_top_z,
    )
    terrain_clearances = _terrain_clearances(geom_xpos, bounds, geometry)
    return {
        "geom_xpos": geom_xpos,
        "bounds": bounds,
        "structure": structure,
        "terrain_clearances": terrain_clearances,
    }


def _physical_failure(info: dict[str, Any]) -> Any:
    """Decode physical failure without conflating successful terminals/timeouts."""
    end_code = jp.asarray(info["end_code"])
    codes = jp.asarray(_PHYSICAL_FAILURE_END_CODES, end_code.dtype)
    return jp.any(end_code[..., None] == codes, axis=-1)


def extract_apex_band_signals(
    state: Any, geometry: TwoPhaseGeometry
) -> ApexBandSignals:
    """Extract Gate A Apex inputs without reward, matcher, or legacy phase use."""
    _, qvel, roll, pitch = _root_pose_velocity(state, geometry)
    physical = _physical_geometry_values(state, geometry)
    clearances = physical["terrain_clearances"]
    illegal_penetration = jp.any(
        clearances < -geometry.body_penetration_tolerance, axis=-1
    )
    physical_failure = _physical_failure(state.info)
    return ApexBandSignals(
        stable_airborne=(
            jp.asarray(state.info["airborne_count"]) >= geometry.airborne_confirm_steps
        ),
        com_vz=qvel[..., 2],
        clearance=physical["structure"].full_structure_clearance,
        roll=roll,
        pitch=pitch,
        angular_speed=jp.linalg.norm(qvel[..., 3:6], axis=-1),
        forward_velocity=qvel[..., 0],
        obstacle_relative_x=physical["structure"].obstacle_relative_x,
        illegal_contact=illegal_penetration,
        physical_failure=physical_failure,
    )


def extract_recovery_signals(
    state: Any,
    geometry: TwoPhaseGeometry,
    *,
    previous_recovery_hold_count: Any,
) -> RecoverySignals:
    """Extract current legal support and recovery values from physical state."""
    _, qvel, roll, pitch = _root_pose_velocity(state, geometry)
    physical = _physical_geometry_values(state, geometry)
    clearances = physical["terrain_clearances"]
    bounds = physical["bounds"]
    wheel_mask = jp.asarray(geometry.wheel_mask)
    body_mask = jp.asarray(geometry.body_mask)
    wheel_clearances = jp.where(wheel_mask, clearances, jp.nan)
    supported = jp.all(
        jp.where(
            wheel_mask,
            jp.abs(wheel_clearances) <= geometry.support_tolerance,
            True,
        ),
        axis=-1,
    )
    wheels_in_region = jp.all(
        jp.where(
            wheel_mask,
            (bounds.min_x >= geometry.landing_x_min)
            & (bounds.max_x <= geometry.landing_x_max)
            & (bounds.min_y >= -geometry.landing_y_limit)
            & (bounds.max_y <= geometry.landing_y_limit),
            True,
        ),
        axis=-1,
    )
    no_body_contact = jp.all(
        jp.where(
            body_mask,
            clearances >= -geometry.body_penetration_tolerance,
            True,
        ),
        axis=-1,
    )
    return RecoverySignals(
        stable_wheel_support=supported,
        landing_region_valid=wheels_in_region,
        no_body_contact=no_body_contact,
        roll=roll,
        pitch=pitch,
        angular_speed=jp.linalg.norm(qvel[..., 3:6], axis=-1),
        forward_velocity=qvel[..., 0],
        previous_recovery_hold_count=jp.asarray(previous_recovery_hold_count),
        physical_failure=_physical_failure(state.info),
    )


def initial_two_phase_event_state() -> TwoPhaseEventState:
    """Return the immutable zero state for the external two-phase event filter."""
    false = jp.asarray(False)
    zero_i = jp.asarray(0, jp.int32)
    return TwoPhaseEventState(
        jump_window_entered=false,
        liftoff_seen=false,
        stable_airborne=false,
        ascending=false,
        apex_band_entered=false,
        descending=false,
        pre_landing=false,
        first_valid_contact=false,
        impact_absorbing=false,
        stable_recovery=false,
        first_event_ticks=jp.full((len(EVENT_NAMES),), -1, jp.int32),
        previous_com_vz=jp.asarray(0.0, jp.float32),
        previous_stable_wheel_support=false,
        recovery_hold_count=zero_i,
        apex_band_width=zero_i,
        max_apex_band_width=zero_i,
    )


def _event_flags(state: TwoPhaseEventState) -> Any:
    return jp.stack([jp.asarray(getattr(state, name)) for name in EVENT_NAMES])


def advance_two_phase_events(
    apex: ApexBandSignals,
    recovery: RecoverySignals,
    previous: TwoPhaseEventState,
    thresholds: TwoPhaseThresholds,
    *,
    tick: Any,
    jump_signal: Any,
) -> TwoPhaseEventState:
    """Advance external event latches without consulting legacy phase ids."""
    no_failure = ~(jp.asarray(apex.physical_failure) | jp.asarray(recovery.physical_failure))
    apex_member = apex_band_membership(apex, thresholds.apex)
    hold_count = advance_recovery_hold_count(recovery, thresholds.recovery)
    recovery_with_count = RecoverySignals(
        stable_wheel_support=recovery.stable_wheel_support,
        landing_region_valid=recovery.landing_region_valid,
        no_body_contact=recovery.no_body_contact,
        roll=recovery.roll,
        pitch=recovery.pitch,
        angular_speed=recovery.angular_speed,
        forward_velocity=recovery.forward_velocity,
        previous_recovery_hold_count=hold_count - 1,
        physical_failure=recovery.physical_failure,
    )
    recovery_complete = descent_recovery_success(
        recovery_with_count, thresholds.recovery
    )

    jump_window_entered = previous.jump_window_entered | (
        jp.asarray(jump_signal) & no_failure
    )
    liftoff_seen = previous.liftoff_seen | (
        previous.jump_window_entered
        & ~jp.asarray(recovery.stable_wheel_support)
        & no_failure
    )
    stable_airborne = previous.stable_airborne | (
        previous.liftoff_seen & jp.asarray(apex.stable_airborne) & no_failure
    )
    ascending = previous.ascending | (
        previous.stable_airborne & (jp.asarray(apex.com_vz) > 0.0) & no_failure
    )
    apex_band_entered = previous.apex_band_entered | (
        previous.ascending & apex_member & no_failure
    )
    descending = previous.descending | (
        previous.apex_band_entered & (jp.asarray(apex.com_vz) < 0.0) & no_failure
    )
    pre_landing = previous.pre_landing | (
        previous.descending
        & jp.asarray(recovery.landing_region_valid)
        & ~jp.asarray(recovery.stable_wheel_support)
        & no_failure
    )
    first_valid_contact = previous.first_valid_contact | (
        previous.pre_landing
        & jp.asarray(recovery.stable_wheel_support)
        & jp.asarray(recovery.landing_region_valid)
        & jp.asarray(recovery.no_body_contact)
        & no_failure
    )
    impact_absorbing = previous.impact_absorbing | (
        previous.first_valid_contact
        & jp.asarray(recovery.stable_wheel_support)
        & ~recovery_complete
        & no_failure
    )
    stable_recovery = previous.stable_recovery | (
        previous.impact_absorbing & recovery_complete & no_failure
    )
    apex_width = jp.where(apex_member, previous.apex_band_width + 1, 0).astype(
        jp.int32
    )
    provisional = TwoPhaseEventState(
        jump_window_entered=jump_window_entered,
        liftoff_seen=liftoff_seen,
        stable_airborne=stable_airborne,
        ascending=ascending,
        apex_band_entered=apex_band_entered,
        descending=descending,
        pre_landing=pre_landing,
        first_valid_contact=first_valid_contact,
        impact_absorbing=impact_absorbing,
        stable_recovery=stable_recovery,
        first_event_ticks=previous.first_event_ticks,
        previous_com_vz=jp.asarray(apex.com_vz),
        previous_stable_wheel_support=jp.asarray(recovery.stable_wheel_support),
        recovery_hold_count=jp.asarray(hold_count, jp.int32),
        apex_band_width=apex_width,
        max_apex_band_width=jp.maximum(previous.max_apex_band_width, apex_width),
    )
    new_events = _event_flags(provisional) & ~_event_flags(previous)
    first_ticks = jp.where(
        new_events & (previous.first_event_ticks < 0),
        jp.asarray(tick, jp.int32),
        previous.first_event_ticks,
    )
    return provisional._replace(first_event_ticks=first_ticks)


def extract_two_phase_events(
    state: Any,
    geometry: TwoPhaseGeometry,
    previous: TwoPhaseEventState,
    thresholds: TwoPhaseThresholds,
    *,
    tick: Any,
) -> TwoPhaseEventState:
    """Extract physical signals and advance the external event filter."""
    apex = extract_apex_band_signals(state, geometry)
    recovery = extract_recovery_signals(
        state,
        geometry,
        previous_recovery_hold_count=previous.recovery_hold_count,
    )
    return advance_two_phase_events(
        apex,
        recovery,
        previous,
        thresholds,
        tick=tick,
        jump_signal=jp.asarray(state.info["jump_signal_latched"]),
    )
def _object_name(model: mujoco.MjModel, kind: str, index: int) -> str:
    value = getattr(model, kind)(index).name
    return "" if value is None else str(value)


def build_two_phase_geometry(model: mujoco.MjModel, cfg: Any) -> TwoPhaseGeometry:
    """Build immutable JAX geometry from the authoritative MuJoCo model."""
    collision = (np.asarray(model.geom_contype) != 0) | (
        np.asarray(model.geom_conaffinity) != 0
    )
    body_ids = np.asarray(model.geom_bodyid, dtype=np.int32)
    robot_ids = np.flatnonzero(collision & (body_ids != 0)).astype(np.int32)
    unsupported = [
        int(index)
        for index in robot_ids
        if int(model.geom_type[index]) not in _GEOM_FORMULAS
    ]
    if unsupported:
        names = [_object_name(model, "geom", index) for index in unsupported]
        raise ValueError(f"Unsupported collision-relevant robot geoms: {names}")
    wheel_body_ids = {
        int(model.body("frontwheel").id),
        int(model.body("rearwheel").id),
    }
    robot_body_ids = body_ids[robot_ids]
    wheel_mask = np.asarray(
        [int(body_id) in wheel_body_ids for body_id in robot_body_ids], dtype=bool
    )
    obstacle = int(model.geom("step").id)
    obstacle_pos = np.asarray(model.geom_pos[obstacle], dtype=np.float64)
    obstacle_size = np.asarray(model.geom_size[obstacle], dtype=np.float64)
    root_joint = int(model.joint("floating_base_joint").id)
    return TwoPhaseGeometry(
        robot_geom_ids=robot_ids,
        robot_geom_types=jp.asarray(model.geom_type[robot_ids], jp.int32),
        robot_geom_sizes=jp.asarray(model.geom_size[robot_ids], jp.float32),
        robot_geom_body_ids=robot_body_ids,
        wheel_mask=jp.asarray(wheel_mask),
        body_mask=jp.asarray(~wheel_mask),
        obstacle_geom_id=obstacle,
        obstacle_front_x=float(obstacle_pos[0] - obstacle_size[0]),
        obstacle_back_x=float(obstacle_pos[0] + obstacle_size[0]),
        obstacle_top_z=float(obstacle_pos[2] + obstacle_size[2]),
        obstacle_half_width=float(obstacle_size[1]),
        landing_x_min=float(cfg.step_front_x + cfg.valid_landing_min_past_edge),
        landing_x_max=float(cfg.step_back_x - cfg.valid_landing_back_margin),
        landing_y_limit=float(cfg.step_half_width - cfg.landing_side_margin),
        root_qpos_adr=int(model.jnt_qposadr[root_joint]),
        root_dof_adr=int(model.jnt_dofadr[root_joint]),
        airborne_confirm_steps=int(cfg.airborne_confirm_steps),
        support_tolerance=float(cfg.landing_candidate_clearance_max),
        body_penetration_tolerance=0.002,
    )


def geometry_manifest(
    model: mujoco.MjModel, geometry: TwoPhaseGeometry
) -> dict[str, Any]:
    """Describe formula coverage for every authoritative XML geom."""
    robot_ids = set(int(value) for value in geometry.robot_geom_ids)
    wheel_ids = {
        int(geom_id)
        for geom_id, is_wheel in zip(
            geometry.robot_geom_ids, np.asarray(geometry.wheel_mask), strict=True
        )
        if bool(is_wheel)
    }
    rows = []
    for index in range(model.ngeom):
        body_id = int(model.geom_bodyid[index])
        geom_type = int(model.geom_type[index])
        collision = bool(
            int(model.geom_contype[index]) or int(model.geom_conaffinity[index])
        )
        rows.append(
            {
                "geom_id": index,
                "geom_name": _object_name(model, "geom", index),
                "geom_type": mujoco.mjtGeom(geom_type).name.removeprefix("mjGEOM_").lower(),
                "collision_participation": collision,
                "body_id": body_id,
                "body_ownership": (
                    "world" if body_id == 0 else _object_name(model, "body", body_id)
                ),
                "classification": (
                    "wheel" if index in wheel_ids else "robot_body" if index in robot_ids else "terrain" if body_id == 0 else "visual_only"
                ),
                "supported_jax_geometry_formula": (
                    _GEOM_FORMULAS.get(geom_type) if index in robot_ids else None
                ),
                "exclusion_reason": (
                    "collision_disabled_visual_geom"
                    if body_id != 0 and not collision
                    else None
                ),
            }
        )
    model_identity = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "contract_version": 1,
        "authoritative_ngeom": int(model.ngeom),
        "authoritative_geom_ids": list(range(model.ngeom)),
        "model_identity_sha256": model_identity,
        "full_structure_clearance": True,
        "geoms": rows,
    }


def validate_geometry_manifest(
    manifest: dict[str, Any],
    *,
    model: mujoco.MjModel,
    geometry: TwoPhaseGeometry,
) -> dict[str, Any]:
    """Validate against the model, never against manifest self-description alone."""
    rows = manifest.get("geoms")
    rows_valid = isinstance(rows, list) and bool(rows)
    collision_robot = (
        [
            row
            for row in rows
            if row.get("body_ownership") != "world"
            and row.get("collision_participation") is True
        ]
        if rows_valid
        else []
    )
    authoritative_ngeom = manifest.get("authoritative_ngeom")
    authoritative_ids = manifest.get("authoritative_geom_ids")
    row_ids = [row.get("geom_id") for row in rows] if rows_valid else []
    computed_identity = (
        hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if rows_valid
        else None
    )
    geom_identity = (
        isinstance(authoritative_ngeom, int)
        and authoritative_ngeom > 0
        and authoritative_ids == list(range(authoritative_ngeom))
        and len(row_ids) == authoritative_ngeom
        and row_ids == authoritative_ids
        and len(set(row_ids)) == authoritative_ngeom
        and manifest.get("model_identity_sha256") == computed_identity
    )
    authoritative = geometry_manifest(model, geometry)
    authoritative_model_match = (
        manifest.get("authoritative_ngeom") == authoritative["authoritative_ngeom"]
        and manifest.get("authoritative_geom_ids")
        == authoritative["authoritative_geom_ids"]
        and manifest.get("model_identity_sha256")
        == authoritative["model_identity_sha256"]
        and rows == authoritative["geoms"]
    )
    checks = {
        "contract_version": manifest.get("contract_version") == 1,
        "geom_rows": rows_valid,
        "geom_identity": geom_identity,
        "authoritative_model_match": authoritative_model_match,
        "collision_robot_geoms": bool(collision_robot),
        "formula_coverage": bool(collision_robot)
        and all(bool(row.get("supported_jax_geometry_formula")) for row in collision_robot),
        "full_structure_claim": manifest.get("full_structure_clearance") is True,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {"valid": not failed, "checks": checks, "failed": failed}
