"""Takeoff signal extraction for the OrangeBike DVGC environment.

This module computes physical signals only.  It does not choose rewards, mutate
state, or read collision labels.  Takeoff success checks should use these wheel
and signed-attitude signals instead of base-height shortcuts.
"""
from __future__ import annotations

from typing import Dict

import jax.numpy as jp
from mujoco import mjx


def _body_center_z(data: mjx.Data, body_id: int, fallback_z: jp.ndarray) -> jp.ndarray:
    if body_id >= 0:
        return data.xpos[body_id, 2]
    return fallback_z


def wheel_tire_bottoms(
    data: mjx.Data,
    *,
    frontwheel_body_id: int,
    rearwheel_body_id: int,
    wheel_radius: float,
    fallback_base_z: jp.ndarray,
) -> tuple[jp.ndarray, jp.ndarray, jp.ndarray, jp.ndarray]:
    """Returns front/rear wheel center z and tire-bottom z."""
    fallback_center_z = fallback_base_z
    front_center_z = _body_center_z(data, frontwheel_body_id, fallback_center_z)
    rear_center_z = _body_center_z(data, rearwheel_body_id, fallback_center_z)
    radius = jp.asarray(float(wheel_radius), dtype=jp.float32)
    return front_center_z, rear_center_z, front_center_z - radius, rear_center_z - radius


def compute_takeoff_signals(
    *,
    cfg,
    data: mjx.Data,
    qpos: jp.ndarray,
    qvel: jp.ndarray,
    roll: jp.ndarray,
    pitch: jp.ndarray,
    action: jp.ndarray,
    frontwheel_body_id: int,
    rearwheel_body_id: int,
    prev_front_tire_bottom_z: jp.ndarray,
    prev_rear_tire_bottom_z: jp.ndarray,
    knee_target: jp.ndarray,
    knee_pos: jp.ndarray,
    knee_vel: jp.ndarray,
) -> Dict[str, jp.ndarray]:
    """Computes takeoff diagnostics from deployable state and body poses.

    Positive signed pitch is treated as the bad takeoff direction in the current
    robot convention.  Dual-wheel liftoff requires both tires to rise and move
    upward together; base z alone is never a success signal here.
    """
    front_center_z, rear_center_z, front_tire_z, rear_tire_z = wheel_tire_bottoms(
        data,
        frontwheel_body_id=frontwheel_body_id,
        rearwheel_body_id=rearwheel_body_id,
        wheel_radius=float(cfg.takeoff_wheel_radius),
        fallback_base_z=qpos[2],
    )
    dt = jp.asarray(float(cfg.ctrl_dt), dtype=jp.float32)
    front_tire_vz = (front_tire_z - prev_front_tire_bottom_z) / jp.maximum(dt, 1e-6)
    rear_tire_vz = (rear_tire_z - prev_rear_tire_bottom_z) / jp.maximum(dt, 1e-6)
    min_tire_z = jp.minimum(front_tire_z, rear_tire_z)
    wheel_height_diff = jp.abs(front_tire_z - rear_tire_z)
    wheel_vz_diff = jp.abs(front_tire_vz - rear_tire_vz)

    pitch_signed_deg = pitch * 180.0 / jp.pi
    roll_signed_deg = roll * 180.0 / jp.pi
    positive_pitch_bad = pitch_signed_deg > float(cfg.takeoff_positive_pitch_soft_deg)
    positive_pitch_hard = pitch_signed_deg > float(cfg.takeoff_positive_pitch_hard_deg)

    ground_z = jp.asarray(0.0, dtype=jp.float32)
    front_lift_height_ready = front_tire_z >= ground_z + float(cfg.takeoff_liftoff_height)
    rear_lift_height_ready = rear_tire_z >= ground_z + float(cfg.takeoff_liftoff_height)
    front_lift_vz_ready = front_tire_vz >= float(cfg.takeoff_liftoff_vz)
    rear_lift_vz_ready = rear_tire_vz >= float(cfg.takeoff_liftoff_vz)
    wheel_height_sync_ready = wheel_height_diff <= float(cfg.takeoff_sync_height_diff_max)
    wheel_vz_sync_ready = wheel_vz_diff <= float(cfg.takeoff_sync_vz_diff_max)
    dual_wheel_liftoff = (
        front_lift_height_ready
        & rear_lift_height_ready
        & front_lift_vz_ready
        & rear_lift_vz_ready
        & wheel_height_sync_ready
        & wheel_vz_sync_ready
        & (~positive_pitch_bad)
    )

    wheel_clearance_req = float(cfg.step_top_z) + float(cfg.takeoff_step_clearance_margin)
    front_clearance_ready = front_tire_z >= wheel_clearance_req
    rear_clearance_ready = rear_tire_z >= wheel_clearance_req
    wheel_clearance_ready = front_clearance_ready & rear_clearance_ready & wheel_height_sync_ready & (~positive_pitch_bad)

    frontmost_x = qpos[0] + float(cfg.takeoff_frontmost_offset)
    distance_to_obstacle = float(cfg.step_front_x) - frontmost_x
    liftoff_deadline_x = float(cfg.step_front_x) - float(cfg.takeoff_liftoff_before_step)
    clearance_deadline_x = float(cfg.step_front_x) - float(cfg.takeoff_clearance_before_step)

    front_grounded_for_wheelie = front_tire_z <= ground_z + float(cfg.takeoff_liftoff_height) + 0.010
    rear_lifted_for_wheelie = rear_tire_z >= ground_z + 0.060
    wheelie_detected = (
        front_grounded_for_wheelie
        & rear_lifted_for_wheelie
        & (pitch_signed_deg > float(cfg.takeoff_wheelie_pitch_deg))
    )

    knee_moved = jp.abs(knee_pos - float(cfg.knee_initial)) > 0.030
    knee_target_delta = jp.abs(knee_target - float(cfg.knee_initial))

    return {
        "frontwheel_center_z": front_center_z,
        "rearwheel_center_z": rear_center_z,
        "front_tire_bottom_z": front_tire_z,
        "rear_tire_bottom_z": rear_tire_z,
        "front_tire_vz": front_tire_vz,
        "rear_tire_vz": rear_tire_vz,
        "min_tire_bottom_z": min_tire_z,
        "wheel_height_diff": wheel_height_diff,
        "wheel_vz_diff": wheel_vz_diff,
        "pitch_signed_deg": pitch_signed_deg,
        "roll_signed_deg": roll_signed_deg,
        "positive_pitch_bad": positive_pitch_bad,
        "positive_pitch_hard": positive_pitch_hard,
        "front_lift_height_ready": front_lift_height_ready,
        "rear_lift_height_ready": rear_lift_height_ready,
        "front_lift_vz_ready": front_lift_vz_ready,
        "rear_lift_vz_ready": rear_lift_vz_ready,
        "wheel_height_sync_ready": wheel_height_sync_ready,
        "wheel_vz_sync_ready": wheel_vz_sync_ready,
        "dual_wheel_liftoff": dual_wheel_liftoff,
        "front_clearance_ready": front_clearance_ready,
        "rear_clearance_ready": rear_clearance_ready,
        "wheel_clearance_ready": wheel_clearance_ready,
        "min_wheel_clearance_over_step": min_tire_z - wheel_clearance_req,
        "frontmost_x": frontmost_x,
        "distance_to_obstacle": distance_to_obstacle,
        "liftoff_deadline_x": jp.asarray(liftoff_deadline_x, dtype=jp.float32),
        "clearance_deadline_x": jp.asarray(clearance_deadline_x, dtype=jp.float32),
        "wheelie_detected": wheelie_detected,
        "knee_moved": knee_moved,
        "knee_target_delta": knee_target_delta,
        "knee_pos": knee_pos,
        "knee_vel": knee_vel,
        "knee_target": knee_target,
        "base_z_success_disabled": jp.asarray(bool(cfg.takeoff_disable_base_z_success), dtype=jp.float32),
    }
