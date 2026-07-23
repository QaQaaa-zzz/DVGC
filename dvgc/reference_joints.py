"""Stage-aligned joint-state contracts for reference-derived proposals."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np


@dataclass(frozen=True)
class StageJointState:
    hip: float
    knee: float
    hip_velocity: float
    knee_velocity: float
    source: str


def _joint_addresses(model: mujoco.MjModel, name: str) -> tuple[int, int]:
    joint_id = int(model.joint(name).id)
    return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])


def stage_joint_state(model: mujoco.MjModel, row: Any, phase: str) -> StageJointState:
    """Return the declared joint state without borrowing a keyframe root pose.

    Takeoff begins from the authoritative XML's contracted preload posture.
    Other phases retain the corresponding reference-trajectory joint state.
    Root position, orientation and rigid-body velocity remain owned by the
    proposal and are never copied from the XML keyframe here.
    """
    if phase == "takeoff":
        if model.nkey < 1:
            raise ValueError("Authoritative XML has no Takeoff keyframe")
        try:
            key_id = int(model.key("initial_state").id)
        except KeyError as exc:
            raise ValueError("Authoritative XML lacks key 'initial_state'") from exc
        hip_qpos, hip_qvel = _joint_addresses(model, "hip_joint")
        knee_qpos, knee_qvel = _joint_addresses(model, "knee_joint")
        return StageJointState(
            hip=float(model.key_qpos[key_id, hip_qpos]),
            knee=float(model.key_qpos[key_id, knee_qpos]),
            hip_velocity=float(model.key_qvel[key_id, hip_qvel]),
            knee_velocity=float(model.key_qvel[key_id, knee_qvel]),
            source="xml_key:initial_state",
        )
    return StageJointState(
        hip=float(row.hip_position),
        knee=float(row.knee_position),
        hip_velocity=float(row.hip_velocity),
        knee_velocity=float(row.knee_velocity),
        source="reference_trajectory",
    )


def apply_stage_joint_state(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qvel: np.ndarray,
    joint_state: StageJointState,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply only hip/knee state, preserving every root coordinate."""
    result_qpos = np.asarray(qpos).copy()
    result_qvel = np.asarray(qvel).copy()
    hip_qpos, hip_qvel = _joint_addresses(model, "hip_joint")
    knee_qpos, knee_qvel = _joint_addresses(model, "knee_joint")
    result_qpos[hip_qpos] = joint_state.hip
    result_qpos[knee_qpos] = joint_state.knee
    result_qvel[hip_qvel] = joint_state.hip_velocity
    result_qvel[knee_qvel] = joint_state.knee_velocity
    return result_qpos, result_qvel
