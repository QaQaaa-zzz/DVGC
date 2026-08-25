"""Authoritative Host MuJoCo audit and explicit MJX-Warp conversion."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from jax import numpy as jp
import mujoco
from mujoco import mjx
import numpy as np

from .action_mapping import ActionMapping
from .config import ResolvedConfig, file_sha256
from .constants import EXPECTED_XML_SHA256, SIM_DT


@dataclass(frozen=True)
class ModelIndex:
    keyframe_id: int
    root_qpos_address: int
    root_dof_address: int
    rearwheel_qpos_address: int
    rearwheel_dof_address: int
    steering_qpos_address: int
    steering_dof_address: int
    frontwheel_qpos_address: int
    frontwheel_dof_address: int
    hip_qpos_address: int
    hip_dof_address: int
    knee_qpos_address: int
    knee_dof_address: int
    floor_geom_id: int
    obstacle_geom_id: int
    frontwheel_geom_id: int
    rearwheel_geom_id: int
    sensor_addresses: tuple[int, ...]


@dataclass(frozen=True)
class ModelBundle:
    xml_path: Path
    xml_sha256: str
    mj_model: mujoco.MjModel
    mjx_model: Any | None
    model_index: ModelIndex
    action_mapping: ActionMapping
    actuator_names: tuple[str, ...]
    payload_mass: float


JOINT_NAMES = (
    "floating_base_joint",
    "rearwheel_joint",
    "steering_joint",
    "frontwheel_joint",
    "hip_joint",
    "knee_joint",
)
ACTUATOR_NAMES = (
    "cmd_steering_v",
    "cmd_rearwheel_f",
    "cmd_hip_f",
    "cmd_knee_f",
)
SENSOR_NAMES = (
    "steering_joint_pos_sensor",
    "steering_joint_vel_sensor",
    "rearwheel_joint_vel_sensor",
    "frontwheel_joint_vel_sensor",
    "acc_local",
    "gyro_local",
    "ori_global",
    "pos_global",
    "vel_global",
    "ang_global",
    "hip_joint_pos_sensor",
    "knee_joint_pos_sensor",
    "hip_joint_vel_sensor",
    "knee_joint_vel_sensor",
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _named_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    identifier = int(mujoco.mj_name2id(model, object_type, name))
    if identifier < 0:
        raise ValueError(f"authoritative model is missing {name}")
    return identifier


def _build_index(model: mujoco.MjModel) -> ModelIndex:
    joints = {
        name: _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in JOINT_NAMES
    }
    sensors = tuple(
        _named_id(model, mujoco.mjtObj.mjOBJ_SENSOR, name) for name in SENSOR_NAMES
    )
    sensor_addresses = tuple(int(model.sensor_adr[index]) for index in sensors)
    return ModelIndex(
        keyframe_id=_named_id(model, mujoco.mjtObj.mjOBJ_KEY, "initial_state"),
        root_qpos_address=int(model.jnt_qposadr[joints["floating_base_joint"]]),
        root_dof_address=int(model.jnt_dofadr[joints["floating_base_joint"]]),
        rearwheel_qpos_address=int(model.jnt_qposadr[joints["rearwheel_joint"]]),
        rearwheel_dof_address=int(model.jnt_dofadr[joints["rearwheel_joint"]]),
        steering_qpos_address=int(model.jnt_qposadr[joints["steering_joint"]]),
        steering_dof_address=int(model.jnt_dofadr[joints["steering_joint"]]),
        frontwheel_qpos_address=int(model.jnt_qposadr[joints["frontwheel_joint"]]),
        frontwheel_dof_address=int(model.jnt_dofadr[joints["frontwheel_joint"]]),
        hip_qpos_address=int(model.jnt_qposadr[joints["hip_joint"]]),
        hip_dof_address=int(model.jnt_dofadr[joints["hip_joint"]]),
        knee_qpos_address=int(model.jnt_qposadr[joints["knee_joint"]]),
        knee_dof_address=int(model.jnt_dofadr[joints["knee_joint"]]),
        floor_geom_id=_named_id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor"),
        obstacle_geom_id=_named_id(model, mujoco.mjtObj.mjOBJ_GEOM, "step"),
        frontwheel_geom_id=_named_id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "frontwheel_collision"
        ),
        rearwheel_geom_id=_named_id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "rearwheel_collision"
        ),
        sensor_addresses=sensor_addresses,
    )


def load_host_model(config: ResolvedConfig) -> ModelBundle:
    xml_path = (_repository_root() / str(config.model["xml_path"])).resolve()
    identity = file_sha256(xml_path)
    declared = str(config.model["xml_sha256"])
    if identity != declared or identity != EXPECTED_XML_SHA256:
        raise ValueError(
            "authoritative XML SHA-256 mismatch: "
            f"declared={declared}, expected={EXPECTED_XML_SHA256}, actual={identity}"
        )
    xml_root = ElementTree.parse(xml_path).getroot()
    payload_geoms = xml_root.findall(".//geom[@name='load']")
    if len(payload_geoms) != 1 or "mass" not in payload_geoms[0].attrib:
        raise ValueError("authoritative XML must contain one explicit load geom mass")
    payload_mass = float(payload_geoms[0].attrib["mass"])
    if not np.isclose(payload_mass, 2.0, atol=0.0, rtol=0.0):
        raise ValueError("authoritative payload is not 2 kg")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    model.opt.timestep = SIM_DT
    actuator_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        for index in range(model.nu)
    )
    if actuator_names != ACTUATOR_NAMES:
        raise ValueError(f"actuator order mismatch: {actuator_names}")
    if not np.array_equal(
        np.asarray(model.actuator_forcerange[2:], dtype=np.float64),
        np.asarray([[-50.0, 50.0], [-50.0, 50.0]], dtype=np.float64),
    ):
        raise ValueError("hip/knee force ranges are not +/-50 N m")
    index = _build_index(model)
    mapping = ActionMapping(
        ctrl_min=jp.asarray(model.actuator_ctrlrange[:, 0], dtype=jp.float32),
        ctrl_max=jp.asarray(model.actuator_ctrlrange[:, 1], dtype=jp.float32),
        hip_initial=float(model.key_qpos[index.keyframe_id, index.hip_qpos_address]),
        knee_initial=float(model.key_qpos[index.keyframe_id, index.knee_qpos_address]),
        base_rear_speed=float(config.action.base_rear_speed),
        rear_speed_delta=float(config.action.rear_speed_delta),
        joint_target_semantics=config.action.joint_target_semantics,
        knee_target_delta=float(config.action.knee_target_delta or 0.0),
    )
    return ModelBundle(
        xml_path=xml_path,
        xml_sha256=identity,
        mj_model=model,
        mjx_model=None,
        model_index=index,
        action_mapping=mapping,
        actuator_names=actuator_names,
        payload_mass=payload_mass,
    )


def put_warp_model(bundle: ModelBundle) -> ModelBundle:
    if bundle.mjx_model is not None:
        return bundle
    return replace(
        bundle,
        mjx_model=mjx.put_model(bundle.mj_model, impl="warp"),
    )
