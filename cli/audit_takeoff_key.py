"""Audit the authoritative XML Takeoff key and actor-to-actuator mapping."""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from dvgc.config import config_hash, file_sha256, load_config
from dvgc.action_mapping import knee_position_target
from dvgc.reference import ReferenceTrajectory
from dvgc.runtime import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--reference", default="data/reference_jump.csv")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    model = mujoco.MjModel.from_xml_path(str(cfg.xml_path))
    key_id = int(model.key("initial_state").id)

    joints = {}
    for name in ("hip_joint", "knee_joint"):
        joint_id = int(model.joint(name).id)
        qpos_address = int(model.jnt_qposadr[joint_id])
        qvel_address = int(model.jnt_dofadr[joint_id])
        joints[name] = {
            "joint_id": joint_id,
            "qpos_address": qpos_address,
            "qvel_address": qvel_address,
            "key_qpos": float(model.key_qpos[key_id, qpos_address]),
            "key_qvel": float(model.key_qvel[key_id, qvel_address]),
            "range_rad": [float(v) for v in model.jnt_range[joint_id]],
            "unit": "radian",
        }

    actuators = {}
    for name in ("cmd_hip_f", "cmd_knee_f"):
        actuator_id = int(model.actuator(name).id)
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        actuators[name] = {
            "actuator_id": actuator_id,
            "joint_id": joint_id,
            "joint_name": model.joint(joint_id).name,
            "ctrlrange_rad": [float(v) for v in model.actuator_ctrlrange[actuator_id]],
            "forcerange_nm": [float(v) for v in model.actuator_forcerange[actuator_id]],
        }

    reference = ReferenceTrajectory.load(args.reference)
    anchors = reference.anchors()
    before = reference.df.iloc[max(0, anchors.approach_end - 20)]
    after = reference.df.iloc[anchors.takeoff_end]
    payload = {
        "status": "PASS",
        "key_name": "initial_state",
        "key_id": key_id,
        "full_key_qpos": [float(v) for v in model.key_qpos[key_id]],
        "full_key_qvel": [float(v) for v in model.key_qvel[key_id]],
        "qpos_order": [
            "root_x", "root_y", "root_z", "root_qw", "root_qx", "root_qy", "root_qz",
            "rearwheel", "steering", "frontwheel", "hip", "knee",
        ],
        "joints": joints,
        "actuators": actuators,
        "actor_action_order": ["steer", "rear-wheel drive", "hip", "knee"],
        "action_direction": {
            "hip_positive": "increases XML hip target from -1.2 toward +0.5; reference launch-extension direction",
            "hip_negative": "decreases XML hip target toward -1.3; contracted direction",
            "knee_positive": "decreases XML knee target from 2.5; reference launch-extension direction",
            "knee_negative": "increases XML knee target toward 2.5; contracted direction",
        },
        "numeric_action_targets_from_key": {
            "hip_action_minus_1": float(cfg.hip_min),
            "hip_action_zero": float(cfg.hip_initial),
            "hip_action_plus_1": float(cfg.hip_max),
            "knee_action_minus_1": float(knee_position_target(
                joints["knee_joint"]["key_qpos"], -1.0,
                target_delta=cfg.knee_action_target_delta,
                knee_min=cfg.knee_min, knee_max=cfg.knee_max, xp=np,
            )[1]),
            "knee_action_zero": float(knee_position_target(
                joints["knee_joint"]["key_qpos"], 0.0,
                target_delta=cfg.knee_action_target_delta,
                knee_min=cfg.knee_min, knee_max=cfg.knee_max, xp=np,
            )[1]),
            "knee_action_plus_1": float(knee_position_target(
                joints["knee_joint"]["key_qpos"], 1.0,
                target_delta=cfg.knee_action_target_delta,
                knee_min=cfg.knee_min, knee_max=cfg.knee_max, xp=np,
            )[1]),
        },
        "reference_launch_joint_change": {
            "from_index": int(max(0, anchors.approach_end - 20)),
            "to_index": int(anchors.takeoff_end),
            "hip_delta_rad": float(after.hip_position - before.hip_position),
            "knee_delta_rad": float(after.knee_position - before.knee_position),
        },
        "checks": {
            "hip_key_is_minus_1_2": abs(joints["hip_joint"]["key_qpos"] + 1.2) <= 1e-9,
            "knee_key_is_2_5": abs(joints["knee_joint"]["key_qpos"] - 2.5) <= 1e-9,
            "joint_order_distinct": joints["hip_joint"]["qpos_address"] != joints["knee_joint"]["qpos_address"],
            "hip_actuator_maps_hip": actuators["cmd_hip_f"]["joint_name"] == "hip_joint",
            "knee_actuator_maps_knee": actuators["cmd_knee_f"]["joint_name"] == "knee_joint",
            "key_within_limits": (
                joints["hip_joint"]["range_rad"][0] <= joints["hip_joint"]["key_qpos"] <= joints["hip_joint"]["range_rad"][1]
                and joints["knee_joint"]["range_rad"][0] <= joints["knee_joint"]["key_qpos"] <= joints["knee_joint"]["range_rad"][1]
            ),
        },
        "xml_path": str(Path(cfg.xml_path).resolve()),
        "xml_sha256": file_sha256(cfg.xml_path),
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": file_sha256(args.config),
        "resolved_config_hash": config_hash(cfg),
        "loader_path": str((Path(__file__).parents[1] / "dvgc/reference_joints.py").resolve()),
        "loader_sha256": file_sha256(Path(__file__).parents[1] / "dvgc/reference_joints.py"),
    }
    if not all(payload["checks"].values()):
        payload["status"] = "FAIL"
    save_json(args.output, payload)
    if payload["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
