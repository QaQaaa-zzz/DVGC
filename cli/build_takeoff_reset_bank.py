"""Build authenticity-audited canonical and reference-aligned Takeoff resets."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from cli.build_candidates import _hip_hold_action
from cli.stage_label_pilot import sample_from_state
from dvgc.bank import SnapshotBank
from dvgc.config import STAGE_ID, config_hash, file_sha256, load_config
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.reference import ReferenceTrajectory
from dvgc.reference_joints import stage_joint_state
from dvgc.reset_geometry import GroundSupportSolver
from dvgc.runtime import save_json
from dvgc.stage_reachability import evaluate_entry


def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.asarray([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], np.float64)


def quantiles(values) -> dict:
    data = np.asarray(values, np.float64)
    return {
        "min": float(data.min()), "p50": float(np.median(data)),
        "p95": float(np.quantile(data, 0.95)), "max": float(data.max()),
    }


def joint_distribution(bank: SnapshotBank, hip_qpos: int, knee_qpos: int) -> dict:
    if not bank.records:
        return {"records": 0}
    return {
        "records": len(bank.records),
        "hip": quantiles([row["qpos"][hip_qpos] for row in bank.records]),
        "knee": quantiles([row["qpos"][knee_qpos] for row in bank.records]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--key-audit", required=True)
    parser.add_argument("--old-bank", default="")
    parser.add_argument("--target", type=int, default=120)
    parser.add_argument("--canonical-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=9_920_000)
    parser.add_argument("--proposal-budget", type=int, default=3000)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--reference", default="data/reference_jump.csv")
    args = parser.parse_args()

    cfg = load_config(args.config, {
        "training_stage": "takeoff", "use_bank_resets": False,
        "domain_randomization": False, "obs_noise_enable": False,
    })
    env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank())
    model = env.mj_model
    solver = GroundSupportSolver(cfg.xml_path)
    reference = ReferenceTrajectory.load(args.reference)
    df = reference.df
    anchors = reference.anchors()
    key_audit = json.loads(Path(args.key_audit).read_text())
    if key_audit.get("status") != "PASS" or key_audit.get("xml_sha256") != file_sha256(cfg.xml_path):
        raise SystemExit("Takeoff key audit is missing, failed, or stale")

    hip_joint = int(model.joint("hip_joint").id)
    knee_joint = int(model.joint("knee_joint").id)
    hip_qpos, knee_qpos = int(model.jnt_qposadr[hip_joint]), int(model.jnt_qposadr[knee_joint])
    hip_qvel, knee_qvel = int(model.jnt_dofadr[hip_joint]), int(model.jnt_dofadr[knee_joint])
    root_joint = int(model.joint("floating_base_joint").id)
    root_qpos, root_qvel = int(model.jnt_qposadr[root_joint]), int(model.jnt_dofadr[root_joint])
    angle_values = np.unwrap(np.deg2rad(df[["roll_angle", "pitch_angle", "yaw_angle"]].to_numpy(float)), axis=0)
    angular_velocities = np.gradient(angle_values, df["time"].to_numpy(float), axis=0)

    compressed = (
        (df.index <= anchors.approach_end)
        & (df["pos_x"] >= float(df.loc[anchors.approach_end, "pos_x"]) - 0.35)
        & (df["hip_position"] <= -1.15)
        & (df["knee_position"] >= 2.30)
        & (df["hip_position"] >= float(model.jnt_range[hip_joint, 0]))
        & (df["hip_position"] <= float(model.jnt_range[hip_joint, 1]))
        & (df["knee_position"] >= float(model.jnt_range[knee_joint, 0]))
        & (df["knee_position"] <= float(model.jnt_range[knee_joint, 1]))
    )
    compressed_indices = [int(i) for i in df.index[compressed]]
    if len(compressed_indices) < 4:
        raise SystemExit(f"Only {len(compressed_indices)} legal compressed reference rows")

    protocol = {
        "version": "takeoff_reset_authenticity_v3",
        "xml_sha256": file_sha256(cfg.xml_path),
        "config_hash": config_hash(cfg),
        "reference_sha256": file_sha256(args.reference),
        "key_audit_sha256": file_sha256(args.key_audit),
        "classes": {
            "canonical_compressed": "complete XML key pose/velocity; task x translation and consistent forward wheel speed only; MuJoCo grounded placement",
            "reference_aligned_compressed": "adjacent-time correlated interpolation of reference root/joint pose, velocity and action; MuJoCo grounded placement",
        },
        "compression_gate": {"hip_max": -1.15, "knee_min": 2.30},
        "no_settle": True,
        "shock_probe_steps": 5,
    }
    protocol["sha256"] = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    rng = np.random.default_rng(args.seed)
    step_fn = jax.jit(env.step)
    target_canonical = int(round(args.target * args.canonical_fraction))
    quotas = {
        "canonical_compressed": target_canonical,
        "reference_aligned_compressed": args.target - target_canonical,
    }
    accepted = Counter()
    rejected = Counter()
    records = []
    identities = set()
    key_id = int(model.key(str(key_audit["key_name"])).id)
    key_joint = stage_joint_state(model, None, "takeoff")
    adjacent_pairs = [
        (a, b) for a, b in zip(compressed_indices[:-1], compressed_indices[1:])
        if b == a + 1
    ]
    if not adjacent_pairs:
        raise SystemExit("No adjacent compressed reference pairs")

    for attempt in range(args.proposal_budget):
        if all(accepted[name] >= count for name, count in quotas.items()):
            break
        kind = "canonical_compressed" if accepted["canonical_compressed"] < quotas["canonical_compressed"] and attempt % 2 == 0 else "reference_aligned_compressed"
        if accepted[kind] >= quotas[kind]:
            kind = "canonical_compressed"
        if kind == "canonical_compressed":
            index = int(rng.choice(compressed_indices))
            row = df.iloc[index].copy()
            reference_interval = None
            interpolation_fraction = None
        else:
            left, right = adjacent_pairs[int(rng.integers(0, len(adjacent_pairs)))]
            interpolation_fraction = float(rng.uniform(0.05, 0.95))
            row = (1.0 - interpolation_fraction) * df.iloc[left] + interpolation_fraction * df.iloc[right]
            index = int(round((1.0 - interpolation_fraction) * left + interpolation_fraction * right))
            reference_interval = [int(left), int(right)]
        qpos = np.asarray(model.key_qpos[key_id], np.float64).copy()
        qvel = np.asarray(model.key_qvel[key_id], np.float64).copy()
        qpos[root_qpos + 0] = float(row.pos_x + rng.uniform(-0.025, 0.025))
        if kind == "canonical_compressed":
            joints = key_joint
            last_action = np.zeros(4, np.float32)
            qpos[root_qpos + 1] = float(rng.uniform(-0.004, 0.004))
            angles = np.deg2rad(rng.uniform([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]))
            qpos[root_qpos + 3:root_qpos + 7] = quat_from_euler(*angles)
            qvel[root_qvel + 0] = float(row.vel_x + rng.uniform(-0.08, 0.08))
        else:
            joints = stage_joint_state(model, row, "flight")
            qpos[root_qpos + 0] = float(row.pos_x)
            qpos[root_qpos + 1] = float(row.pos_y)
            angles = np.deg2rad([row.roll_angle, row.pitch_angle, row.yaw_angle])
            qpos[root_qpos + 3:root_qpos + 7] = quat_from_euler(*angles)
            qvel[root_qvel:root_qvel + 3] = np.asarray([row.vel_x, row.vel_y, row.vel_z])
            left, right = reference_interval
            qvel[root_qvel + 3:root_qvel + 6] = (
                (1.0 - interpolation_fraction) * angular_velocities[left]
                + interpolation_fraction * angular_velocities[right]
            )
            last_action = np.clip(
                [row.action_steering, row.action_rearwheel, row.action_hip, row.action_knee],
                -1.0, 1.0,
            ).astype(np.float32)
        qvel[int(model.jnt_dofadr[model.joint("rearwheel_joint").id])] = qvel[root_qvel] / float(cfg.wheel_roll_radius)
        qvel[int(model.jnt_dofadr[model.joint("frontwheel_joint").id])] = qvel[root_qvel] / float(cfg.wheel_roll_radius)
        qpos[hip_qpos], qpos[knee_qpos] = joints.hip, joints.knee
        qvel[hip_qvel], qvel[knee_qvel] = joints.hip_velocity, joints.knee_velocity
        if not (model.jnt_range[hip_joint, 0] <= qpos[hip_qpos] <= model.jnt_range[hip_joint, 1]
                and model.jnt_range[knee_joint, 0] <= qpos[knee_qpos] <= model.jnt_range[knee_joint, 1]):
            rejected["joint_limit"] += 1
            continue
        ctrl = np.asarray(jax.device_get(env._action_to_ctrl(jp.asarray(last_action), jp.asarray(qpos[knee_qpos]))))
        placement = solver.solve(qpos, qvel, ctrl)
        if not placement.accepted:
            rejected[f"placement:{placement.reason}"] += 1
            continue
        state = env.reset_from_snapshot(
            jp.asarray(placement.qpos, jp.float32), jp.asarray(qvel, jp.float32),
            jp.asarray(ctrl, jp.float32), jax.random.PRNGKey(args.seed + attempt),
            jp.asarray(STAGE_ID["takeoff"], jp.int32), jp.asarray(0, jp.int32),
            jp.asarray(0, jp.int32), jp.asarray(0, jp.int32),
            last_action=jp.asarray(last_action),
            stage_entry_ever=jp.asarray(0, jp.int32),
            apex_seen=jp.asarray(0, jp.int32),
            jump_signal_latched=jp.asarray(False),
            jump_window_start_x=jp.asarray(float(placement.qpos[root_qpos])),
            jump_window_end_x=jp.asarray(float(placement.qpos[root_qpos] + cfg.takeoff_window_far)),
        )
        if not all(np.isfinite(np.asarray(jax.device_get(value))).all() for value in (
            state.data.qpos, state.data.qvel, state.data.ctrl, state.obs["state"], state.obs["privileged_state"]
        )):
            rejected["nonfinite"] += 1
            continue
        if int(np.asarray(jax.device_get(state.info["phase"]))) != STAGE_ID["takeoff"]:
            rejected["phase"] += 1
            continue
        t0 = sample_from_state(env, state, float(qvel[root_qvel + 2]))
        t0["dual_wheel_airborne"] = False
        if evaluate_entry("takeoff", t0, cfg)["valid"]:
            rejected["premature_next_stage"] += 1
            continue
        hold_action = jp.asarray([0.0, 0.0, np.clip(_hip_hold_action(float(qpos[hip_qpos]), cfg), -1, 1), 0.0], jp.float32)
        probe = state
        shock_reason = None
        for _ in range(5):
            probe = step_fn(probe, hold_action)
            if float(np.asarray(jax.device_get(probe.done))) > 0.5:
                code = int(np.asarray(jax.device_get(probe.info["end_code"])))
                shock_reason = END_REASON.get(code, f"unknown_{code}")
                break
        if shock_reason is not None:
            rejected[f"shock:{shock_reason}"] += 1
            continue
        snapshot = env.snapshot_record(state, "takeoff")
        identity = hashlib.sha256(
            np.asarray(snapshot["qpos"], np.float32).tobytes()
            + np.asarray(snapshot["qvel"], np.float32).tobytes()
        ).hexdigest()
        if identity in identities:
            rejected["duplicate"] += 1
            continue
        identities.add(identity)
        snapshot.update({
            "id": hashlib.sha256(f"{args.seed}:{attempt}:{kind}:{index}".encode()).hexdigest()[:32],
            "candidate_kind": kind,
            "reference_index": index,
            "joint_state_source": joints.source,
            "joint_state_reference_index": None if kind == "canonical_compressed" else index,
            "reference_interval": reference_interval,
            "reference_interpolation_fraction": interpolation_fraction,
            "hip_qpos": float(qpos[hip_qpos]), "knee_qpos": float(qpos[knee_qpos]),
            "hip_qvel": float(qvel[hip_qvel]), "knee_qvel": float(qvel[knee_qvel]),
            "root_pose": np.asarray(placement.qpos[root_qpos:root_qpos + 7], np.float32),
            "root_velocity": np.asarray(qvel[root_qvel:root_qvel + 6], np.float32),
            "contact_summary": placement.summary(),
            "minimum_clearance_m": float(min(placement.wheel_clearance_min_m, placement.nonwheel_clearance_min_m)),
            "minimum_penetration_m": float(placement.minimum_penetration_m),
            "root_z_shift_m": float(placement.root_z_shift_m),
            "phase_detector_at_reset": "takeoff",
            "next_stage_at_reset": False,
            "shock_probe_steps": 5,
            "shock_probe_failure": None,
            "reset_protocol_sha256": protocol["sha256"],
            "generation_seed": int(args.seed),
            "proposal_index": int(attempt),
            "bootstrap_eligible": True,
            "training_only": False,
        })
        records.append(snapshot)
        accepted[kind] += 1

    status = "PASS" if len(records) == args.target and all(accepted[k] == v for k, v in quotas.items()) else "FAIL"
    metadata = {
        "artifact_role": "takeoff_reset_proposal_support_v3",
        "certified_tube": False,
        "safe_claim_allowed": False,
        "reset_protocol": protocol,
        "reset_protocol_sha256": protocol["sha256"],
        "xml_sha256": file_sha256(cfg.xml_path),
        "config_hash": config_hash(cfg),
        "reference_sha256": file_sha256(args.reference),
        "generation_seed": int(args.seed),
    }
    bank = SnapshotBank(records, metadata)
    bank.save(args.output)
    old = SnapshotBank.load(args.old_bank) if args.old_bank else SnapshotBank()
    report = {
        "status": status,
        "artifact_role": "takeoff_reset_authenticity_audit",
        "records": len(records),
        "quotas": quotas,
        "accepted": dict(accepted),
        "rejected": dict(rejected),
        "proposal_budget": int(args.proposal_budget),
        "compression_reference_indices": compressed_indices,
        "reset_legal_rate": float(len(records) / max(1, len(records) + sum(rejected.values()))),
        "joint_distribution_before": joint_distribution(old, hip_qpos, knee_qpos),
        "joint_distribution_after": joint_distribution(bank, hip_qpos, knee_qpos),
        "root_z": quantiles([row["qpos"][root_qpos + 2] for row in records]) if records else None,
        "root_z_shift": quantiles([row["root_z_shift_m"] for row in records]) if records else None,
        "minimum_penetration": quantiles([row["minimum_penetration_m"] for row in records]) if records else None,
        "wheel_contact_records": sum(row["contact_summary"]["wheel_terrain_contacts"] > 0 for row in records),
        "body_contact_records": sum(row["contact_summary"]["body_terrain_contacts"] > 0 for row in records),
        "premature_next_stage_records": sum(bool(row["next_stage_at_reset"]) for row in records),
        "shock_failure_records": sum(row["shock_probe_failure"] is not None for row in records),
        "bank": str(Path(args.output).resolve()),
        "bank_sha256": file_sha256(args.output),
        "reset_protocol": protocol,
    }
    save_json(args.report, report)
    print(json.dumps(report, indent=2))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
