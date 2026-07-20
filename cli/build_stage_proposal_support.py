"""Build the frozen Descent proposal support used by Apex reachability.

This artifact is deliberately not a certified Tube.  It merges immutable,
physically legal Descent snapshots and retains controller-conditioned outcome
evidence only as provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import mujoco
import numpy as np

from dvgc.bank import SnapshotBank
from dvgc.config import ACTION_MAPPING_VERSION, AUTHORITATIVE_XML_SHA256, config_hash, file_sha256, load_config
from dvgc.runtime import save_json

FEATURE_NAMES = ("x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz",
                 "wx", "wy", "wz", "steer", "hip", "knee", "rearwheel_velocity")
DEFAULT_INPUTS = (
    "artifacts/flight_candidates_augmented_v1.pkl",
    "runs/jump_envelope_seed0_20260719/support/descent_proposal_support.pkl",
    "runs/stage_experts/descent_tube_seed0_20260716T2330/round_3/frozen/D_all_unique.pkl",
    "runs/stage_experts/trajectory_mining_resume_seed0_20260718T194327/trajectory_mining_corrected/candidate_pool.pkl",
)


def state_hash(row: dict) -> str:
    return hashlib.sha256(b"".join(np.ascontiguousarray(np.asarray(row[k], np.float32)).tobytes()
                                   for k in ("qpos", "qvel", "ctrl", "qacc_warmstart"))).hexdigest()


def contact_flags(model: mujoco.MjModel, data: mujoco.MjData, row: dict, deep=-0.005) -> dict:
    data.qpos[:] = row["qpos"]; data.qvel[:] = row["qvel"]; data.ctrl[:] = row["ctrl"]
    mujoco.mj_forward(model, data)
    enabled = lambda g: bool(int(model.geom_contype[g]) or int(model.geom_conaffinity[g]))
    terrain = {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == 0 and enabled(g)}
    wheel_bodies = {model.body(name).id for name in ("frontwheel", "rearwheel")}
    wheel = {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in wheel_bodies and enabled(g)}
    robot = {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) != 0 and enabled(g)}
    body = robot - wheel
    any_contact = body_contact = deep_contact = False
    for i in range(data.ncon):
        c = data.contact[i]; a, b = int(c.geom1), int(c.geom2)
        rt = (a in robot and b in terrain) or (b in robot and a in terrain)
        if not rt: continue
        any_contact = True
        body_contact |= (a in body and b in terrain) or (b in body and a in terrain)
        deep_contact |= float(c.dist) < deep
    return {"robot_terrain_contact": any_contact, "body_terrain_contact": body_contact,
            "deep_penetration": deep_contact}


def robust_matcher(rows: list[dict], cfg, bank_hashes: list[str]) -> dict:
    f = np.asarray([r["physical_feature"] for r in rows], np.float64)
    center = np.median(f, axis=0)
    mad = np.median(np.abs(f - center), axis=0) * 1.4826
    std = f.std(axis=0)
    floors = np.asarray(cfg.descent_entry_scale_floors, np.float64)
    scale = np.maximum(np.maximum(mad, .25 * std), floors)
    z = (f - center) / scale
    distance = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1)
    np.fill_diagonal(distance, np.inf)
    nn = distance.min(axis=1) if len(rows) > 1 else np.asarray([1.0])
    radius = float(np.clip(np.quantile(nn, .95) * 1.25, 1.0, 2.5))
    envelope = {name: {"min": float(f[:, i].min()), "max": float(f[:, i].max())}
                for i, name in enumerate(FEATURE_NAMES)}
    payload = {"version": "descent_support_matcher_v1", "feature_names": FEATURE_NAMES,
               "center": center.tolist(), "scale": scale.tolist(), "radius": radius,
               "reference_envelope": envelope, "source_bank_hashes": bank_hashes,
               "max_abs_roll_rate": float(cfg.recovery_max_angvel),
               "max_abs_pitch_rate": float(cfg.recovery_max_angvel),
               "descent_vz_min": -2.5, "descent_vz_max": -0.05,
               "envelope_tolerance_z": 0.08, "envelope_tolerance_x": 0.20,
               "derivation": "robust physical scales; 1.25*p95 support NN clipped to [1,2.5]"}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["matcher_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    payload["nearest_neighbor_distance"] = {"min": float(nn.min()), "p50": float(np.median(nn)),
                                              "p95": float(np.quantile(nn, .95)), "max": float(nn.max())}
    return payload


def build(inputs: list[Path], output: Path, report: Path, config: str) -> dict:
    cfg = load_config(config)
    if file_sha256(cfg.xml_path) != AUTHORITATIVE_XML_SHA256:
        raise RuntimeError("Authoritative XML hash mismatch")
    model = mujoco.MjModel.from_xml_path(str(cfg.xml_path)); data = mujoco.MjData(model)
    unique: dict[str, dict] = {}; sources = []; rejected = Counter(); evidence = defaultdict(list)
    for path in inputs:
        if not path.exists():
            sources.append({"path": str(path), "exists": False}); continue
        bank = SnapshotBank.load(path); digest = file_sha256(path)
        sources.append({"path": str(path.resolve()), "exists": True, "sha256": digest,
                        "records": len(bank.records), "source_role": bank.metadata.get("artifact_role", bank.metadata.get("entry_bank_role"))})
        for source_index, original in enumerate(bank.records):
            row = dict(original)
            if row.get("source_phase") != "flight" or int(row.get("oracle_phase", -1)) != 2 or row.get("flight_subinterval") != "descent":
                rejected["not_descent"] += 1; continue
            if not bool(row.get("bootstrap_eligible", True)) or row.get("final", {}).get("label") == "dead" or row.get("empirical_label") == "dead":
                rejected["dead_or_ineligible"] += 1; continue
            if not all(np.isfinite(np.asarray(row[k])).all() for k in ("qpos", "qvel", "ctrl", "qacc_warmstart", "physical_feature")):
                rejected["nonfinite"] += 1; continue
            flags = contact_flags(model, data, row)
            if flags["robot_terrain_contact"] or flags["deep_penetration"]:
                rejected["terrain_contact"] += 1; continue
            sid = state_hash(row)
            evidence[sid].append({"source_path": str(path.resolve()), "source_sha256": digest,
                                  "source_record_index": source_index, "source_record_id": row.get("id")})
            if sid in unique: continue
            row.update({"state_byte_hash": sid, "artifact_role": "descent_proposal_support_v1",
                        "candidate_kind": "descent_proposal_support", "certified_safe": False,
                        "tube_metrics_eligible": False, "safe_claim_allowed": False,
                        "training_only": False, "bootstrap_eligible": True,
                        "current_geometry_audit": flags,
                        "origin_reference_index": row.get("reference_index", row.get("entry_source_reference_index")),
                        "origin_parent": row.get("trajectory_parent_id", row.get("parent_candidate_id", row.get("entry_source_id"))),
                        "origin_dynamics_variant": row.get("dynamics_variant", row.get("entry_source_dynamics_variant")),
                        "source_controller": row.get("entry_source_policy", row.get("policy_version"))})
            unique[sid] = row
    rows = list(unique.values())
    if not rows: raise RuntimeError("No legal Descent proposal support")
    for row in rows: row["support_provenance"] = evidence[row["state_byte_hash"]]
    xs = np.asarray([float(r["physical_feature"][0]) for r in rows])
    q1, q2 = np.quantile(xs, (1/3, 2/3))
    for row in rows:
        x = float(row["physical_feature"][0])
        row["descent_support_region"] = "early" if x <= q1 else "middle" if x <= q2 else "late"
    hashes = [x["sha256"] for x in sources if x.get("exists")]
    matcher = robust_matcher(rows, cfg, hashes)
    metadata = {"artifact_role": "descent_proposal_support_v1", "certified_tube": False,
                "safe_claim_allowed": False, "tube_metrics_eligible": False,
                "xml_sha256": file_sha256(cfg.xml_path), "config_hash": config_hash(cfg),
                "action_mapping_version": ACTION_MAPPING_VERSION, "sources": sources,
                "stage_entry_matcher": matcher,
                "support_features": [np.asarray(r["physical_feature"], np.float32).tolist() for r in rows]}
    SnapshotBank(rows, metadata).save(output)
    layers = Counter(str(r["descent_support_region"]) for r in rows)
    kinds = Counter(str(r.get("candidate_kind", "missing")) for r in rows)
    payload = {"status": "PASS", "artifact_role": "descent_proposal_support_v1_build_report",
               "bank": str(output.resolve()), "bank_sha256": file_sha256(output), "records": len(rows),
               "state_byte_unique": len({r["state_byte_hash"] for r in rows}) == len(rows),
               "sources": sources, "rejections": dict(rejected), "descent_layers": dict(layers),
               "candidate_kinds": dict(kinds), "matcher": matcher,
               "claims": {"certified_tube": False, "safe": False, "proposal_support_only": True}}
    save_json(report, payload); return payload


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--input", action="append")
    p.add_argument("--output", required=True); p.add_argument("--report", required=True)
    p.add_argument("--config", default="configs/default.json"); a = p.parse_args()
    result = build([Path(x) for x in (a.input or DEFAULT_INPUTS)], Path(a.output), Path(a.report), a.config)
    print(json.dumps({k: result[k] for k in ("status", "records", "bank_sha256", "rejections", "descent_layers")}, indent=2))


if __name__ == "__main__": main()
