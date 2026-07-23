"""Construct authentic Apex resets from legal anchors and dynamically reached events."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np
import pandas as pd

from cli.prepare_stage_controller_pilots import aligned_reference_anchors
from cli.search_takeoff_actions import SEQUENCES, action_at, reference_action_sequence
from cli.stage_label_pilot import sample_from_state
from dvgc.bank import SnapshotBank
from dvgc.config import STAGE_ID, file_sha256, load_config
from dvgc.env import OrangeBikeDVGC
from dvgc.reset_geometry import GroundSupportSolver
from dvgc.rollout import restore_snapshot
from dvgc.runtime import build_inference, load_params, save_json
from dvgc.stage_reachability import evaluate_entry


SEARCH_SEQUENCES = {
    "hold": SEQUENCES["hold"],
    "extend_half": SEQUENCES["extend_half"],
    "extend_full": SEQUENCES["extend_full"],
    "hip_full_knee_half": SEQUENCES["hip_full_knee_half"],
    "hip_half_knee_full": SEQUENCES["hip_half_knee_full"],
    "relax": [(80, [0., 0., -.35, -.35])],
}


def _state_hash(row):
    return hashlib.sha256(b"".join(
        np.ascontiguousarray(np.asarray(row[name], np.float32)).tobytes()
        for name in ("qpos", "qvel", "ctrl", "qacc_warmstart")
    )).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flight-bank", required=True)
    p.add_argument("--output-bank", required=True)
    p.add_argument("--output-report", required=True)
    p.add_argument("--policy", action="append", default=[], help="name=policy_dir")
    p.add_argument("--target", type=int, default=24)
    p.add_argument("--seed", type=int, default=10_300_000)
    p.add_argument("--horizon", type=int, default=100)
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--reference", default="data/reference_jump.csv")
    a = p.parse_args()
    if not 16 <= a.target <= 32:
        raise SystemExit("Apex v3 target must be 16..32")
    source = SnapshotBank.load(a.flight_bank)
    cfg = load_config(a.config, {
        "training_stage": "flight", "use_bank_resets": False,
        "domain_randomization": False, "obs_noise_enable": False,
        "stage_reachability_objective": "",
    })
    model = mujoco.MjModel.from_xml_path(str(cfg.xml_path))
    reference = pd.read_csv(a.reference)
    ascent, rejected_ascent = aligned_reference_anchors(
        [r for r in source.records if r.get("flight_subinterval") == "ascent"],
        reference, model, "ascent",
    )
    apex, rejected_apex = aligned_reference_anchors(
        [r for r in source.records if r.get("flight_subinterval") == "apex"],
        reference, model, "apex",
    )
    late_ascent = sorted(ascent, key=lambda row: int(row["reference_index"]))[-16:]
    anchors = sorted(apex, key=lambda row: int(row["reference_index"]))[:4]
    env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank())
    step = jax.jit(env.step)
    support = GroundSupportSolver(cfg.xml_path)
    hip_id = int(model.joint("hip_joint").id); knee_id = int(model.joint("knee_joint").id)
    hip_q = int(model.jnt_qposadr[hip_id]); knee_q = int(model.jnt_qposadr[knee_id])
    controllers = []
    for name, sequence in SEARCH_SEQUENCES.items():
        controllers.append((f"bounded:{name}", None, sequence))
    controllers.append(("bounded:reference_time_aligned", None, None))
    for spec in a.policy:
        name, path = spec.split("=", 1)
        infer = build_inference(env, load_params(Path(path) / "params.pkl"), deterministic=True)
        controllers.append((f"policy:{name}", infer, None))
    scale = np.asarray(source.metadata.get("augmentation_feature_scale", np.ones(16)), float)
    scale = np.maximum(scale, np.asarray([.05, .05, .05, .05, .05, .05, .2, .2, .2,
                                          .2, .2, .2, .1, .1, .2, .2]))
    rows = []
    rejection = Counter()
    for row in anchors:
        item = copy.deepcopy(row)
        item.update({
            "candidate_kind": "apex_reference_anchor_reset_valid",
            "apex_support_class": "reset_valid_candidate",
            "dynamically_reached": False,
            "trajectory_parent_id": f"reference:{row['reference_index']}",
            "generation_seed": None,
            "generation_controller": None,
        })
        rows.append(item)
    for pi, parent in enumerate(late_ascent):
        for ci, (name, infer, sequence) in enumerate(controllers):
            if len(rows) >= a.target:
                break
            seed = a.seed + pi * 1000 + ci * 10
            key = jax.random.PRNGKey(seed)
            state = restore_snapshot(env, parent, key)
            previous_vz = float(np.asarray(state.data.qvel[2]))
            captured = None; action_history = []
            for tick in range(a.horizon):
                key, action_key, noise_key = jax.random.split(key, 3)
                if infer is not None:
                    action, _ = infer(state.obs, action_key)
                else:
                    spec = (reference_action_sequence(reference, parent, a.horizon)
                            if sequence is None else sequence)
                    action = action_at(spec, tick)
                action = jp.clip(action + .015 * jax.random.normal(noise_key, (4,)), -1, 1)
                action_history.append(np.asarray(action).tolist())
                state = step(state, action)
                sample = sample_from_state(env, state, previous_vz)
                entry = evaluate_entry("ascent", sample, cfg)
                if entry["valid"] and float(sample["physical_feature"][8]) > -.05:
                    captured = (tick + 1, state, sample); break
                if float(np.asarray(state.done)) > .5:
                    break
                previous_vz = float(sample["physical_feature"][8])
            if captured is None:
                rejection["did_not_reach_apex_event"] += 1; continue
            tick, state, sample = captured
            snap = env.snapshot_record(state, "flight")
            qpos = np.asarray(snap["qpos"]); qvel = np.asarray(snap["qvel"])
            contact = support.measure(qpos, qvel, snap["ctrl"])
            reason = None
            if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
                reason = "nonfinite"
            elif not (model.jnt_range[hip_id, 0] <= qpos[hip_q] <= model.jnt_range[hip_id, 1]
                      and model.jnt_range[knee_id, 0] <= qpos[knee_q] <= model.jnt_range[knee_id, 1]):
                reason = "joint_limit"
            elif contact["body_contacts"] or contact["wheel_contacts"]:
                reason = "terrain_contact"
            elif int(np.asarray(state.info["phase"])) != STAGE_ID["flight"]:
                reason = "wrong_phase"
            elif float(sample["physical_feature"][8]) <= -.05:
                reason = "premature_descent"
            if reason:
                rejection[reason] += 1; continue
            # Restoring the exact snapshot must not create an immediate 5-step
            # failure.  This does not settle or mutate the candidate.
            probe = restore_snapshot(env, snap, jax.random.PRNGKey(seed + 500_000))
            shock = False
            for _ in range(5):
                probe = step(probe, jp.zeros(4, jp.float32))
                if float(np.asarray(probe.done)) > .5:
                    shock = True; break
            if shock:
                rejection["five_step_reset_shock"] += 1; continue
            feature = np.asarray(snap["physical_feature"], float)
            if any(np.linalg.norm((feature - np.asarray(old["physical_feature"])) / scale) < .05
                   for old in rows):
                rejection["normalized_duplicate"] += 1; continue
            snap.update({
                "id": _state_hash(snap)[:32],
                "candidate_kind": "apex_dynamically_reached",
                "flight_subinterval": "apex", "reference_index": None,
                "trajectory_parent_id": f"{parent['id']}:{name}:{seed}",
                "source_parent_id": parent["id"],
                "source_reference_index": parent.get("reference_index"),
                "apex_support_class": "dynamically_reached_candidate",
                "dynamically_reached": True,
                "generation_seed": seed, "generation_controller": name,
                "generation_entry_tick": tick, "generation_actions": action_history,
                "contact_summary": contact,
                "five_step_reset_shock": False,
            })
            rows.append(snap)
        if len(rows) >= a.target:
            break
    dynamic = [row for row in rows if row.get("dynamically_reached")]
    parent_count = len({row["source_parent_id"] for row in dynamic})
    status = "PASS" if len(rows) >= 16 and parent_count >= 4 else "FAIL"
    protocol = {
        "version": "apex_reset_authenticity_v3",
        "source_bank_sha256": file_sha256(a.flight_bank),
        "xml_sha256": file_sha256(cfg.xml_path),
        "reference_sha256": file_sha256(a.reference),
        "generation_seed": a.seed,
        "target": a.target,
        "reset_valid_is_not_reachability_evidence": True,
    }
    protocol["sha256"] = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    SnapshotBank(rows, {
        "artifact_role": "apex_reset_proposal_bank_v3",
        "certified_tube": False, "safe_claim_allowed": False,
        "reset_protocol": protocol, "reset_protocol_sha256": protocol["sha256"],
        "augmentation_feature_scale": scale.tolist(),
    }).save(a.output_bank)
    save_json(a.output_report, {
        "status": status, "artifact_role": "apex_reset_bank_v3_construction",
        "bank": str(Path(a.output_bank).resolve()), "bank_sha256": file_sha256(a.output_bank),
        "records": len(rows), "reference_reset_valid": len(anchors),
        "dynamically_reached": len(dynamic), "dynamic_parent_count": parent_count,
        "source_parent_count": len(late_ascent),
        "rejections": dict(rejection), "protocol": protocol,
        "reference_alignment_rejections": {"ascent": rejected_ascent, "apex": rejected_apex},
        "training_authorized": status == "PASS",
    })
    print(json.dumps({"status": status, "records": len(rows), "dynamic": len(dynamic),
                      "parents": parent_count, "rejections": dict(rejection)}, indent=2))


if __name__ == "__main__":
    main()
