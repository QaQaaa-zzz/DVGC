"""Bounded local controllability search for Ascent->Apex or Apex->Descent."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import pandas as pd

from cli.search_takeoff_actions import SEQUENCES, action_at, reference_action_sequence
from cli.stage_label_pilot import sample_from_state
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.rollout import restore_snapshot
from dvgc.runtime import build_inference, load_params, save_json
from dvgc.stage_reachability import evaluate_entry

FEATURE_NAMES = (
    "x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz",
    "wx", "wy", "wz", "steer", "hip", "knee", "rearwheel_velocity",
)

LOCAL = {
    "hold": SEQUENCES["hold"],
    "extend_half": SEQUENCES["extend_half"],
    "extend_full": SEQUENCES["extend_full"],
    "hip_full_knee_half": SEQUENCES["hip_full_knee_half"],
    "hip_half_knee_full": SEQUENCES["hip_half_knee_full"],
    "drive_extend_full": SEQUENCES["drive_extend_full"],
    "relax": [(50, [0., 0., -0.35, -0.35])],
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("ascent", "apex"), required=True)
    p.add_argument("--bank", required=True)
    p.add_argument("--support-bank", default="")
    p.add_argument("--policy", action="append", default=[], help="name=policy_dir")
    p.add_argument("--output", required=True)
    p.add_argument("--horizon", type=int, default=80)
    p.add_argument("--seed", type=int, default=10_500_000)
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--reference", default="data/reference_jump.csv")
    a = p.parse_args()
    bank = SnapshotBank.load(a.bank)
    support = SnapshotBank.load(a.support_bank) if a.support_bank else None
    support_metadata = dict(support.metadata) if support else None
    if support_metadata is not None:
        support_metadata["support_features"] = [row["physical_feature"] for row in support.records]
    cfg = load_config(a.config, {
        "training_stage": "flight", "use_bank_resets": False,
        "domain_randomization": False, "obs_noise_enable": False,
    })
    env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank(), stage_support_bank=support)
    step = jax.jit(env.step)
    reference = pd.read_csv(a.reference)
    controllers = []
    for name, sequence in LOCAL.items():
        controllers.append((f"bounded:{name}", lambda state, key, tick, s=sequence: action_at(s, tick)))
    controllers.append(("bounded:reference_time_aligned", None))
    for spec in a.policy:
        name, path = spec.split("=", 1)
        infer = build_inference(env, load_params(Path(path) / "params.pkl"), deterministic=True)
        controllers.append((f"policy:{name}", lambda state, key, tick, f=infer: f(state.obs, key)[0]))
    outcomes = []
    success_parents = set()
    matcher_audit = None
    if support_metadata is not None:
        matcher = support_metadata.get("stage_entry_matcher", {})
        names = tuple(matcher.get("feature_names", ()))
        center = np.asarray(matcher.get("center", []), float)
        scale = np.asarray(matcher.get("scale", []), float)
        support_features = np.asarray(support_metadata["support_features"], float)
        matcher_audit = {
            "feature_names": list(names),
            "feature_order_matches_physical_feature": names == FEATURE_NAMES,
            "dimensions": len(names),
            "center_dimensions": int(center.size),
            "scale_dimensions": int(scale.size),
            "support_feature_dimensions": (
                int(support_features.shape[1]) if support_features.ndim == 2 else None
            ),
            "normalization": "(feature-center)/scale; Euclidean nearest-neighbour",
            "angular_handling": "direct_linear_difference_in_current_frozen_matcher",
            "radius": matcher.get("radius"),
            "matcher_sha256": matcher.get("matcher_sha256"),
        }
    for i, row in enumerate(bank.records):
        parent = row.get(
            "independent_trajectory_parent_id",
            row.get("source_parent_id", row.get("trajectory_parent_id", row.get("id"))),
        )
        for ci, (name, action_fn) in enumerate(controllers):
            state = restore_snapshot(env, row, jax.random.PRNGKey(a.seed + i * 100 + ci))
            previous_vz = float(np.asarray(state.data.qvel[2]))
            minimum_distance = float("inf"); trace = []; reason = "horizon_exhaustion"
            success = False; entry_tick = None
            for tick in range(a.horizon):
                key = jax.random.PRNGKey(a.seed + i * 10000 + ci * 100 + tick)
                if action_fn is None:
                    sequence = reference_action_sequence(reference, row, a.horizon)
                    action = action_at(sequence, tick)
                else:
                    action = action_fn(state, key, tick)
                state = step(state, jp.asarray(action, jp.float32))
                sample = sample_from_state(env, state, previous_vz)
                entry = evaluate_entry(a.stage, sample, cfg, support_metadata)
                distance = entry.get("support_distance")
                if distance is not None:
                    minimum_distance = min(minimum_distance, float(distance))
                feature = sample["physical_feature"]
                nearest_id = None
                distance_contribution = None
                if support_metadata is not None:
                    z = ((np.asarray(feature, float) - center) / scale)
                    support_z = (support_features - center) / scale
                    squared = np.square(support_z - z[None, :])
                    nearest = int(np.argmin(np.sum(squared, axis=1)))
                    nearest_id = support.records[nearest]["id"]
                    distance_contribution = dict(zip(
                        FEATURE_NAMES, squared[nearest].tolist()
                    ))
                trace.append({
                    "tick": tick + 1, "vertical_velocity": float(feature[8]),
                    "roll": float(feature[3]), "pitch": float(feature[4]),
                    "angular_velocity": np.asarray(feature[9:12]).tolist(),
                    "hip": float(feature[12]), "knee": float(feature[13]),
                    "action": np.asarray(action).tolist(),
                    "support_distance": None if distance is None else float(distance),
                    "nearest_support_id": nearest_id,
                    "squared_distance_contribution": distance_contribution,
                    "apex_seen": bool(sample.get("apex_seen")), "valid_entry": bool(entry["valid"]),
                })
                if entry["valid"]:
                    success = True; entry_tick = tick + 1; reason = "next_stage_entry"
                    success_parents.add(parent); break
                if float(np.asarray(state.done)) > .5:
                    code = int(np.asarray(state.info["end_code"]))
                    reason = END_REASON.get(code, f"unknown_{code}"); break
                previous_vz = float(feature[8])
            outcomes.append({
                "candidate_id": row["id"], "trajectory_parent": parent,
                "diagnostic_stratum": row.get("diagnostic_stratum", row.get("flight_subinterval")),
                "controller": name, "success": success, "entry_tick": entry_tick,
                "minimum_support_distance": None if np.isinf(minimum_distance) else minimum_distance,
                "failure_reason": None if success else reason,
                "action_saturation_fraction": float(np.mean([
                    np.mean(np.abs(t["action"]) >= .999) for t in trace
                ])) if trace else 0., "trace": trace,
            })
    strata = {}
    for name in sorted({row["diagnostic_stratum"] for row in outcomes}):
        rows = [row for row in outcomes if row["diagnostic_stratum"] == name]
        ids = {row["candidate_id"] for row in rows}
        strata[str(name)] = {
            "states": len(ids), "successful_unique_states": len({
                row["candidate_id"] for row in rows if row["success"]
            }), "successful_branches": sum(row["success"] for row in rows),
            "termination_reasons": dict(Counter(
                "next_stage_entry" if row["success"] else row["failure_reason"] for row in rows
            )),
        }
    save_json(a.output, {
        "status": "PASS", "artifact_role": f"{a.stage}_bounded_local_controllability",
        "stage": a.stage, "bank": str(Path(a.bank).resolve()),
        "bank_sha256": file_sha256(a.bank), "support_bank_sha256":
        file_sha256(a.support_bank) if a.support_bank else None,
        "states": len(bank.records), "controllers": len(controllers),
        "successful_unique_states": len({row["candidate_id"] for row in outcomes if row["success"]}),
        "successful_parent_count": len(success_parents),
        "strata": strata, "matcher_audit": matcher_audit, "outcomes": outcomes,
        "failure_semantics": "negative_under_bounded_controller_bank_only",
    })
    print(json.dumps({"states": len(bank.records), "successful_parents": len(success_parents),
                      "strata": strata}, indent=2))


if __name__ == "__main__":
    main()
