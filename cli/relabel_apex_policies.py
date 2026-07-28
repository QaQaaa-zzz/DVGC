"""Minimal fixed-bank Apex->Descent relabel of frozen policy checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from dvgc.bank import SnapshotBank
from dvgc.certification import DYNAMICS_VARIANTS, branch_seed
from dvgc.config import ID_STAGE, file_sha256, load_config
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.rollout import restore_snapshot
from dvgc.runtime import build_inference, load_params, save_json
from dvgc.stage_reachability import evaluate_entry, protocol_payload, reachability_label


def evenly(rows: list[dict], count: int) -> list[dict]:
    ordered = sorted(rows, key=lambda r: (int(r.get("source_index", r.get("reference_index", 10**9))), r["id"]))
    if len(ordered) <= count: return ordered
    return [ordered[i] for i in np.linspace(0, len(ordered)-1, count, dtype=int)]


def state_sample(env, state, *, apex_seen: bool, previous_vz: float) -> tuple[dict, str]:
    snap = env.snapshot_record(state, ID_STAGE[int(np.asarray(jax.device_get(state.info["phase"])))])
    code = int(np.asarray(jax.device_get(state.info["end_code"])))
    reason = END_REASON.get(code, f"unknown_{code}")
    terminated = bool(int(np.asarray(jax.device_get(state.info["terminated"]))))
    f = np.asarray(snap["physical_feature"], np.float64)
    cfg = env._config
    terrain = float(cfg.step_top_z) if (cfg.step_front_x <= f[0] <= cfg.step_back_x and abs(f[1]) <= cfg.step_half_width) else 0.0
    airborne = bool(f[2] > terrain + float(cfg.nominal_base_z_ground) + float(cfg.imu_airborne_height_margin))
    failure = terminated and reason not in ("recovery", "chain_entry", "next_stage_entry")
    snap.update({"canonical_phase": snap["source_phase"], "previous_vz": previous_vz,
                 "apex_seen": bool(apex_seen), "dual_wheel_airborne": airborne,
                 "prohibited_contact": reason == "prohibited_contact",
                 "body_terrain_contact": reason == "prohibited_contact",
                 "deep_penetration": False, "invalid_wheel_contact": reason == "invalid_wheel_step_contact",
                 "physical_failure": failure, "nonfinite": not np.isfinite(f).all()})
    return snap, reason


def support_diagnostic(feature, support_metadata):
    matcher = support_metadata["stage_entry_matcher"]
    support = np.asarray(support_metadata["support_features"], np.float64)
    scale = np.asarray(matcher["scale"], np.float64)
    physical = np.asarray(feature, np.float64)
    raw = ((support - physical) / scale) ** 2
    distances = np.sqrt(raw.sum(axis=1))
    radii = np.asarray(matcher.get("radii", np.full(len(support), matcher["radius"])), np.float64)
    normalized = distances / radii
    index = int(np.argmin(normalized))
    return {"distance": float(normalized[index]), "anchor_index": index,
            "raw_feature_error": (physical - support[index]).tolist(),
            "squared_scaled_contributions": (raw[index] / (radii[index] ** 2)).tolist()}


def rollout(env, step, inference, row, support_metadata, seed: int, horizon: int, noise: float) -> dict:
    key = jax.random.PRNGKey(seed); state = restore_snapshot(env, row, key)
    previous_vz = float(np.asarray(jax.device_get(state.data.qvel[2])))
    apex_index = int(row.get("reference_index", row.get("source_index", -1)))
    reference_apex = int(row.get("reference_apex_index", 220))
    # Reset provenance is the only permitted initial latch.  A negative-vz
    # state without post-apex reference provenance is merely falling.
    apex_seen = apex_index >= reference_apex
    best_distance = float("inf"); best = None
    for tick in range(1, horizon + 1):
        key, ak, nk = jax.random.split(key, 3); action, _ = inference(state.obs, ak)
        if noise: action = jp.clip(action + noise * jax.random.normal(nk, action.shape), -1.0, 1.0)
        state = step(state, action)
        vz = float(np.asarray(jax.device_get(state.data.qvel[2])))
        apex_seen = apex_seen or previous_vz > 0.0 and vz <= 0.0
        sample, reason = state_sample(env, state, apex_seen=apex_seen, previous_vz=previous_vz)
        result = evaluate_entry("apex", sample, env._config, support_metadata)
        diagnostic = support_diagnostic(sample["physical_feature"], support_metadata)
        if diagnostic["distance"] < best_distance:
            best_distance = diagnostic["distance"]; best = {"tick": tick, **diagnostic,
                "feature": np.asarray(sample["physical_feature"]).tolist()}
        if result["valid"]:
            return {"success": True, "time_to_entry": tick, "entry_quality": result,
                    "entry_feature": np.asarray(sample["physical_feature"]).tolist(),
                    "termination_reason": "next_stage_entry", "minimum_support_distance": best_distance,
                    "closest_support": best}
        if float(np.asarray(jax.device_get(state.done))) > .5:
            return {"success": False, "time_to_entry": None, "entry_quality": result,
                    "termination_reason": reason, "minimum_support_distance": best_distance,
                    "closest_support": best}
        previous_vz = vz
    return {"success": False, "time_to_entry": None, "entry_quality": {},
            "termination_reason": "horizon_exhaustion", "minimum_support_distance": best_distance,
            "closest_support": best}


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--flight-bank", required=True)
    p.add_argument("--descent-support", required=True); p.add_argument("--policy", action="append", required=True)
    p.add_argument("--output", required=True); p.add_argument("--states", type=int, default=20)
    p.add_argument("--branches", type=int, default=4); p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--seed", type=int, default=971000000); p.add_argument("--action-noise", type=float, default=.03)
    p.add_argument("--config", default="configs/default.json"); a = p.parse_args()
    bank = SnapshotBank.load(a.flight_bank); support = SnapshotBank.load(a.descent_support)
    support_meta = dict(support.metadata); support_meta["support_features"] = [r["physical_feature"] for r in support.records]
    rows = evenly([r for r in bank.records if r.get("flight_subinterval") == "apex"], a.states)
    if len(rows) != a.states: raise SystemExit(f"Expected {a.states} Apex states, found {len(rows)}")
    policies = [Path(x) for x in a.policy]; seed_registry = set(); outcomes = []
    protocol = protocol_payload(load_config(a.config), support_meta)
    for variant_index, variant in enumerate(DYNAMICS_VARIANTS):
        cfg = load_config(a.config, {"training_stage": "flight", "use_bank_resets": False,
                                    "domain_randomization": False, "obs_noise_enable": False,
                                    **{k: v for k, v in variant.items() if k != "id"}})
        env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank()); step = jax.jit(env.step)
        for policy_index, policy in enumerate(policies):
            inference = build_inference(env, load_params(policy / "params.pkl"), deterministic=True)
            policy_hash = file_sha256(policy / "params.pkl")
            for state_index, row in enumerate(rows):
                for branch in range(a.branches):
                    # Four branches rotate all declared dynamics while keeping
                    # every controller/state seed globally disjoint.
                    if branch % len(DYNAMICS_VARIANTS) != variant_index: continue
                    seed = branch_seed(a.seed + policy_index * 10_000_000, state_index, branch)
                    if seed in seed_registry: raise RuntimeError("Seed collision")
                    seed_registry.add(seed)
                    result = rollout(env, step, inference, row, support_meta, seed, a.horizon, a.action_noise)
                    outcomes.append({"policy": str(policy.resolve()), "policy_hash": policy_hash,
                                     "candidate_id": row["id"], "reference_index": row.get("reference_index"),
                                     "parent": row.get("parent_anchor_id", row.get("reference_index", row["id"])),
                                     "branch": branch, "seed": seed, "seed_namespace": "apex_to_descent_relabel_v1",
                                     "dynamics_variant": variant["id"], **result})
    summaries = []
    for policy in policies:
        policy_hash = file_sha256(policy / "params.pkl"); subset = [r for r in outcomes if r["policy_hash"] == policy_hash]
        grouped = defaultdict(list)
        for row in subset: grouped[row["candidate_id"]].append(row)
        labels = [reachability_label(stage="apex", successes=sum(x["success"] for x in branches), branches=len(branches),
                                     branch_records=branches, controller_bank_exhausted=False)
                  for branches in grouped.values()]
        successful_ids = {r["candidate_id"] for r in subset if r["success"]}
        successful_parents = {str(r["parent"]) for r in subset if r["success"]}
        times = [r["time_to_entry"] for r in subset if r["success"]]
        terminal = Counter(r["termination_reason"] for r in subset if not r["success"])
        physical = sum(v for k, v in terminal.items() if k not in ("horizon_exhaustion", "stage_timeout"))
        summaries.append({"policy": str(policy.resolve()), "policy_hash": policy_hash, "branches": len(subset),
                          "successes": sum(r["success"] for r in subset), "reach_rate": sum(r["success"] for r in subset)/len(subset),
                          "successful_unique_states": len(successful_ids), "successful_parents": len(successful_parents),
                          "time_to_entry": {"min": min(times) if times else None, "median": float(np.median(times)) if times else None,
                                            "max": max(times) if times else None},
                          "physical_failures": physical, "termination_reasons": dict(terminal), "state_labels": labels})
    ranked = sorted(summaries, key=lambda x: (-x["successful_unique_states"], -x["reach_rate"], x["physical_failures"], x["policy_hash"]))
    complementary = [x["policy_hash"] for x in ranked if x["successful_unique_states"] > 0]
    save_json(a.output, {"status": "PASS", "artifact_role": "apex_to_descent_existing_policy_relabel",
                         "metric": "next_stage_reach", "not_a_c_l_gate": True, "not_certified_tube": True,
                         "protocol": protocol, "protocol_sha256": protocol["protocol_sha256"],
                         "flight_bank_sha256": file_sha256(a.flight_bank), "descent_support_sha256": file_sha256(a.descent_support),
                         "states": len(rows), "branches_per_policy_state": a.branches, "seed_count": len(seed_registry),
                         "seed_unique": len(seed_registry) == len(outcomes), "policies": summaries,
                         "selected_policy_hash": ranked[0]["policy_hash"], "complementary_policy_hashes": complementary,
                         "requires_apex_ppo": ranked[0]["successful_unique_states"] < 2, "outcomes": outcomes})
    print(json.dumps([{k: x[k] for k in ("policy_hash", "reach_rate", "successful_unique_states", "physical_failures")} for x in ranked], indent=2))


if __name__ == "__main__": main()
