"""Calibrate a small per-anchor continuous Descent-entry construction pilot."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cli.run_backward_descent_nominal_pilot import C_L, EXPECTED, PI_D, PI_L, _restore
from cli.run_backward_descent_rsi_pilot import certify_policy
from cli.run_descent_localized_consolidation_v1 import verified_assets_allowing_runtime_gate_refresh
from cli.runtime_gate import source_fingerprint
from dvgc.backward_search import compact_observation_command_adapter
from dvgc.bank import SnapshotBank, beta_posterior, posterior_label
from dvgc.config import file_sha256, load_config
from dvgc.descent_entry import DESCENT_ENTRY_FEATURE_NAMES, descent_entry_feature
from dvgc.entry import robust_normalization
from dvgc.env import OrangeBikeDVGC
from dvgc.local_entry import calibrate_local_radii
from dvgc.policy import load_bundle
from dvgc.runtime import save_json


TUBE = Path("runs/descent_natural_bridge_candidates_v1/independent_audit_2x32/descent_tube_v3.pkl")
CONSTRUCTION = Path("runs/descent_compact_matcher_neighborhood_v1/construction_24_adaptive/construction_bank.pkl")
EXPERT = Path("runs/descent_diverse_p1_predecessor_recovery_v5_p1_core_adapter")
DEFAULT_RUN = Path("runs/descent_local_entry_v1/pilot_4anchors")
SEED = 3_900_000_000
DELTAS = np.asarray([
    [-.010, 0.], [.010, 0.], [0., -.010], [0., .010],
    [-.007, -.007], [-.007, .007], [.007, -.007], [.007, .007],
], np.float32)


def select_pilot_anchors(records):
    """Outcome-frozen region coverage with one extra independent late anchor."""
    early = next(row for row in records if row.get("descent_region") == "early")
    middle = next(row for row in records
                  if row.get("descent_region") == "middle" and row["final"]["successes"] == row["final"]["branches"])
    late = next(row for row in records
                if row.get("descent_region") == "late" and row["final"]["successes"] == row["final"]["branches"])
    extension = next(row for row in records if row.get("target_distance") is not None)
    return [early, middle, late, extension]


def _perturb_record(env, record, delta, seed):
    base = _restore(env, record, jax.random.PRNGKey(seed)); info = base.info
    qvel = base.data.qvel.at[env._qvel0].add(delta[0]).at[env._qvel0 + 2].add(delta[1])
    state = env.reset_from_snapshot(
        base.data.qpos, qvel, base.data.ctrl, jax.random.PRNGKey(seed + 1),
        info["phase"], info["had_airborne"], info["had_valid_landing"], info["contact_age"], info["last_action"],
        estimated_phase=info["estimated_phase"], phase_probs=info["phase_probs"],
        airborne_count=info["airborne_count"], prelaunch_airborne_count=info["prelaunch_airborne_count"],
        landing_bounce_count=info["landing_bounce_count"], invalid_wheel_count=info["invalid_wheel_count"],
        recovery_count=info["recovery_count"], prev_acc_z=info["prev_acc_z"], prev_vz=info["prev_vz"],
        obs_history=info["actor_obs_history_pre"], obs_history_valid=jnp.asarray(True),
        stage_entry_ever=info["stage_entry_ever"], apex_seen=info["apex_seen"],
        jump_signal_latched=info["jump_signal_latched"], jump_window_start_x=info["jump_window_start_x"],
        jump_window_end_x=info["jump_window_end_x"],
    )
    return env.snapshot_record(state, "flight")


def _combined_exact_label(anchor, row, cfg):
    successes = int(anchor["final"]["successes"]) + int(row["final"]["successes"])
    branches = int(anchor["final"]["branches"]) + int(row["final"]["branches"])
    posterior = beta_posterior(successes, branches - successes)
    return posterior_label(
        posterior, branches, min_branches=int(cfg.min_branches), safe_threshold=float(cfg.safe_threshold),
        dead_threshold=float(cfg.dead_threshold), boundary_max_width=float(cfg.boundary_max_width),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=str(DEFAULT_RUN)); args = parser.parse_args(); root = Path(args.run)
    if root.exists(): raise SystemExit(f"refusing overwrite {root}")
    valid, failed, raw = verified_assets_allowing_runtime_gate_refresh()
    if not valid: raise SystemExit(f"frozen asset mismatch: {failed}; raw={raw}")
    gate = json.loads(Path("docs/RUNTIME_GATE.json").read_text())
    if gate.get("status") != "PASS" or gate.get("source_fingerprint") != source_fingerprint(Path.cwd()):
        raise SystemExit("runtime gate stale")
    tube = SnapshotBank.load(TUBE); anchors = select_pilot_anchors(tube.records)
    cfg = load_config("configs/backward_descent_rsi_pilot_v1.json", {
        "use_bank_resets": False, "expert_chain_termination": False,
        "domain_randomization": False, "obs_noise_enable": False,
    })
    artifact = pickle.loads((EXPERT / "adapter.pkl").read_bytes())
    dparams, _, _ = load_bundle(PI_D, verify_files=True); lparams, _, _ = load_bundle(PI_L, verify_files=True)
    env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank(), cert_bank=SnapshotBank.load(C_L))
    adapter = compact_observation_command_adapter(
        jnp.asarray(artifact["prototypes"]), jnp.asarray(artifact["targets"]),
        jnp.asarray(artifact["normalizer_mean"]), jnp.asarray(artifact["normalizer_std"]),
        float(artifact["radius"]), float(artifact["core_radius"]),
    )
    root.mkdir(parents=True)
    inputs = {"tube_sha256": file_sha256(TUBE), "construction_sha256": file_sha256(CONSTRUCTION),
              "adapter_sha256": file_sha256(EXPERT / "adapter.pkl"), "policy_identity_hash": artifact["policy_identity_hash"],
              "C_L": file_sha256(C_L), "xml": EXPECTED["xml"], "seed": SEED}
    save_json(root / "manifest.json", {"status": "FROZEN_BEFORE_OUTCOMES", "inputs": inputs,
              "anchor_ids": [row["id"] for row in anchors], "regions": [row.get("descent_region") or "late" for row in anchors],
              "deltas_vx_vz": DELTAS.tolist(), "selection": "first early, full-success middle/late, independent extension"})
    save_json(root / "cost_estimate.json", {"estimated_seconds": 1200, "anchors": 4, "local_states": 32,
              "rollouts_per_state": "2 exact + 4 micro", "maximum_rollouts": 192, "PPO_steps": 0})
    candidates = []; nodes = []
    for ai, anchor in enumerate(anchors):
        for di, delta in enumerate(DELTAS):
            seed = SEED + ai * 1000 + di * 10
            record = _perturb_record(env, anchor, delta, seed)
            identifier = hashlib.sha256(f"local-entry:{anchor['id']}:{di}:{SEED}".encode()).hexdigest()[:32]
            record.update({"id": identifier, "origin_anchor_id": anchor["id"], "anchor_index": ai,
                           "candidate_kind": "descent_local_entry_construction", "descent_region": anchor.get("descent_region") or "late",
                           "construction_delta_vx_vz": delta.tolist(), "artifact_role": "proposal_support_bank",
                           "safe_claim_allowed": False, "tube_metrics_eligible": False})
            candidates.append(record)
            nodes.append({"node_id": identifier, "candidate_id": anchor["id"], "layer": 0,
                          "region": record["descent_region"], "source_state_hash": record["state_byte_hash"],
                          "physical_state": record, "parent_node_id": anchor["id"]})
    result = certify_policy(env, dparams, lparams, nodes, SEED + 100_000,
                            record_loader=lambda node: node["physical_state"], descent_action_adapter=adapter,
                            policy_identity_hash=artifact["policy_identity_hash"])
    p1 = {row["node_id"] for row in result["rows"] if row["P1"]["pass"]}
    anchor_features = np.asarray([descent_entry_feature(row["physical_feature"], cfg) for row in anchors])
    _, scale = robust_normalization(np.asarray([row["entry_feature"] for row in tube.records]), cfg.descent_entry_scale_floors)
    calibration = [{"anchor_index": int(row["anchor_index"]),
                    "feature": descent_entry_feature(row["physical_feature"], cfg).tolist(), "safe": row["id"] in p1,
                    "source": "local_construction"} for row in candidates]
    prior = SnapshotBank.load(CONSTRUCTION).records
    for row in prior:
        feature = descent_entry_feature(row["physical_feature"], cfg)
        distances = np.linalg.norm((anchor_features - feature) / scale, axis=1); ai = int(np.argmin(distances))
        label = row["final"]["label"]
        if distances[ai] <= 1e-8:
            label = _combined_exact_label(anchors[ai], row, cfg)
        calibration.append({"anchor_index": ai, "feature": feature.tolist(), "safe": label == "safe",
                            "source": "prior_construction", "label": label})
    calibrated = calibrate_local_radii(anchor_features, calibration, scale, minimum_safe_per_anchor=4,
                                       minimum_precision=float(cfg.descent_entry_minimum_calibration_precision))
    active = calibrated["active_anchor_indices"]
    active_regions = {anchors[index].get("descent_region") or "late" for index in active}
    passed = calibrated["status"] == "PASS" and active_regions >= {"early", "middle", "late"}
    if passed:
        selected = [copy.deepcopy(anchors[index]) for index in active]
        radii = [calibrated["radii"][index] for index in active]
        raw = np.asarray([row["physical_feature"] for row in selected], float)
        matcher = {"version": "descent_local_entry_v1_construction_pilot", "feature_names": DESCENT_ENTRY_FEATURE_NAMES,
                   "center": np.zeros(16).tolist(), "scale": scale.tolist(), "radii": radii, "radius": max(radii),
                   "reference_envelope": {"x": {"min": float(raw[:, 0].min()), "max": float(raw[:, 0].max())},
                                          "z": {"min": float(raw[:, 2].min()), "max": float(raw[:, 2].max())}},
                   "envelope_tolerance_x": 0.0, "envelope_tolerance_z": 0.0,
                   "max_abs_roll_rate": 4.0, "max_abs_pitch_rate": 4.0,
                   "descent_vz_min": -2.5, "descent_vz_max": -0.05,
                   "construction_only": True, "minimum_precision": float(cfg.descent_entry_minimum_calibration_precision)}
        metadata = {"artifact_role": "proposal_support_bank", "safe_claim_allowed": False,
                    "continuous_matcher_active": False, "stage_entry_matcher": matcher,
                    "support_features": [row["physical_feature"] for row in selected], "inputs": inputs}
        SnapshotBank(selected, metadata).save(root / "local_entry_matcher_construction.pkl")
    SnapshotBank(candidates, {"artifact_role": "proposal_support_bank", "safe_claim_allowed": False,
                              "inputs": inputs}).save(root / "local_candidates.pkl")
    report = {"status": "PASS" if passed else "FAIL", "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "anchors": len(anchors), "local_states": len(candidates), "P0": result["P0"], "P1": result["P1"],
              "regions": {region: {"anchors": sum((row.get("descent_region") or "late") == region for row in anchors),
                                    "P1": sum(row["id"] in p1 and row["descent_region"] == region for row in candidates)}
                          for region in ("early", "middle", "late")},
              "calibration": calibrated, "active_regions": sorted(active_regions), "matcher_activated": False,
              "PPO_authorization": False, "next": "fresh_local_matcher_independent_audit" if passed else "local_entry_support_gap"}
    save_json(root / "DESCENT_LOCAL_ENTRY_PILOT_V1_REPORT.json", report)
    save_json(root / "completed.json", {"status": report["status"], "next": report["next"]})
    print(json.dumps({key: report[key] for key in ("status", "P0", "P1", "regions", "active_regions", "next")}
                     | {"precision": calibrated["precision"], "recall": calibrated["recall"]}, indent=2))


if __name__ == "__main__":
    main()
