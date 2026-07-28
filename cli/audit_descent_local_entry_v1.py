"""Fresh dynamics audit for the construction-only local Descent matcher."""
from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cli.audit_descent_compact_adapter_v1 import _perturb_batch
from cli.pilot_descent_local_entry_v1 import C_L, EXPECTED, EXPERT, PI_D, PI_L, _perturb_record
from cli.runtime_gate import source_fingerprint
from dvgc.backward_search import compact_observation_command_adapter, make_descent_landing_rollout
from dvgc.bank import SnapshotBank
from dvgc.certification import DYNAMICS_VARIANTS, branch_seed, detailed_terminal_summary
from dvgc.config import file_sha256, load_config
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.runtime import save_json
from dvgc.stage_reachability import support_distance


SOURCE = Path("runs/descent_local_entry_v1/pilot_4anchors_v2/local_entry_matcher_construction.pkl")
DEFAULT_RUN = Path("runs/descent_local_entry_v1/independent_audit_v1")
SEED = 4_000_000_000
NAMESPACE = "descent-local-entry-independent-audit-v1"
BRANCHES = 8


def audit_offsets(scale, radius):
    """Eight interior and four exterior points, fixed before outcomes."""
    vx_scale, vz_scale = float(scale[6]), float(scale[8])
    rows = []
    for index in range(8):
        angle = 2.0 * math.pi * index / 8.0
        factor = (.25, .50, .75, .90)[index % 4]
        rows.append((factor * radius * vx_scale * math.cos(angle),
                     factor * radius * vz_scale * math.sin(angle), "inside"))
    for index in range(4):
        angle = 2.0 * math.pi * (index + .5) / 4.0
        factor = 1.25
        rows.append((factor * radius * vx_scale * math.cos(angle),
                     factor * radius * vz_scale * math.sin(angle), "outside"))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=str(DEFAULT_RUN)); parser.add_argument("--source", default=str(SOURCE)); args = parser.parse_args(); root = Path(args.run); source_path = Path(args.source)
    if root.exists(): raise SystemExit(f"refusing overwrite {root}")
    gate = json.loads(Path("docs/RUNTIME_GATE.json").read_text())
    if gate.get("status") != "PASS" or gate.get("source_fingerprint") != source_fingerprint(Path.cwd()):
        raise SystemExit("runtime gate stale")
    frozen = {"C_L": file_sha256(C_L), "pi_D": file_sha256(PI_D / "params.pkl"),
              "pi_L": file_sha256(PI_L / "params.pkl"), "xml": file_sha256("assets/orange_bike_4kg_horizontal.xml")}
    expected = {"C_L": EXPECTED["C_L"], "pi_D": EXPECTED["pi_D"],
                "pi_L": EXPECTED["pi_L"], "xml": EXPECTED["xml"]}
    if frozen != expected: raise SystemExit(f"frozen scientific asset mismatch: {frozen}")
    source = SnapshotBank.load(source_path); matcher = source.metadata["stage_entry_matcher"]
    if not matcher.get("construction_only") or source.metadata.get("continuous_matcher_active"):
        raise SystemExit("source is not an inactive construction matcher")
    artifact = pickle.loads((EXPERT / "adapter.pkl").read_bytes())
    dparams, _, _ = load_bundle(PI_D, verify_files=True); lparams, _, _ = load_bundle(PI_L, verify_files=True)
    cfg0 = load_config("configs/backward_descent_rsi_pilot_v1.json", {
        "use_bank_resets": False, "expert_chain_termination": False,
        "domain_randomization": False, "obs_noise_enable": False,
    })
    generation_env = OrangeBikeDVGC(cfg0, snapshot_bank=SnapshotBank(), cert_bank=SnapshotBank.load(C_L))
    scale = np.asarray(matcher["scale"], float); radii = np.asarray(matcher["radii"], float)
    states = []
    for ai, (anchor, radius) in enumerate(zip(source.records, radii, strict=True)):
        for oi, (dvx, dvz, shell) in enumerate(audit_offsets(scale, radius)):
            state = _perturb_record(generation_env, anchor, np.asarray([dvx, dvz], np.float32), SEED + ai * 1000 + oi * 10)
            state.update({"id": f"audit-{ai:02d}-{oi:02d}", "anchor_index": ai, "audit_shell": shell,
                          "descent_region": anchor.get("descent_region") or "late"})
            states.append(state)
    root.mkdir(parents=True)
    inputs = {"construction_matcher_sha256": file_sha256(source_path), "adapter_sha256": file_sha256(EXPERT / "adapter.pkl"),
              "policy_identity_hash": artifact["policy_identity_hash"], **frozen, "seed": SEED,
              "seed_namespace": NAMESPACE, "branches_per_state": BRANCHES}
    save_json(root / "manifest.json", {"status": "FROZEN_BEFORE_AUDIT", "inputs": inputs,
              "states": len(states), "inside_per_anchor": 8, "outside_per_anchor": 4,
              "dynamics_variants": DYNAMICS_VARIANTS, "labels_reused_for_training": False})
    save_json(root / "cost_estimate.json", {"estimated_seconds": 2400, "states": len(states),
              "branches_per_state": BRANCHES, "rollouts": len(states) * BRANCHES, "PPO_steps": 0})
    variants = []
    for spec in DYNAMICS_VARIANTS:
        cfg = load_config("configs/backward_descent_rsi_pilot_v1.json", {
            "use_bank_resets": False, "expert_chain_termination": False, "domain_randomization": False,
            "obs_noise_enable": False, **{key: value for key, value in spec.items() if key != "id"},
        })
        env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank(), cert_bank=SnapshotBank.load(C_L))
        adapter = compact_observation_command_adapter(
            jnp.asarray(artifact["prototypes"]), jnp.asarray(artifact["targets"]),
            jnp.asarray(artifact["normalizer_mean"]), jnp.asarray(artifact["normalizer_std"]),
            float(artifact["radius"]), float(artifact["core_radius"]),
        )
        variants.append((spec["id"], env, make_descent_landing_rollout(
            env, dparams, lparams, horizon=200, residual_ticks=8, descent_action_adapter=adapter)))
    rows = []
    for state_index, record in enumerate(states):
        evidence = []
        for variant_index, (variant, env, rollout) in enumerate(variants):
            indices = list(range(variant_index, BRANCHES, len(variants))); seeds = []; deltas = []
            for branch_index in indices:
                seed = branch_seed(SEED, state_index, branch_index); seeds.append(seed)
                deltas.append(np.random.default_rng(seed).uniform(-.002, .002, 2).astype(np.float32))
            batch = _perturb_batch(env, record, seeds, deltas)
            features = np.asarray(jax.device_get(jax.vmap(env._physical_feature)(batch.data)))
            raw = jax.device_get(rollout(batch, jnp.zeros((len(seeds), 2, 4), jnp.float32),
                                        jax.random.PRNGKey(branch_seed(SEED, state_index, variant_index))))
            for local, branch_index in enumerate(indices):
                distance, predicted = support_distance(features[local], source.metadata)
                code = int(np.asarray(raw["end_code"])[local]); final = bool(np.asarray(raw["final_recovery"])[local])
                chain = bool(np.asarray(raw["downstream_entry"])[local])
                cause = "final_recovery" if final else ("timeout" if code == 8 else
                        ("horizon_exhausted" if code == 0 else "physical_failure"))
                evidence.append({"branch_index": branch_index, "branch_seed": seeds[local], "seed_namespace": NAMESPACE,
                                 "dynamics_variant": variant, "matcher_predicted": predicted, "matcher_distance": distance,
                                 "chain_success": chain, "final_recovery": final, "terminal_cause": cause,
                                 "end_code": code, "end_reason": END_REASON.get(code, "unknown"),
                                 "steps": int(np.asarray(raw["termination_tick"])[local])})
        evidence.sort(key=lambda row: row["branch_index"])
        rows.append({"id": record["id"], "anchor_index": record["anchor_index"],
                     "region": record["descent_region"], "audit_shell": record["audit_shell"], "branches": evidence})
        print(f"[local-entry-audit] {state_index + 1}/{len(states)}", flush=True)
    branches = [branch for row in rows for branch in row["branches"]]
    tp = sum(row["matcher_predicted"] and row["final_recovery"] for row in branches)
    fp = sum(row["matcher_predicted"] and not row["final_recovery"] for row in branches)
    fn = sum(not row["matcher_predicted"] and row["final_recovery"] for row in branches)
    precision = tp / (tp + fp) if tp + fp else 1.; recall = tp / (tp + fn) if tp + fn else 0.
    terminal = detailed_terminal_summary(branches)
    predicted_by_region = {region: sum(branch["matcher_predicted"] for row in rows if row["region"] == region for branch in row["branches"])
                           for region in ("early", "middle", "late")}
    passed = precision >= float(cfg0.descent_entry_minimum_calibration_precision) and all(predicted_by_region.values()) \
        and terminal["timeouts"] == terminal["horizon_exhaustions"] == terminal["physical_end_reasons"]["nonfinite"] == 0
    if passed:
        records = copy.deepcopy(source.records); metadata = copy.deepcopy(source.metadata)
        audited_matcher = copy.deepcopy(matcher); audited_matcher.update({"construction_only": False,
            "independent_audit_namespace": NAMESPACE, "independent_audit_precision": precision,
            "independent_audit_recall": recall})
        metadata.update({"artifact_role": "certified_stage_entry_region", "safe_claim_allowed": False,
                         "continuous_matcher_active": True, "stage_entry_matcher": audited_matcher,
                         "independent_audit": True, "audit_seed_namespace": NAMESPACE})
        SnapshotBank(records, metadata).save(root / "canonical_descent_local_entry_v1.pkl")
    report = {"status": "PASS" if passed else "FAIL", "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "states": len(states), "branches": len(branches), "precision": precision, "recall": recall,
              "true_positive": tp, "false_positive": fp, "false_negative": fn,
              "predicted_by_region": predicted_by_region, "terminal_summary": terminal, "rows": rows,
              "continuous_matcher_active": passed, "PPO_authorization": False,
              "entry_bank": str(root / "canonical_descent_local_entry_v1.pkl") if passed else None,
              "entry_bank_sha256": file_sha256(root / "canonical_descent_local_entry_v1.pkl") if passed else None,
              "next": "apex_parent_bridge_reprobe" if passed else "local_entry_independent_audit_failure"}
    save_json(root / "DESCENT_LOCAL_ENTRY_INDEPENDENT_AUDIT_V1_REPORT.json", report)
    save_json(root / "completed.json", {"status": report["status"], "next": report["next"]})
    print(json.dumps({key: report[key] for key in ("status", "states", "branches", "precision", "recall",
                                                    "false_positive", "predicted_by_region", "terminal_summary", "next")}, indent=2))


if __name__ == "__main__":
    main()
