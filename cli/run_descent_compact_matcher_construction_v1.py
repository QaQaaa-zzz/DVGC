"""Expand construction-only C_D support under the frozen compact expert."""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cli.audit_descent_compact_adapter_v1 import _perturb_batch
from cli.run_backward_descent_nominal_pilot import C_L, EXPECTED, PI_D, PI_L, _load_record
from cli.run_descent_localized_consolidation_v1 import verified_assets_allowing_runtime_gate_refresh
from cli.runtime_gate import source_fingerprint
from dvgc.backward_search import compact_observation_command_adapter, make_descent_landing_rollout
from dvgc.bank import SnapshotBank, beta_posterior, posterior_label
from dvgc.certification import DYNAMICS_VARIANTS, branch_seed, detailed_terminal_summary
from dvgc.config import ACTION_MAPPING_VERSION, file_sha256, load_config
from dvgc.descent_entry import descent_entry_feature
from dvgc.entry import robust_normalization
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.runtime import save_json


TUBE = Path("runs/descent_diverse_p1_predecessor_recovery_v5_p1_core_adapter/independent_audit_v1/descent_tube_v2.pkl")
EXPERT = Path("runs/descent_diverse_p1_predecessor_recovery_v5_p1_core_adapter")
INDEX = Path("runs/backward_recovery_tube_fast_track_v1/proposal_state_index.json")
DEFAULT_RUN = Path("runs/descent_compact_matcher_neighborhood_v1/construction_24_adaptive")
SEED_BASE = 3_300_000_000
SEED_NAMESPACE = "descent-compact-matcher-construction-v1"


def select_neighborhood(rows, per_region=8):
    """Nearest geometry strata with global parent diversity and no outcomes."""
    selected, used_parents = [], set()
    for region in ("early", "middle", "late"):
        ordered = sorted((row for row in rows if row["region"] == region),
                         key=lambda row: (row["tube_distance"], row["candidate_id"], row["proposal_id"]))
        region_rows = []
        for row in ordered:
            if row["candidate_id"] in used_parents:
                continue
            region_rows.append(row)
            used_parents.add(row["candidate_id"])
            if len(region_rows) == per_region:
                break
        if len(region_rows) != per_region:
            raise ValueError(f"insufficient globally distinct {region} parents")
        selected.extend(region_rows)
    return selected


def next_budget(successes, branches, cfg):
    """Pre-registered 4 -> 8 -> 16/32 construction funnel."""
    if branches < 8:
        return 8
    label = outcome(successes, branches, cfg)["label"]
    if branches == 8:
        if label == "dead":
            return branches
        return 32 if label == "safe" else 16
    if branches == 16:
        return 32 if label in ("safe", "unknown") else branches
    return branches


def outcome(successes, branches, cfg):
    posterior = beta_posterior(successes, branches-successes, alpha0=cfg.beta_alpha0,
                               beta0=cfg.beta_beta0, q_low=cfg.posterior_q_low, q_high=cfg.posterior_q_high)
    label = posterior_label(posterior, branches, min_branches=int(cfg.min_branches),
                            safe_threshold=float(cfg.safe_threshold), dead_threshold=float(cfg.dead_threshold),
                            boundary_max_width=float(cfg.boundary_max_width))
    return {"successes": int(successes), "failures": int(branches-successes), "branches": int(branches),
            "posterior": posterior, "label": label}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=str(DEFAULT_RUN))
    args = parser.parse_args(); root = Path(args.run)
    if (root/"completed.json").exists():
        raise SystemExit(f"refusing overwrite completed run {root}")
    valid, failed, raw = verified_assets_allowing_runtime_gate_refresh()
    if not valid:
        raise SystemExit(f"frozen asset mismatch: {failed}; raw={raw}")
    gate = json.loads(Path("docs/RUNTIME_GATE.json").read_text())
    if gate.get("status") != "PASS" or gate.get("source_fingerprint") != source_fingerprint(Path.cwd()):
        raise SystemExit("runtime gate stale")
    artifact = pickle.loads((EXPERT/"adapter.pkl").read_bytes())
    if artifact["base_policy_sha256"] != EXPECTED["pi_D"]:
        raise SystemExit("base policy mismatch")
    dparams, _, _ = load_bundle(PI_D, verify_files=True); lparams, _, _ = load_bundle(PI_L, verify_files=True)
    base = load_config("configs/backward_descent_rsi_pilot_v1.json", {"use_bank_resets": False,
        "expert_chain_termination": False, "domain_randomization": False, "obs_noise_enable": False})
    if file_sha256(base.xml_path) != EXPECTED["xml"] or base.action_mapping_version != ACTION_MAPPING_VERSION:
        raise SystemExit("runtime model mismatch")
    tube = SnapshotBank.load(TUBE); features = np.asarray([row["entry_feature"] for row in tube.records], float)
    center, scale = robust_normalization(features, base.descent_entry_scale_floors)
    pool = []
    for proposal in json.loads(INDEX.read_text())["rows"]:
        record = _load_record(proposal); feature = descent_entry_feature(record["physical_feature"], base)
        distance = float(np.min(np.linalg.norm((features-feature[None, :])/scale, axis=1)))
        pool.append({**proposal, "tube_distance": distance})
    selected = select_neighborhood(pool); root.mkdir(parents=True, exist_ok=True)
    inputs = {"policy_identity_hash": artifact["policy_identity_hash"], "adapter_sha256": file_sha256(EXPERT/"adapter.pkl"),
        "tube_sha256": file_sha256(TUBE), "proposal_index_sha256": file_sha256(INDEX), "C_L": EXPECTED["C_L"],
        "pi_L": EXPECTED["pi_L"], "xml": EXPECTED["xml"], "seed_base": SEED_BASE, "seed_namespace": SEED_NAMESPACE}
    manifest = {"status": "FROZEN_BEFORE_OUTCOMES", "inputs": inputs,
        "selection": "8 nearest globally parent-distinct proposals per early/middle/late by fixed Tube-normalized geometry",
        "center": center.tolist(), "scale": scale.tolist(), "funnel": [4, 8, 16, 32],
        "rows": [{k: row[k] for k in ("proposal_id", "candidate_id", "region", "shell_layer", "tube_distance", "physical_state_sha256")} for row in selected]}
    manifest_path = root/"selection_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise SystemExit("partial construction selection mismatch")
    save_json(manifest_path, manifest)
    save_json(root/"cost_estimate.json", {"estimated_seconds": 3600, "states": len(selected),
        "minimum_branches": 8, "maximum_branches": 32, "maximum_rollouts": len(selected)*32,
        "fraction_of_pool": len(selected)/len(pool), "PPO_steps": 0, "audit_labels_used": False})
    variants = []
    for spec in DYNAMICS_VARIANTS:
        cfg = load_config("configs/backward_descent_rsi_pilot_v1.json", {"use_bank_resets": False,
            "expert_chain_termination": False, "domain_randomization": False, "obs_noise_enable": False,
            **{k: v for k, v in spec.items() if k != "id"}})
        env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank(), cert_bank=SnapshotBank.load(C_L))
        adapter = compact_observation_command_adapter(jnp.asarray(artifact["prototypes"]), jnp.asarray(artifact["targets"]),
            jnp.asarray(artifact["normalizer_mean"]), jnp.asarray(artifact["normalizer_std"]),
            float(artifact["radius"]), float(artifact["core_radius"]))
        variants.append((spec["id"], env, make_descent_landing_rollout(env, dparams, lparams, horizon=200,
            residual_ticks=8, descent_action_adapter=adapter)))
    partial = root/"partial.json"; rows = []
    if partial.exists():
        saved = json.loads(partial.read_text())
        if saved.get("inputs") != inputs:
            raise SystemExit("partial construction input mismatch")
        rows = saved["rows"]
    for state_index in range(len(rows), len(selected)):
        proposal = selected[state_index]; record = _load_record(proposal); evidence = []
        target = 4
        while len(evidence) < target:
            requested = list(range(len(evidence), target)); new = []
            for variant_index, (variant_id, env, rollout) in enumerate(variants):
                indices = [b for b in requested if b % len(variants) == variant_index]
                if not indices:
                    continue
                seeds, deltas = [], []
                for b in indices:
                    seed = branch_seed(SEED_BASE, state_index, b); rng = np.random.default_rng(seed)
                    seeds.append(seed); deltas.append(rng.uniform(-.02, .02, size=2).astype(np.float32))
                batch = _perturb_batch(env, record, seeds, deltas); zero = jnp.zeros((len(seeds), 2, 4), jnp.float32)
                raw_out = jax.device_get(rollout(batch, zero, jax.random.PRNGKey(branch_seed(SEED_BASE, state_index, variant_index))))
                for local, b in enumerate(indices):
                    code = int(np.asarray(raw_out["end_code"])[local]); final = bool(np.asarray(raw_out["final_recovery"])[local])
                    chain = bool(np.asarray(raw_out["downstream_entry"])[local]); qualified = bool(chain and final)
                    cause = "final_recovery" if qualified else ("timeout" if code == 8 else ("horizon_exhausted" if code == 0 else "physical_failure"))
                    new.append({"branch_index": b, "branch_seed": seeds[local], "seed_namespace": SEED_NAMESPACE,
                        "dynamics_variant": variant_id, "chain_success": chain, "final_recovery": qualified,
                        "raw_final_recovery": final, "terminal_cause": cause, "end_code": code,
                        "end_reason": END_REASON.get(code, "unknown"), "steps": int(np.asarray(raw_out["termination_tick"])[local])})
            evidence.extend(new); evidence.sort(key=lambda row: row["branch_index"])
            successes = sum(row["final_recovery"] for row in evidence)
            target = next_budget(successes, len(evidence), base)
        finals = sum(row["final_recovery"] for row in evidence); chains = sum(row["chain_success"] for row in evidence)
        rows.append({"state_index": state_index, "id": proposal["proposal_id"], "candidate_id": proposal["candidate_id"],
            "region": proposal["region"], "layer": proposal["shell_layer"], "tube_distance": proposal["tube_distance"],
            "chain": outcome(chains, len(evidence), base), "final": outcome(finals, len(evidence), base), "branches": evidence})
        save_json(partial, {"inputs": inputs, "completed": len(rows), "rows": rows})
        print(f"[C_D construction] {len(rows)}/{len(selected)} {proposal['region']} d={proposal['tube_distance']:.4f} final={finals}/{len(evidence)} {rows[-1]['final']['label']}", flush=True)
    records = []
    by_proposal = {row["proposal_id"]: row for row in selected}
    for result in rows:
        proposal = by_proposal[result["id"]]; item = copy.deepcopy(_load_record(proposal))
        item.update({"id": result["id"], "source_phase": "flight", "origin_phase": "descent",
            "entry_feature": descent_entry_feature(item["physical_feature"], base).astype(np.float32),
            "entry_source_id": result["candidate_id"], "descent_region": result["region"], "descent_layer": result["layer"],
            "tube_distance_at_selection": result["tube_distance"], "chain": result["chain"], "final": result["final"],
            "certification_branches": result["branches"], "policy_version": artifact["policy_identity_hash"],
            "estimator_version": "event_filter_v1", "tube_version": "descent-compact-construction-v1",
            "artifact_role": "construction_only_matcher_calibration", "safe_claim_allowed": False,
            "tube_metrics_eligible": False, "independent_audit_pending": True})
        records.append(item)
    metadata = {"artifact_role": "construction_only_matcher_calibration", "entry_bank_role": "descent_matcher_construction_neighborhood",
        "phase": "descent", "last_tube_version": "descent-compact-construction-v1", "last_policy_version": artifact["policy_identity_hash"],
        "policy_identity_hash": artifact["policy_identity_hash"], "adapter_sha256": file_sha256(EXPERT/"adapter.pkl"),
        "landing_entry_set_sha256": EXPECTED["C_L"], "xml_sha256": EXPECTED["xml"], "action_mapping_version": ACTION_MAPPING_VERSION,
        "construction_seed_namespace": SEED_NAMESPACE, "independent_audit": False, "safe_claim_allowed": False}
    bank_path = root/"construction_bank.pkl"; SnapshotBank(records, metadata).save(bank_path)
    counts = {label: sum(row["final"]["label"] == label for row in rows) for label in ("safe", "boundary", "dead", "unknown")}
    summary = detailed_terminal_summary([branch for row in rows for branch in row["branches"]])
    report = {"status": "PASS" if counts["safe"] >= 4 and (counts["boundary"]+counts["dead"]+counts["unknown"]) > 0 else "FAIL",
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "states": len(rows),
        "counts": counts, "regions": {region: {label: sum(r["region"] == region and r["final"]["label"] == label for r in rows) for label in counts} for region in ("early", "middle", "late")},
        "terminal_summary": summary, "bank_path": str(bank_path), "bank_sha256": file_sha256(bank_path),
        "matcher_activated": False, "PPO_authorization": False,
        "next": "freeze_construction_matcher" if counts["safe"] >= 4 else "construction_support_diagnosis", "rows": rows}
    save_json(root/"DESCENT_COMPACT_MATCHER_CONSTRUCTION_V1_REPORT.json", report)
    save_json(root/"completed.json", {"status": report["status"], "next": report["next"]})
    print(json.dumps({k: report[k] for k in ("status", "states", "counts", "regions", "terminal_summary", "next")}, indent=2))


if __name__ == "__main__":
    main()
