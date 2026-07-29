"""Independent 32-branch Final-safety audit and Descent Tube-v5 extension."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import subprocess
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cli.audit_descent_compact_adapter_v1 import _perturb_batch
from cli.finalize_descent_compact_tube_v2 import certified_outcome
from cli.run_backward_descent_nominal_pilot import C_L, EXPECTED, PI_D, PI_L
from cli.runtime_gate import source_fingerprint
from dvgc.backward_search import compact_observation_command_adapter, make_descent_landing_rollout
from dvgc.bank import SnapshotBank, beta_posterior, posterior_label
from dvgc.certification import DYNAMICS_VARIANTS, branch_seed
from dvgc.config import file_sha256, load_config
from dvgc.descent_entry import descent_entry_feature
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.runtime import save_json


SOURCE = Path("runs/descent_reachability_network_v3/frozen_policy_screen_3x8_20260729/frozen_policy_screen_survivors.pkl")
BASE = Path("runs/descent_natural_bridge_candidates_v1/independent_audit_round2_round3_2x32/descent_tube_v4.pkl")
EXPERT = Path("runs/descent_diverse_p1_predecessor_recovery_v5_p1_core_adapter")
DEFAULT_RUN = Path("runs/descent_reachability_network_v3/independent_tube_extension_3x32_20260729")
BRANCHES = 32
SEED = 4_120_000_000
NAMESPACE = "descent-network-ranked-independent-audit-v1"


def union_records(base: list[dict], extension: list[dict]) -> list[dict]:
    by_id = {row["id"]: copy.deepcopy(row) for row in base}
    for row in extension:
        if row["id"] in by_id:
            raise ValueError(f"Tube extension id collision: {row['id']}")
        by_id[row["id"]] = copy.deepcopy(row)
    return list(by_id.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=str(DEFAULT_RUN))
    parser.add_argument("--source", default=str(SOURCE))
    parser.add_argument("--base", default=str(BASE))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--namespace", default=NAMESPACE)
    args = parser.parse_args()
    root, source_path, base_path = Path(args.run), Path(args.source), Path(args.base)
    if root.exists():
        raise SystemExit(f"refusing overwrite {root}")
    if "independent_audit" in str(source_path).lower():
        raise SystemExit("independent audit labels cannot be an acquisition source")
    gate = json.loads(Path("docs/RUNTIME_GATE.json").read_text())
    if gate.get("status") != "PASS" or gate.get("source_fingerprint") != source_fingerprint(Path.cwd()):
        raise SystemExit("runtime gate stale")
    if file_sha256(C_L) != EXPECTED["C_L"] or file_sha256(PI_D / "params.pkl") != EXPECTED["pi_D"] or file_sha256(PI_L / "params.pkl") != EXPECTED["pi_L"]:
        raise SystemExit("frozen policy/C_L mismatch")
    source, base = SnapshotBank.load(source_path), SnapshotBank.load(base_path)
    artifact = pickle.loads((EXPERT / "adapter.pkl").read_bytes())
    if source.metadata.get("policy_identity_hash") != artifact["policy_identity_hash"] or base.metadata.get("policy_identity_hash") != artifact["policy_identity_hash"]:
        raise SystemExit("candidate/base/adapter policy identity mismatch")
    if {row["id"] for row in source.records} & {row["id"] for row in base.records}:
        raise SystemExit("source overlaps base Tube")

    root.mkdir(parents=True)
    inputs = {
        "source": {"path": str(source_path), "sha256": file_sha256(source_path)},
        "base": {"path": str(base_path), "sha256": file_sha256(base_path)},
        "adapter": {"path": str(EXPERT / "adapter.pkl"), "sha256": file_sha256(EXPERT / "adapter.pkl")},
        "policy_identity_hash": artifact["policy_identity_hash"],
        "C_L": EXPECTED["C_L"], "pi_D": EXPECTED["pi_D"], "pi_L": EXPECTED["pi_L"],
        "xml": EXPECTED["xml"], "seed": int(args.seed), "namespace": args.namespace,
    }
    save_json(root / "manifest.json", {"status": "FROZEN_BEFORE_AUDIT", "inputs": inputs,
        "states": len(source.records), "branches": BRANCHES, "dynamics_variants": DYNAMICS_VARIANTS,
        "label": "legal Final-Recovery posterior; Chain reported separately",
        "labels_reused_for_training": False})
    save_json(root / "cost_estimate.json", {"estimated_seconds": 600,
        "states": len(source.records), "branches_per_state": BRANCHES,
        "rollouts": len(source.records) * BRANCHES, "PPO_steps": 0})

    dparams, _, _ = load_bundle(PI_D, verify_files=True)
    lparams, _, _ = load_bundle(PI_L, verify_files=True)
    cfg0 = load_config("configs/backward_descent_rsi_pilot_v1.json", {
        "use_bank_resets": False, "expert_chain_termination": False,
        "domain_randomization": False, "obs_noise_enable": False})
    variants = []
    for spec in DYNAMICS_VARIANTS:
        cfg = load_config("configs/backward_descent_rsi_pilot_v1.json", {
            "use_bank_resets": False, "expert_chain_termination": False,
            "domain_randomization": False, "obs_noise_enable": False,
            **{key: value for key, value in spec.items() if key != "id"},
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
    for state_index, record in enumerate(source.records):
        evidence = []
        for variant_index, (variant, env, rollout) in enumerate(variants):
            indices = list(range(variant_index, BRANCHES, len(variants)))
            seeds, deltas = [], []
            for branch in indices:
                seed = branch_seed(int(args.seed), state_index, branch)
                seeds.append(seed)
                deltas.append(np.random.default_rng(seed).uniform(-0.02, 0.02, 2).astype(np.float32))
            batch = _perturb_batch(env, record, seeds, deltas)
            raw = jax.device_get(rollout(batch, jnp.zeros((len(seeds), 2, 4), jnp.float32),
                                         jax.random.PRNGKey(branch_seed(int(args.seed), state_index, variant_index))))
            for local, branch in enumerate(indices):
                code = int(np.asarray(raw["end_code"])[local])
                final = bool(np.asarray(raw["final_recovery"])[local])
                chain = bool(np.asarray(raw["downstream_entry"])[local])
                evidence.append({
                    "branch_index": branch, "branch_seed": seeds[local], "seed_namespace": args.namespace,
                    "dynamics_variant": variant, "final_recovery": final,
                    "chain_success": chain, "chain_final": chain and final,
                    "terminal_cause": "final_recovery" if final else (
                        "timeout" if code == 8 else ("horizon_exhausted" if code == 0 else "physical_failure")),
                    "end_code": code, "end_reason": END_REASON.get(code, "unknown"),
                    "steps": int(np.asarray(raw["termination_tick"])[local]),
                })
        evidence.sort(key=lambda row: row["branch_index"])
        finals = sum(row["final_recovery"] for row in evidence)
        chains = sum(row["chain_final"] for row in evidence)
        posterior = beta_posterior(finals, BRANCHES - finals)
        label = posterior_label(posterior, BRANCHES, min_branches=cfg0.min_branches,
                                safe_threshold=cfg0.safe_threshold, dead_threshold=cfg0.dead_threshold,
                                boundary_max_width=cfg0.boundary_max_width)
        rows.append({"id": record["id"], "candidate_id": record.get("candidate_id"),
                     "final": finals, "chain_final": chains, "branches": BRANCHES,
                     "posterior": posterior, "label": label, "evidence": evidence})
        save_json(root / "audit.partial.json", {"inputs": inputs, "rows": rows})
        print(f"[independent-tube-audit] {state_index + 1}/{len(source.records)} Final={finals}/32 Chain+Final={chains}/32 {label}", flush=True)

    safe_ids = {row["id"] for row in rows if row["label"] == "safe"}
    by_result = {row["id"]: row for row in rows}
    new = []
    version = "descent-network-" + hashlib.sha256(
        (file_sha256(base_path) + file_sha256(source_path) + args.namespace).encode()).hexdigest()[:12]
    for record in source.records:
        if record["id"] not in safe_ids:
            continue
        audit = by_result[record["id"]]
        item = copy.deepcopy(record)
        item.update({
            "source_phase": "flight", "origin_phase": "descent",
            "entry_feature": descent_entry_feature(item["physical_feature"], cfg0).astype(np.float32),
            "chain": certified_outcome(audit["chain_final"], BRANCHES, cfg0),
            "final": certified_outcome(audit["final"], BRANCHES, cfg0),
            "policy_version": artifact["policy_identity_hash"],
            "estimator_version": "descent_reachability_network_v3",
            "tube_version": version, "certification_branches": audit["evidence"],
            "artifact_role": "certified_tube", "certified_safe": True,
            "tube_metrics_eligible": True, "safe_claim_allowed": True,
            "independent_audit": True,
        })
        new.append(item)
    combined = union_records(base.records, new)
    tube_path = root / "descent_tube_v5.pkl"
    if new:
        metadata = copy.deepcopy(base.metadata)
        metadata.update({
            "last_tube_version": version, "last_policy_version": artifact["policy_identity_hash"],
            "supersedes": str(base_path.resolve()), "extension_audit_namespace": args.namespace,
            "extension_source_sha256": file_sha256(source_path),
            "branches_per_extension_state": BRANCHES, "independent_audit": True,
            "expert_conditioned": True, "formal_jel_eligible": False,
            "safety_label_semantics": "Final-Recovery; Chain separate",
        })
        SnapshotBank(combined, metadata).save(tube_path)
    causes = Counter(event["end_reason"] for row in rows for event in row["evidence"] if not event["final_recovery"])
    status = "PASS" if new else "FAIL"
    report = {
        "status": status, "artifact_role": "independent_descent_tube_extension_audit",
        "states": len(rows), "safe_states": len(new),
        "labels": dict(Counter(row["label"] for row in rows)),
        "final_successes": sum(row["final"] for row in rows),
        "chain_final_successes": sum(row["chain_final"] for row in rows),
        "failure_reasons": dict(causes), "base_states": len(base.records),
        "combined_states": len(combined) if new else len(base.records),
        "tube_version": version if new else None,
        "tube_path": str(tube_path) if new else None,
        "tube_sha256": file_sha256(tube_path) if new else None,
        "rows": rows, "PPO_authorization": False,
        "next": "tube_rsi_retention_pilot" if new else "candidate_audit_support_exhausted",
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    save_json(root / "DESCENT_NETWORK_TUBE_EXTENSION_V1_REPORT.json", report)
    save_json(root / "completed.json", {"status": status, "next": report["next"]})
    print(json.dumps({key: report[key] for key in (
        "status", "states", "safe_states", "labels", "final_successes", "chain_final_successes",
        "failure_reasons", "base_states", "combined_states", "tube_version", "tube_sha256",
        "PPO_authorization", "next")}, indent=2))


if __name__ == "__main__":
    main()
