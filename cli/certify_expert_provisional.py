"""Four-branch pilot for an expert-conditioned provisional Flight envelope."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import jax

from dvgc.bank import SnapshotBank, beta_posterior, posterior_label
from dvgc.certification import DYNAMICS_VARIANTS, branch_evidence, branch_seed, detailed_terminal_summary
from dvgc.composite import CanonicalEntryMatcher, composite_rollout
from dvgc.config import file_sha256, load_config
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.experts import StageExpertRegistry
from dvgc.policy import load_bundle
from dvgc.rollout import restore_snapshot_mode
from dvgc.snapshot_timing import authority_replay_mode
from dvgc.runtime import build_inference, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--candidate-bank", required=True)
    parser.add_argument("--entry-set", required=True)
    parser.add_argument("--output-bank", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--runtime-gate", default="docs/RUNTIME_GATE.json")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--branches", type=int, default=4)
    args = parser.parse_args()
    output = Path(args.output_bank)
    report_path = Path(args.report)
    if output.exists() or report_path.exists():
        raise SystemExit("Provisional pilot output already exists")
    if not 1 <= args.branches <= 32:
        raise SystemExit("Expert provisional certification is capped at 32 branches")

    gate = json.loads(Path(args.runtime_gate).read_text())
    registry = StageExpertRegistry.load(args.registry)
    if gate.get("status") != "PASS" or gate.get("source_fingerprint") != registry.runtime_source_fingerprint:
        raise SystemExit("Expert registry runtime provenance is stale")
    flight = registry.specs["flight"]
    landing = registry.specs["landing"]
    if file_sha256(args.entry_set) != flight.downstream_entry_set_sha256:
        raise SystemExit("Frozen canonical C_L hash mismatch")
    flight_params, flight_cfg, flight_manifest = load_bundle(flight.checkpoint_path, verify_files=True)
    landing_params, _, landing_manifest = load_bundle(landing.checkpoint_path, verify_files=True)
    source = SnapshotBank.load(args.candidate_bank)
    rows = source.records_for_phase("flight", include_training_only=False)

    variants = []
    base_cfg = load_config(overrides={
        **flight_cfg, "training_stage": "flight", "expert_chain_termination": False,
        "domain_randomization": False, "obs_noise_enable": False, "use_bank_resets": False,
    })
    for spec in DYNAMICS_VARIANTS:
        cfg = load_config(overrides={
            **base_cfg.to_dict(),
            **{key: value for key, value in spec.items() if key != "id"},
        })
        env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank(), cert_bank=SnapshotBank.load(args.entry_set))
        inference = {
            "flight": build_inference(env, flight_params, deterministic=True),
            "landing": build_inference(env, landing_params, deterministic=True),
        }
        variants.append((spec["id"], env, jax.jit(env.step), inference, CanonicalEntryMatcher(env, "flight", args.entry_set)))

    result_records = copy.deepcopy(source.records)
    by_id = {row["id"]: row for row in result_records}
    all_evidence = []
    rows_report = []
    for index, row in enumerate(rows):
        evidence = []
        for branch in range(args.branches):
            variant, env, step, inference, matcher = variants[branch % len(variants)]
            seed = branch_seed(args.seed, index, branch)
            key = jax.random.PRNGKey(seed)
            _, outcome = composite_rollout(
                env, ("flight", "landing"), inference, {"flight": matcher},
                restore_snapshot_mode(env, row, key, observation_mode=authority_replay_mode(row)), key, horizon=int(env._config.branch_horizon),
                step_fn=step, action_noise_std=float(env._config.action_noise_std),
            )
            item = branch_evidence(
                branch_index=branch, seed=seed,
                seed_namespace="expert_provisional_flight_pilot",
                dynamics_variant=variant, outcome=outcome,
            )
            item.update({
                "end_code": int(outcome["end_code"]),
                "end_reason": END_REASON.get(int(outcome["end_code"]), "unknown"),
                "chain_missed_final": bool(outcome["chain_missed_final"]),
                "controller_stack_hash": flight.controller_stack_hash,
            })
            evidence.append(item)
            all_evidence.append(item)
        chain = sum(item["chain_success"] for item in evidence)
        final = sum(item["final_recovery"] for item in evidence)
        chain_posterior = beta_posterior(chain, args.branches - chain)
        final_posterior = beta_posterior(final, args.branches - final)
        def label(value):
            return posterior_label(
                value, args.branches, min_branches=int(base_cfg.min_branches),
                safe_threshold=float(base_cfg.safe_threshold),
                dead_threshold=float(base_cfg.dead_threshold),
                boundary_max_width=float(base_cfg.boundary_max_width),
            )
        provisional = {
            "artifact_role": "expert_conditioned_provisional_envelope",
            "controller_mode": "expert_stack", "formal_jel_eligible": False,
            "branches": args.branches, "chain_successes": chain,
            "final_successes": final,
            "chain_posterior": chain_posterior, "chain_label": label(chain_posterior),
            "final_posterior": final_posterior, "final_label": label(final_posterior),
            "branch_evidence": evidence,
        }
        by_id[row["id"]]["expert_conditioned"] = provisional
        rows_report.append({
            "candidate_id": row["id"], "candidate_kind": row.get("candidate_kind"),
            "flight_subinterval": row.get("flight_subinterval"),
            "chain_successes": chain, "final_successes": final,
            "branches": args.branches, "chain_label": provisional["chain_label"],
            "final_label": provisional["final_label"],
        })
        print(f"[expert-provisional-pilot] {index + 1}/{len(rows)} chain={chain}/{args.branches} final={final}/{args.branches}", flush=True)

    metadata = copy.deepcopy(source.metadata)
    metadata.update({
        "artifact_role": "expert_conditioned_provisional_envelope",
        "controller_mode": "expert_stack", "formal_jel_eligible": False,
        "independent_recertification": False, "pilot_only": args.branches < int(base_cfg.min_branches),
        "registry": str(Path(args.registry).resolve()), "registry_hash": registry.registry_hash,
        "controller_stack_hash": flight.controller_stack_hash,
        "flight_policy_version": flight_manifest["policy_version"],
        "flight_policy_hash": flight.policy_hash,
        "landing_policy_version": landing_manifest["policy_version"],
        "landing_policy_hash": landing.policy_hash,
        "canonical_c_l": str(Path(args.entry_set).resolve()),
        "canonical_c_l_sha256": file_sha256(args.entry_set),
        "candidate_bank_sha256": file_sha256(args.candidate_bank),
        "runtime_source_fingerprint": registry.runtime_source_fingerprint,
        "seed": args.seed, "branches_per_state": args.branches,
    })
    SnapshotBank(result_records, metadata).save(output)
    summary = detailed_terminal_summary(all_evidence)
    label_counts = {
        label: sum(row["final_label"] == label for row in rows_report)
        for label in ("safe", "boundary", "dead", "unknown")
    }
    save_json(report_path, {
        "status": "PASS", "artifact_role": "expert_conditioned_provisional_envelope",
        "formal_jel_eligible": False, "states": len(rows),
        "branches_per_state": args.branches, "terminal_summary": summary,
        "states_with_chain": sum(row["chain_successes"] > 0 for row in rows_report),
        "states_with_final": sum(row["final_successes"] > 0 for row in rows_report),
        "final_label_counts": label_counts,
        "registry_hash": registry.registry_hash,
        "controller_stack_hash": flight.controller_stack_hash,
        "canonical_c_l_sha256": file_sha256(args.entry_set),
        "candidate_bank_sha256": file_sha256(args.candidate_bank),
        "output_bank": str(output.resolve()), "output_bank_sha256": file_sha256(output),
        "rows": rows_report,
    })


if __name__ == "__main__":
    main()
