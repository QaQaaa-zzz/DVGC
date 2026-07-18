"""Run one immutable Stage-A, Stage-B, or adaptive stable-construction shard."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax

from cli.certify_descent_entries import qualified_descent_success
from dvgc.bank import SnapshotBank
from dvgc.certification import DYNAMICS_VARIANTS, branch_evidence, branch_seed, detailed_terminal_summary
from dvgc.composite import CanonicalEntryMatcher, composite_rollout
from dvgc.config import config_hash, file_sha256, load_config
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.rollout import restore_snapshot
from dvgc.runtime import build_inference, save_json
from dvgc.snapshot_provenance import validate_snapshot_source_records,verify_source_policy_paths
from dvgc.stable_construction import protocol_from_config


def indices_hash(indices) -> str:
    return hashlib.sha256(json.dumps([int(value) for value in indices], separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage_a", "stage_b", "adaptive"), required=True)
    parser.add_argument("--descent-policy", required=True)
    parser.add_argument("--candidate-source-policy", action="append", required=True)
    parser.add_argument("--landing-policy", required=True)
    parser.add_argument("--candidate-bank", required=True)
    parser.add_argument("--landing-entry-set", required=True)
    parser.add_argument("--indices-file", required=True)
    parser.add_argument("--start-index", type=int, required=True, help="start offset in indices-file")
    parser.add_argument("--end-index", type=int, required=True, help="end offset in indices-file")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--runtime-gate", default="docs/RUNTIME_GATE.json")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"Output exists: {output}")

    selected = [int(value) for value in json.loads(Path(args.indices_file).read_text())]
    if len(selected) != len(set(selected)) or selected != sorted(selected):
        raise SystemExit("Stable construction indices must be sorted and unique")
    if not (0 <= args.start_index < args.end_index <= len(selected)):
        raise SystemExit("Invalid stable construction selection slice")

    descent_params, descent_config, descent_manifest = load_bundle(args.descent_policy, verify_files=True)
    landing_params, _, landing_manifest = load_bundle(args.landing_policy, verify_files=True)
    source = SnapshotBank.load(args.candidate_bank)
    rows = source.records_for_phase("flight", include_training_only=False)
    if selected and (selected[0] < 0 or selected[-1] >= len(rows)):
        raise SystemExit("Stable construction candidate index out of range")
    try:
        source_policy_hashes=validate_snapshot_source_records(rows,source.metadata)
        verify_source_policy_paths(source_policy_hashes,
            [str(Path(path)/"params.pkl") for path in args.candidate_source_policy],file_sha256)
    except ValueError as exc:raise SystemExit(str(exc)) from exc
    entry_hash = file_sha256(args.landing_entry_set)
    if source.metadata.get("landing_entry_set_sha256") != entry_hash:
        raise SystemExit("Stable construction C_L mismatch")

    cfg = load_config(args.config, {
        **descent_config, "training_stage": "flight", "expert_chain_termination": False,
        "domain_randomization": False, "obs_noise_enable": False, "use_bank_resets": False,
    })
    branch_budget = (int(cfg.stable_construction_adaptive_max_branches)
                     if args.stage == "adaptive" else int(cfg.stable_construction_stage_branches))
    gate = json.loads(Path(args.runtime_gate).read_text())
    if gate.get("status") != "PASS":
        raise SystemExit("Runtime gate is not PASS")

    entry = SnapshotBank.load(args.landing_entry_set)
    variants = []
    for spec in DYNAMICS_VARIANTS:
        variant_cfg = load_config(args.config, {**cfg.to_dict(), **{k: v for k, v in spec.items() if k != "id"}})
        env = OrangeBikeDVGC(variant_cfg, snapshot_bank=SnapshotBank(), cert_bank=entry)
        inference = {
            "flight": build_inference(env, descent_params, deterministic=True),
            "landing": build_inference(env, landing_params, deterministic=True),
        }
        variants.append((spec["id"], env, jax.jit(env.step), inference,
                         CanonicalEntryMatcher(env, "flight", args.landing_entry_set)))

    results, all_evidence = [], []
    namespace = f"{args.namespace}:{args.stage}:descent_entry"
    for offset in range(args.start_index, args.end_index):
        global_index = selected[offset]
        record = rows[global_index]
        evidence = []
        for branch_index in range(branch_budget):
            variant, env, step, inference, matcher = variants[branch_index % len(variants)]
            seed = branch_seed(args.seed, global_index, branch_index)
            key = jax.random.PRNGKey(seed)
            _, result = composite_rollout(
                env, ("flight", "landing"), inference, {"flight": matcher},
                restore_snapshot(env, record, key), key, horizon=int(cfg.branch_horizon),
                step_fn=step, action_noise_std=float(cfg.action_noise_std),
            )
            item = branch_evidence(
                branch_index=branch_index, seed=seed, seed_namespace=namespace,
                dynamics_variant=variant, outcome=result,
            )
            qualified = qualified_descent_success(result)
            item.update({
                "end_code": int(result["end_code"]),
                "end_reason": END_REASON.get(int(result["end_code"]), "unknown"),
                "raw_composite_final_recovery": bool(result["final"]),
                "descent_entry_final_success": bool(qualified),
            })
            if result["final"] and not result["chain"]:
                item["final_recovery"] = False
                item["terminal_cause"] = "handoff_missed_final"
            evidence.append(item)
            all_evidence.append(item)
        results.append({
            "id": record["id"], "candidate_index": global_index,
            "source_id": record.get("entry_source_id"),
            "parent_candidate_id": record.get("parent_candidate_id"),
            "descent_layer": record.get("descent_layer"),
            "chain": sum(bool(item["chain_success"]) for item in evidence),
            "final": sum(bool(item["final_recovery"]) for item in evidence),
            "branches": len(evidence), "branch_evidence": evidence,
        })
        print(f"[stable {args.stage}] {offset-args.start_index+1}/{args.end_index-args.start_index} global={global_index}", flush=True)

    payload = {
        "status": "PASS", "complete": True, "artifact_role": "stable_construction_shard",
        "stage": args.stage, "seed": int(args.seed), "seed_namespace": namespace,
        "candidate_bank_sha256": file_sha256(args.candidate_bank),
        "candidate_source_policy_hash": source_policy_hashes[0] if len(source_policy_hashes)==1 else None,
        "candidate_source_policy_hashes": list(source_policy_hashes),
        "descent_policy_hash": file_sha256(Path(args.descent_policy) / "params.pkl"),
        "descent_policy_version": descent_manifest["policy_version"],
        "landing_policy_hash": file_sha256(Path(args.landing_policy) / "params.pkl"),
        "landing_policy_version": landing_manifest["policy_version"],
        "landing_entry_set_sha256": entry_hash, "xml_sha256": file_sha256(cfg.xml_path),
        "config_hash": config_hash(cfg), "runtime_source_fingerprint": gate.get("source_fingerprint"),
        "protocol": protocol_from_config(cfg), "branch_horizon": int(cfg.branch_horizon),
        "branches_per_state": branch_budget, "total_states": len(rows),
        "selected_states": len(selected), "selected_indices_sha256": indices_hash(selected),
        "selection_start": int(args.start_index), "selection_end": int(args.end_index),
        "candidate_indices": selected[args.start_index:args.end_index],
        "terminal_summary": detailed_terminal_summary(all_evidence), "rows": results,
    }
    save_json(output, payload)
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
