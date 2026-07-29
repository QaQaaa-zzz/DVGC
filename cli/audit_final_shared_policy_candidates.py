"""Evaluate phase-balanced candidates by end-to-end Final-Recovery under one frozen actor."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from dvgc.config import file_sha256
from dvgc.runtime import save_json


STAGES = ("takeoff", "ascent", "apex", "descent", "landing")


def exact_final_label(branches: list[dict]) -> dict:
    successes = sum(row.get("final_recovery") is True for row in branches)
    total = len(branches)
    return {
        "s": successes, "n": total,
        "label": "positive" if total > 0 and successes == total else
                 ("negative" if successes == 0 else "boundary"),
        "branches": branches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--candidate-bank", required=True)
    parser.add_argument("--canonical-entry-bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--branches", type=int, choices=(4, 8, 32), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--horizon", type=int, default=400)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit("refusing to overwrite final shared-policy branch audit")

    import jax
    from cli.runtime_gate import source_fingerprint
    from dvgc.bank import SnapshotBank
    from dvgc.certification import DYNAMICS_VARIANTS, branch_seed
    from dvgc.config import load_config
    from dvgc.env import END_REASON, OrangeBikeDVGC
    from dvgc.policy import load_bundle
    from dvgc.rollout import frozen_rollout, restore_snapshot_mode
    from dvgc.runtime import build_inference
    from dvgc.snapshot_timing import authority_replay_mode

    gate = json.loads(Path("docs/RUNTIME_GATE.json").read_text())
    if gate.get("status") != "PASS" or gate.get("source_fingerprint") != source_fingerprint(Path.cwd()):
        raise SystemExit("runtime gate stale")
    params, policy_cfg, manifest = load_bundle(args.policy, verify_files=True)
    if manifest.get("artifact_role") != "unified_rsi_pilot_checkpoint" or manifest.get("promoted") is not True:
        raise SystemExit("only a promoted frozen unified RSI policy may enter Final certification")
    candidates = SnapshotBank.load(args.candidate_bank); entry = SnapshotBank.load(args.canonical_entry_bank)
    role = candidates.metadata.get("artifact_role")
    if role not in {"phase_balanced_tube_rsi_reset_bank", "independent_audit_candidate_bank",
                    "final_shared_policy_audit_candidate_bank"}:
        raise SystemExit(f"invalid final audit candidate role {role!r}")
    policy_candidate_hash = manifest.get("candidate_bank_sha256")
    input_hash = file_sha256(args.candidate_bank)
    root_hash = candidates.metadata.get("root_source_bank_sha256", input_hash)
    if policy_candidate_hash != root_hash:
        raise SystemExit("frozen policy candidate bank does not match audit input")
    if manifest.get("downstream_bank_sha256") != file_sha256(args.canonical_entry_bank):
        raise SystemExit("frozen policy canonical entry bank does not match audit input")
    if not candidates.records or any(row.get("phase_rsi_stage") not in STAGES for row in candidates.records):
        raise SystemExit("candidate bank lacks explicit five-stage identities")

    runtimes = []
    for variant in DYNAMICS_VARIANTS:
        overrides = {key: value for key, value in variant.items() if key != "id"}
        cfg = load_config(overrides={
            **policy_cfg, **overrides, "training_stage": "flight", "use_bank_resets": False,
            "domain_randomization": False, "obs_noise_enable": False,
            "expert_chain_termination": False, "stage_reachability_objective": "",
        })
        env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank(), cert_bank=entry)
        runtimes.append((variant["id"], env, jax.jit(env.step)))
    inference = build_inference(runtimes[0][1], params, deterministic=True)
    labels = []; all_seeds = []; causes = Counter(); phase_summary = {}
    for state_index, record in enumerate(candidates.records):
        branch_rows = []
        for branch_index in range(args.branches):
            variant, env, step = runtimes[branch_index % len(runtimes)]
            seed = branch_seed(args.seed, state_index, branch_index); all_seeds.append(seed)
            key = jax.random.PRNGKey(seed)
            state = restore_snapshot_mode(
                env, record, key, observation_mode=authority_replay_mode(record)
            )
            _, outcome = frozen_rollout(
                env, inference, state, key, horizon=args.horizon,
                action_noise_std=float(env._config.action_noise_std), step_fn=step,
            )
            reason = END_REASON.get(outcome["end_code"], "unknown")
            final = bool(outcome["final"]); chain = bool(outcome["chain"])
            if final:
                cause = "final_recovery"
            elif outcome["terminated"]:
                cause = "physical_failure"
            elif outcome["truncated"]:
                cause = "timeout"
            else:
                cause = "horizon_exhausted"
            causes[cause] += 1
            branch_rows.append({
                "branch_index": branch_index, "seed": seed, "seed_namespace": args.namespace,
                "dynamics_variant": variant, "success": final, "final_recovery": final,
                "chain_ever": chain, "terminated": bool(outcome["terminated"]),
                "truncated": bool(outcome["truncated"]), "steps": int(outcome["steps"]),
                "terminal_cause": cause, "termination_reason": reason,
            })
        label = exact_final_label(branch_rows)
        label.update({
            "candidate_id": record["id"], "origin_record_id": record.get("origin_record_id"),
            "phase": record["phase_rsi_stage"], "state_index": state_index,
        })
        labels.append(label)
    if len(all_seeds) != len(set(all_seeds)):
        raise SystemExit("final shared-policy audit generated duplicate branch seeds")
    for stage in STAGES:
        subset = [row for row in labels if row["phase"] == stage]
        phase_summary[stage] = {
            "states": len(subset), "exact_final_safe": sum(row["s"] == row["n"] for row in subset),
            "final_branches": sum(row["s"] for row in subset),
            "total_branches": sum(row["n"] for row in subset),
        }
    report = {
        "status": "PASS", "artifact_role": "final_shared_policy_branch_audit",
        "formal_tube_or_jel": False, "policy_path": str(Path(args.policy)),
        "policy_params_sha256": file_sha256(Path(args.policy) / "params.pkl"),
        "xml_sha256": manifest["xml_sha256"],
        "action_mapping_version": manifest["action_mapping_version"],
        "candidate_bank": str(Path(args.candidate_bank)),
        "candidate_bank_sha256": input_hash,
        "root_candidate_bank_sha256": root_hash,
        "root_candidate_state_count": int(candidates.metadata.get(
            "root_source_state_count", len(candidates.records)
        )),
        "root_phase_state_counts": candidates.metadata.get("root_phase_state_counts", {
            stage: sum(row.get("phase_rsi_stage") == stage for row in candidates.records)
            for stage in STAGES
        }),
        "canonical_entry_bank_sha256": file_sha256(args.canonical_entry_bank),
        "branches_per_state": args.branches, "seed_base": args.seed,
        "seed_namespace": args.namespace, "horizon": args.horizon,
        "states": len(labels), "exact_final_safe": sum(row["s"] == row["n"] for row in labels),
        "final_branches": sum(row["s"] for row in labels),
        "total_branches": sum(row["n"] for row in labels),
        "terminal_causes": dict(causes), "phase_summary": phase_summary, "labels": labels,
    }
    save_json(output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "labels"}, indent=2))


if __name__ == "__main__":
    main()
