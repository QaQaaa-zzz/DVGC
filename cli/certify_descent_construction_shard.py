"""Certify one resumable construction shard of C_D candidates.

Unlike ``certify_descent_entries`` this command never writes a partially
certified bank.  It writes immutable, globally indexed branch evidence that is
validated and assembled only after every construction shard is complete.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax

from cli.certify_descent_entries import (
    current_label,
    label_decided,
    protocol,
    qualified_descent_success,
)
from dvgc.bank import SnapshotBank
from dvgc.certification import DYNAMICS_VARIANTS, branch_evidence, branch_seed, summarize_branches
from dvgc.composite import CanonicalEntryMatcher, composite_rollout
from dvgc.config import config_hash, file_sha256, load_config
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.rollout import restore_snapshot_mode
from dvgc.snapshot_timing import authority_replay_mode
from dvgc.runtime import build_inference, save_json


def _terminal_summary(evidence):
    summary = summarize_branches(evidence)
    reasons = {name: 0 for name in END_REASON.values()}
    for row in evidence:
        reasons[str(row["end_reason"])] = reasons.get(str(row["end_reason"]), 0) + 1
    summary.update(
        {
            "end_reasons": {key: value for key, value in reasons.items() if value},
            "nonfinite": reasons.get("nonfinite", 0),
            "pitch_failures": reasons.get("pitch_limit", 0),
            "roll_failures": reasons.get("roll_limit", 0),
        }
    )
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--descent-policy", required=True)
    p.add_argument("--candidate-source-policy", required=True)
    p.add_argument("--landing-policy", required=True)
    p.add_argument("--candidate-bank", required=True)
    p.add_argument("--landing-entry-set", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--runtime-gate", default="docs/RUNTIME_GATE.json")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--namespace", required=True)
    p.add_argument("--start-index", type=int, required=True)
    p.add_argument("--end-index", type=int, required=True)
    p.add_argument("--confirm-safe-to-max", action="store_true")
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise SystemExit(f"Output exists: {out}")

    dp, dc, dm = load_bundle(a.descent_policy, verify_files=True)
    lp, _, lm = load_bundle(a.landing_policy, verify_files=True)
    source = SnapshotBank.load(a.candidate_bank)
    source_policy_hash = source.metadata.get("descent_policy_hash")
    current_policy_hash = file_sha256(Path(a.descent_policy) / "params.pkl")
    if source_policy_hash != file_sha256(Path(a.candidate_source_policy) / "params.pkl"):
        raise SystemExit("C_D proposal source-policy provenance mismatch")
    entry_hash = file_sha256(a.landing_entry_set)
    if source.metadata.get("landing_entry_set_sha256") != entry_hash:
        raise SystemExit("C_D proposal C_L provenance mismatch")

    base = load_config(
        a.config,
        {
            **dc,
            "training_stage": "flight",
            "expert_chain_termination": False,
            "domain_randomization": False,
            "obs_noise_enable": False,
            "use_bank_resets": False,
        },
    )
    runtime_gate = json.loads(Path(a.runtime_gate).read_text(encoding="utf-8"))
    if runtime_gate.get("status") != "PASS":
        raise SystemExit("Runtime gate is not PASS")
    rows = source.records_for_phase("flight", include_training_only=False)
    if not (0 <= a.start_index < a.end_index <= len(rows)):
        raise SystemExit(f"Invalid shard [{a.start_index},{a.end_index})/{len(rows)}")

    variants = []
    entry = SnapshotBank.load(a.landing_entry_set)
    for spec in DYNAMICS_VARIANTS:
        cfg = load_config(a.config, {**base.to_dict(), **{k: v for k, v in spec.items() if k != "id"}})
        env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank(), cert_bank=entry)
        inference = {
            "flight": build_inference(env, dp, deterministic=True),
            "landing": build_inference(env, lp, deterministic=True),
        }
        variants.append(
            (spec["id"], env, jax.jit(env.step), inference, CanonicalEntryMatcher(env, "flight", a.landing_entry_set))
        )

    results = []
    all_evidence = []
    namespace = f"{a.namespace}:descent_entry"
    for local_index, i in enumerate(range(a.start_index, a.end_index)):
        row = rows[i]
        successes = failures = chain_successes = chain_failures = raw_final_successes = 0
        evidence = []
        for b in range(int(base.max_branches)):
            variant, env, step, inference, matcher = variants[b % len(variants)]
            seed = branch_seed(a.seed, i, b)
            key = jax.random.PRNGKey(seed)
            _, outcome = composite_rollout(
                env,
                ("flight", "landing"),
                inference,
                {"flight": matcher},
                restore_snapshot_mode(env, row, key, observation_mode=authority_replay_mode(row)),
                key,
                horizon=int(base.branch_horizon),
                step_fn=step,
                action_noise_std=float(base.action_noise_std),
            )
            ev = branch_evidence(
                branch_index=b,
                seed=seed,
                seed_namespace=namespace,
                dynamics_variant=variant,
                outcome=outcome,
            )
            qualified = qualified_descent_success(outcome)
            ev.update(
                {
                    "end_code": int(outcome["end_code"]),
                    "end_reason": END_REASON.get(int(outcome["end_code"]), "unknown"),
                    "raw_composite_final_recovery": bool(outcome["final"]),
                    "descent_entry_final_success": bool(qualified),
                }
            )
            if outcome["final"] and not outcome["chain"]:
                ev["final_recovery"] = False
                ev["terminal_cause"] = "handoff_missed_final"
            evidence.append(ev)
            all_evidence.append(ev)
            chain_successes += int(outcome["chain"])
            chain_failures += int(not outcome["chain"])
            raw_final_successes += int(outcome["final"])
            successes += int(qualified)
            failures += int(not qualified)
            branches = b + 1
            if branches >= int(base.min_branches) and label_decided(successes, failures, base) and label_decided(chain_successes, chain_failures, base):
                provisional_safe = current_label(successes, failures, base) == "safe" or current_label(chain_successes, chain_failures, base) == "safe"
                if not (a.confirm_safe_to_max and provisional_safe and branches < int(base.max_branches)):
                    break
        results.append(
            {
                "id": row["id"],
                "candidate_index": i,
                "source_id": row.get("entry_source_id"),
                "parent_candidate_id": row.get("parent_candidate_id"),
                "proposal_step": row.get("proposal_step"),
                "descent_layer": row.get("descent_layer"),
                "bootstrap_group": row.get("bootstrap_group"),
                "chain": chain_successes,
                "raw_final": raw_final_successes,
                "final": successes,
                "branches": successes + failures,
                "final_rate": successes / (successes + failures),
                "branch_evidence": evidence,
            }
        )
        print(
            f"[C_D construction shard] {local_index + 1}/{a.end_index-a.start_index} "
            f"global={i} chain={chain_successes}/{chain_successes+chain_failures} "
            f"final={successes}/{successes+failures}",
            flush=True,
        )

    common = {
        "status": "PASS",
        "artifact_role": "descent_entry_construction_shard",
        "complete": True,
        "confirm_safe_to_max": bool(a.confirm_safe_to_max),
        "seed": int(a.seed),
        "seed_namespace": namespace,
        "candidate_bank_sha256": file_sha256(a.candidate_bank),
        "candidate_source_policy_hash": source_policy_hash,
        "landing_entry_set_sha256": entry_hash,
        "descent_policy_hash": current_policy_hash,
        "descent_policy_version": dm["policy_version"],
        "descent_estimator_version": dm.get("estimator_version", "event_filter_v1"),
        "landing_policy_hash": file_sha256(Path(a.landing_policy) / "params.pkl"),
        "landing_policy_version": lm["policy_version"],
        "xml_sha256": file_sha256(base.xml_path),
        "config_hash": config_hash(base),
        "runtime_source_fingerprint": runtime_gate.get("source_fingerprint"),
        "protocol": protocol(base),
        "min_branches": int(base.min_branches),
        "max_branches": int(base.max_branches),
        "branch_horizon": int(base.branch_horizon),
        "states": len(results),
        "total_states": len(rows),
        "start_index": int(a.start_index),
        "end_index": int(a.end_index),
        "terminal_summary": _terminal_summary(all_evidence),
        "rows": results,
    }
    save_json(out, common)
    print(json.dumps({k: v for k, v in common.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
