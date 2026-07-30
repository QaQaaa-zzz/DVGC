"""Read-only fixed evaluation of per-reset next-stage objectives."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from dvgc.config import file_sha256
from dvgc.runtime import save_json


STAGES = ("takeoff", "ascent", "apex", "descent")


def fixed_parent_diverse(records: list[dict], per_stage: int) -> list[dict]:
    selected: list[dict] = []
    for stage in STAGES:
        rows = sorted(
            (row for row in records if row.get("phase_rsi_stage") == stage),
            key=lambda row: str(row["id"]),
        )
        seen: set[str] = set()
        for row in rows:
            parent = str(row.get("reset_parent_id") or row["id"])
            if parent in seen:
                continue
            seen.add(parent); selected.append(row)
            if len(seen) == per_stage:
                break
        if len(seen) < min(per_stage, len(rows)):
            raise ValueError(f"insufficient parent-diverse {stage} states")
    return selected


def evaluate_policy(env, params, records: list[dict], seed: int) -> dict:
    import jax
    import jax.numpy as jnp

    from dvgc.env import END_REASON, PHASE_RSI_OBJECTIVE
    from dvgc.rollout import frozen_rollout, restore_snapshot
    from dvgc.runtime import build_inference

    inference = build_inference(env, params, deterministic=True)
    step = jax.jit(env.step)
    rows = []
    for index, record in enumerate(records):
        stage = str(record["phase_rsi_stage"])
        key = jax.random.PRNGKey(seed + index)
        state = restore_snapshot(env, record, key)
        # The objective is reset-sampler metadata, not serialized physical
        # state.  Explicitly install it and clear any historical proposal
        # latch so this read-only replay measures a fresh local handoff.
        info = dict(state.info)
        info["reachability_objective_id"] = jnp.asarray(
            PHASE_RSI_OBJECTIVE[stage], jnp.int32
        )
        info["stage_entry_ever"] = jnp.zeros((), jnp.int32)
        info["apex_descent_stable_count"] = jnp.zeros((), jnp.int32)
        state = state.replace(info=info)
        final_state, outcome = frozen_rollout(
            env, inference, state, key, horizon=200, step_fn=step
        )
        entry = bool(np.asarray(final_state.info["stage_entry_ever"]))
        rows.append({
            "stage": stage, "record_id": record["id"], "entry": entry,
            "steps": outcome["steps"],
            "physical_failure": bool(outcome["terminated"] and not entry),
            "timeout": bool(outcome["truncated"]),
            "termination_reason": END_REASON.get(outcome["end_code"], "unknown"),
        })
    by_stage = {}
    for stage in STAGES:
        values = [row for row in rows if row["stage"] == stage]
        by_stage[stage] = {
            "states": len(values), "entries": sum(row["entry"] for row in values),
            "physical_failures": sum(row["physical_failure"] for row in values),
            "timeouts": sum(row["timeout"] for row in values),
            "termination_reasons": dict(Counter(
                row["termination_reason"] for row in values
            )),
        }
    return {
        "states": len(rows), "entries": sum(row["entry"] for row in rows),
        "by_stage": by_stage, "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-bank", required=True)
    parser.add_argument("--canonical-entry-bank", required=True)
    parser.add_argument("--descent-entry-support-bank", required=True)
    parser.add_argument("--baseline-policy", required=True)
    parser.add_argument("--final-policy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--states-per-stage", type=int, default=3)
    parser.add_argument("--seed", type=int, default=10_950_000)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")

    from dvgc.bank import SnapshotBank
    from dvgc.config import load_config
    from dvgc.env import OrangeBikeDVGC
    from dvgc.policy import load_bundle

    bank = SnapshotBank.load(args.phase_bank)
    entry = SnapshotBank.load(args.canonical_entry_bank)
    support = SnapshotBank.load(args.descent_entry_support_bank)
    baseline, policy_cfg, _ = load_bundle(args.baseline_policy, verify_files=True)
    final, _, _ = load_bundle(args.final_policy, verify_files=True)
    cfg = load_config(overrides={
        **policy_cfg, "training_stage": "flight", "use_bank_resets": True,
        "domain_randomization": False, "obs_noise_enable": False,
        "expert_chain_termination": False,
        "stage_reachability_objective": "phase_balanced_rsi",
    })
    env = OrangeBikeDVGC(
        cfg, snapshot_bank=bank, cert_bank=entry, stage_support_bank=support
    )
    records = fixed_parent_diverse(bank.records, args.states_per_stage)
    baseline_result = evaluate_policy(env, baseline, records, args.seed)
    final_result = evaluate_policy(env, final, records, args.seed)
    report = {
        "status": "PASS",
        "artifact_role": "read_only_phase_balanced_local_objective_evaluation",
        "formal_tube_or_jel": False,
        "fresh_stage_entry_latch": True,
        "baseline": baseline_result, "final": final_result,
        "local_entry_improvement": final_result["entries"] > baseline_result["entries"],
        "phase_bank_sha256": file_sha256(args.phase_bank),
        "descent_entry_support_bank_sha256": file_sha256(args.descent_entry_support_bank),
        "baseline_policy_params_sha256": file_sha256(Path(args.baseline_policy) / "params.pkl"),
        "final_policy_params_sha256": file_sha256(Path(args.final_policy) / "params.pkl"),
    }
    save_json(output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
