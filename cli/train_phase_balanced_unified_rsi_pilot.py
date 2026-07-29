"""Run the bounded first joint Tube-RSI pilot for the distilled shared actor."""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from cli.train_stage_reachability_model import parent_key
from dvgc.config import file_sha256
from dvgc.runtime import save_json


STAGES = ("takeoff", "ascent", "apex", "descent", "landing")
EFFECTIVE_STEPS = 4096


def select_parent_diverse(records: list[dict], per_stage: int = 3) -> list[dict]:
    selected = []
    for stage in STAGES:
        rows = sorted((row for row in records if row.get("phase_rsi_stage") == stage),
                      key=lambda row: str(row["id"]))
        seen = set(); stage_rows = []
        for row in rows:
            parent = str(row.get("reset_parent_id") or parent_key(row))
            if parent in seen:
                continue
            seen.add(parent); stage_rows.append(row)
            if len(stage_rows) == per_stage:
                break
        if len(stage_rows) < min(per_stage, len(rows)):
            raise ValueError(f"unable to choose parent-diverse fixed {stage} evaluation")
        selected.extend(stage_rows)
    return selected


def acceptance(baseline: dict, final: dict, finite: bool, drift: dict) -> dict:
    before = baseline["by_stage"]; after = final["by_stage"]
    retention = {
        stage: after[stage]["final_states"] >= before[stage]["final_states"]
        for stage in ("descent", "landing")
    }
    improvement = (after["takeoff"]["final_states"] + after["ascent"]["final_states"]
                   + after["apex"]["final_states"]
                   > before["takeoff"]["final_states"] + before["ascent"]["final_states"]
                   + before["apex"]["final_states"])
    total_improvement = final["final_states"] > baseline["final_states"]
    downstream_drift = {
        stage: drift["by_stage"][stage]["rms"] <= .02
        and drift["by_stage"][stage]["max"] <= .05
        for stage in ("descent", "landing")
    }
    return {
        "descent_retention": retention["descent"], "landing_retention": retention["landing"],
        "descent_action_drift": downstream_drift["descent"],
        "landing_action_drift": downstream_drift["landing"],
        "upstream_final_improvement": improvement, "total_final_improvement": total_improvement,
        "finite_training": finite, "no_new_nonfinite": final["nonfinite"] == 0,
        "promote": all(retention.values()) and all(downstream_drift.values())
                   and finite and final["nonfinite"] == 0
                   and (improvement or total_improvement),
    }


def phase_action_drift(before: np.ndarray, after: np.ndarray,
                       phases: list[str]) -> dict:
    """Summarize policy drift over every authoritative phase-bank state."""
    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    if before.shape != after.shape or before.ndim != 2 or len(phases) != len(before):
        raise ValueError("invalid phase action arrays")
    delta = after - before

    def summarize(indices: list[int]) -> dict:
        values = delta[indices]
        action_l2 = np.linalg.norm(values, axis=1)
        return {
            "states": len(indices),
            "rms": float(np.sqrt(np.mean(values * values))),
            "mean_action_l2": float(np.mean(action_l2)),
            "p95_action_l2": float(np.percentile(action_l2, 95)),
            "max_action_l2": float(np.max(action_l2)),
            "max": float(np.max(np.abs(values))),
        }

    by_stage = {}
    for stage in STAGES:
        indices = [index for index, value in enumerate(phases) if value == stage]
        if not indices:
            raise ValueError(f"missing {stage} action-drift states")
        by_stage[stage] = summarize(indices)
    return {"all": summarize(list(range(len(phases)))), "by_stage": by_stage}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-bank", required=True)
    parser.add_argument("--initial-policy", required=True)
    parser.add_argument("--canonical-entry-bank", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--teacher-dataset", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = Path(args.run)
    if root.exists():
        raise SystemExit(f"refusing to overwrite {root}")

    import jax
    import jax.numpy as jnp
    from cli.runtime_gate import source_fingerprint
    from dvgc.bank import SnapshotBank
    from dvgc.config import load_config, save_config
    from dvgc.descent_supervised import build_actor_tools
    from dvgc.env import END_NONFINITE, END_REASON, OrangeBikeDVGC
    from dvgc.policy import load_bundle, save_bundle
    from dvgc.rollout import frozen_rollout, restore_snapshot
    from dvgc.runtime import (
        build_inference, make_ppo_train_fn, ppo_effective_timesteps,
        validate_ppo_batch_layout,
    )

    gate = json.loads(Path("docs/RUNTIME_GATE.json").read_text())
    if gate.get("status") != "PASS" or gate.get("source_fingerprint") != source_fingerprint(Path.cwd()):
        raise SystemExit("runtime gate stale")
    preflight = json.loads(Path(args.preflight).read_text())
    if preflight.get("status") != "PASS" or preflight.get("PPO_authorization") is not True:
        raise SystemExit("unified RSI engineering preflight is not PASS")
    if preflight.get("phase_bank_sha256") != file_sha256(args.phase_bank):
        raise SystemExit("preflight phase-bank provenance mismatch")
    bank = SnapshotBank.load(args.phase_bank); entry = SnapshotBank.load(args.canonical_entry_bank)
    if bank.metadata.get("artifact_role") != "phase_balanced_tube_rsi_reset_bank":
        raise SystemExit("invalid phase-balanced bank role")
    params, policy_cfg, manifest = load_bundle(args.initial_policy, verify_files=True)
    if manifest.get("artifact_role") != "final_shared_policy_initialization":
        raise SystemExit("initial policy is not the bounded distillation output")
    if preflight.get("policy_params_sha256") != file_sha256(Path(args.initial_policy) / "params.pkl"):
        raise SystemExit("preflight policy provenance mismatch")
    validate_ppo_batch_layout(num_envs=128, batch_size=32, num_minibatches=4)
    effective = ppo_effective_timesteps(
        1600, unroll_length=32, batch_size=32, num_minibatches=4, num_evals=2
    )
    if effective != EFFECTIVE_STEPS:
        raise SystemExit(f"unexpected effective pilot size {effective}")
    cfg = load_config(overrides={
        **policy_cfg, "training_stage": "flight", "use_bank_resets": True,
        "expert_chain_termination": False, "stage_reachability_objective": "",
    })
    eval_cfg = load_config(overrides={
        **policy_cfg, "training_stage": "flight", "use_bank_resets": False,
        "domain_randomization": False, "obs_noise_enable": False,
        "expert_chain_termination": False, "stage_reachability_objective": "",
    })
    env = OrangeBikeDVGC(cfg, snapshot_bank=bank, cert_bank=entry)
    eval_env = OrangeBikeDVGC(eval_cfg, snapshot_bank=SnapshotBank(), cert_bank=entry)
    fixed = select_parent_diverse(bank.records, 3)

    def evaluate(policy, seed):
        infer = build_inference(eval_env, policy, deterministic=True); step = jax.jit(eval_env.step)
        rows = []
        for index, record in enumerate(fixed):
            key = jax.random.PRNGKey(seed + index)
            _, outcome = frozen_rollout(
                eval_env, infer, restore_snapshot(eval_env, record, key), key,
                horizon=200, step_fn=step,
            )
            rows.append({
                "stage": record["phase_rsi_stage"], "record_id": record["id"],
                "final_recovery": bool(outcome["final"]), "chain": bool(outcome["chain"]),
                "physical_failure": bool(outcome["terminated"] and not outcome["final"]),
                "timeout": bool(outcome["truncated"]),
                "termination_reason": END_REASON.get(outcome["end_code"], "unknown"),
                "steps": outcome["steps"],
            })
        by_stage = {}
        for stage in STAGES:
            values = [row for row in rows if row["stage"] == stage]
            by_stage[stage] = {
                "states": len(values), "final_states": sum(row["final_recovery"] for row in values),
                "chain_states": sum(row["chain"] for row in values),
                "physical_failures": sum(row["physical_failure"] for row in values),
                "timeouts": sum(row["timeout"] for row in values),
                "termination_reasons": dict(Counter(row["termination_reason"] for row in values)),
            }
        return {
            "states": len(rows), "final_states": sum(row["final_recovery"] for row in rows),
            "chain_states": sum(row["chain"] for row in rows),
            "physical_failures": sum(row["physical_failure"] for row in rows),
            "timeouts": sum(row["timeout"] for row in rows),
            "nonfinite": sum(row["termination_reason"] == "nonfinite" for row in rows),
            "by_stage": by_stage, "rows": rows,
        }

    root.mkdir(parents=True); save_config(cfg, root / "effective_config.json")
    save_json(root / "cost_estimate.json", {
        "effective_PPO_steps": EFFECTIVE_STEPS, "fixed_evaluation_states": len(fixed),
        "estimated_upper_seconds": 7200, "pilot_fraction_of_100k": .04096,
        "longer_training_authorized": False,
    })
    baseline = evaluate(params, 10_900_000)
    # A shared initialization that already erased the frozen downstream skills
    # must be repaired at distillation, not hidden by PPO.
    downstream_start_ok = (baseline["by_stage"]["landing"]["final_states"] >= 2
                           and baseline["by_stage"]["descent"]["final_states"] >= 1)
    save_json(root / "baseline_evaluation.json", {
        **baseline, "downstream_start_gate": downstream_start_ok,
    })
    if not downstream_start_ok:
        save_json(root / "report.json", {
            "status": "DISTILLATION_RETENTION_BLOCKER", "PPO_started": False,
            "baseline": baseline, "reason": "Landing/Descent fixed probe below start gate",
        })
        raise SystemExit(40)

    progress_rows = []; started = time.time()
    def progress(step, metrics):
        row = {"step": int(step), **{key: float(value) for key, value in metrics.items()
                                      if hasattr(value, "__float__") and math.isfinite(float(value))}}
        progress_rows.append(row); save_json(root / "training_metrics.json", {
            "status": "running", "effective_steps": EFFECTIVE_STEPS, "progress": progress_rows,
        })
    train = make_ppo_train_fn(
        timesteps=1600, episode_length=int(cfg.episode_length), num_envs=128,
        num_eval_envs=64, num_evals=2, seed=args.seed, learning_rate=1e-5,
        entropy_cost=1e-4, reward_scaling=.1, checkpoint_dir=root / "orbax",
        unroll_length=32, batch_size=32, num_minibatches=4, num_updates_per_batch=2,
        discounting=.995, gae_lambda=.97, clipping_epsilon=.10, max_grad_norm=.75,
        restore_params=params,
    )
    _, final_params, final_metrics = train(environment=env, progress_fn=progress, eval_env=eval_env)
    finite = all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(final_params))
    # Use exactly the baseline seed namespace so policy change is the only
    # cause of an outcome change in the fixed pilot evaluation.
    final = evaluate(final_params, 10_900_000)
    observations = np.asarray([
        row["policy_state"]["actor_observation"] for row in bank.records
    ], dtype=np.float32)
    phases = [str(row["phase_rsi_stage"]) for row in bank.records]
    _, before_actor, _ = build_actor_tools(env, params)
    _, after_actor, _ = build_actor_tools(env, final_params)
    before_actions = np.asarray(before_actor(params[1], observations))
    after_actions = np.asarray(after_actor(final_params[1], observations))
    drift = phase_action_drift(before_actions, after_actions, phases)
    decision = acceptance(baseline, final, finite, drift)
    save_bundle(
        root / "policy", params=final_params, config=cfg, xml_path=cfg.xml_path,
        candidate_bank=args.phase_bank, downstream_bank=args.canonical_entry_bank,
        policy_version="phase-balanced-unified-rsi-pilot-seed0-v1",
        extra={
            "artifact_role": "unified_rsi_pilot_checkpoint", "formal_tube_or_jel": False,
            "promoted": decision["promote"], "effective_steps": EFFECTIVE_STEPS,
            "initial_policy_params_sha256": file_sha256(Path(args.initial_policy) / "params.pkl"),
            "teacher_dataset_sha256": file_sha256(args.teacher_dataset),
        },
    )
    report = {
        "status": "PASS_PROMOTE" if decision["promote"] else "NO_PROMOTION",
        "PPO_started": True, "effective_steps": EFFECTIVE_STEPS,
        "baseline": baseline, "final": final, "acceptance": decision,
        "phase_action_drift": drift,
        "finite_parameters": finite, "final_metrics": final_metrics,
        "elapsed_seconds": time.time() - started,
        "policy_params_sha256": file_sha256(root / "policy" / "params.pkl"),
        "formal_tube_or_jel": False,
        "next_gate": ("expanded fixed evaluation before any larger block" if decision["promote"]
                      else "diagnose phase reward/reset/action drift; no blind budget increase"),
    }
    save_json(root / "report.json", report)
    save_json(root / "training_metrics.json", {
        "status": "completed", "effective_steps": EFFECTIVE_STEPS, "progress": progress_rows,
        "elapsed_seconds": report["elapsed_seconds"],
    })
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"baseline", "final", "final_metrics"}}, indent=2))


if __name__ == "__main__":
    main()
