"""Run a 1,600-step Tube-v5 RSI retention pilot from frozen pi_D."""
from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cli.audit_descent_compact_adapter_v1 import _perturb_batch
from cli.run_backward_descent_nominal_pilot import C_L, EXPECTED, PI_D, PI_L
from cli.runtime_gate import source_fingerprint
from dvgc.backward_search import make_descent_landing_rollout
from dvgc.bank import SnapshotBank
from dvgc.config import ACTION_MAPPING_VERSION, file_sha256, load_config, save_config
from dvgc.descent_supervised import build_actor_tools
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.policy import load_bundle, save_bundle
from dvgc.runtime import (
    make_ppo_train_fn,
    ppo_effective_timesteps,
    save_json,
    validate_ppo_batch_layout,
)


TUBE = Path("runs/descent_reachability_network_v3/independent_tube_extension_3x32_20260729/descent_tube_v5.pkl")
DEFAULT_RUN = Path("runs/descent_reachability_network_v3/tube_v5_rsi_retention_pilot_1600_seed0_20260729")
STEPS = 1600
LR = 1e-5
EVAL_SEED = 4_180_000_000


def record_parent(record: dict) -> str:
    return str(record.get("candidate_id") or record.get("origin_parent") or record["id"])


def build_training_bank(tube: SnapshotBank, source_hash: str) -> SnapshotBank:
    regions: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for record in tube.records:
        if record.get("final", {}).get("label") != "safe" or not record.get("certified_safe"):
            raise ValueError("Tube-RSI input contains a non-safe/non-certified state")
        region = str(record.get("descent_region") or record.get("descent_support_region") or "middle")
        if region not in {"early", "middle", "late"}:
            raise ValueError(f"invalid Descent region: {region}")
        regions[region][record_parent(record)].append(record)
    if set(regions) != {"early", "middle", "late"}:
        raise ValueError(f"Tube-RSI requires all Descent regions, got {sorted(regions)}")
    rows = []
    region_mass = 1.0 / len(regions)
    for region in sorted(regions):
        parents = regions[region]
        parent_mass = region_mass / len(parents)
        for parent in sorted(parents):
            for record in parents[parent]:
                item = copy.deepcopy(record)
                for key in ("certified_safe", "safe_claim_allowed", "tube_metrics_eligible"):
                    item.pop(key, None)
                item.update({
                    "artifact_role": "proposal_support_bank", "training_only": True,
                    "reset_source": "flight_curriculum", "bootstrap_group": "provisional_safe",
                    "descent_layer": region, "reset_parent_id": parent,
                    "reset_weight": parent_mass / len(parents[parent]),
                    "original_bank_sha256": source_hash,
                })
                rows.append(item)
    if not np.isclose(sum(row["reset_weight"] for row in rows), 1.0, atol=1e-7):
        raise ValueError("Tube-RSI reset weights do not sum to one")
    return SnapshotBank(rows, {
        "artifact_role": "proposal_support_bank", "formal_tube_or_jel": False,
        "reset_source_protocol": {"name": "tube_v5_safe_region_parent_balanced",
                                  "source_bank_sha256": source_hash},
        "source_bank_sha256": source_hash,
    })


def evaluate_composite(env: OrangeBikeDVGC, dparams, lparams, records: list[dict], seed: int) -> dict:
    rollout = make_descent_landing_rollout(env, dparams, lparams, horizon=200, residual_ticks=8)
    rows = []
    perturbations = np.asarray([[.02, .02], [.02, -.02], [-.02, .02], [-.02, -.02]], np.float32)
    for index, record in enumerate(records):
        seeds = [seed + index * 100 + branch for branch in range(4)]
        state = _perturb_batch(env, record, seeds, perturbations)
        raw = jax.device_get(rollout(state, jnp.zeros((4, 2, 4), jnp.float32),
                                     jax.random.PRNGKey(seed + index)))
        evidence = []
        for branch in range(4):
            code = int(np.asarray(raw["end_code"])[branch])
            evidence.append({
                "branch": branch, "final_recovery": bool(np.asarray(raw["final_recovery"])[branch]),
                "chain": bool(np.asarray(raw["downstream_entry"])[branch]),
                "end_code": code, "end_reason": END_REASON.get(code, "unknown"),
                "termination_tick": int(np.asarray(raw["termination_tick"])[branch]),
            })
        finals = sum(row["final_recovery"] for row in evidence)
        rows.append({"id": record["id"], "parent": record_parent(record),
                     "region": record.get("descent_region") or record.get("descent_support_region") or "middle",
                     "final": finals, "chain_final": sum(row["chain"] and row["final_recovery"] for row in evidence),
                     "P1": finals >= 3, "evidence": evidence})
    return {
        "states": len(rows), "P1_states": sum(row["P1"] for row in rows),
        "final_branches": sum(row["final"] for row in rows),
        "chain_final_branches": sum(row["chain_final"] for row in rows),
        "regions": {region: {
            "states": sum(row["region"] == region for row in rows),
            "P1": sum(row["region"] == region and row["P1"] for row in rows),
            "final_branches": sum(row["final"] for row in rows if row["region"] == region),
        } for region in ("early", "middle", "late")},
        "failure_reasons": dict(Counter(event["end_reason"] for row in rows for event in row["evidence"]
                                         if not event["final_recovery"])),
        "rows": rows,
    }


def action_drift(env: OrangeBikeDVGC, before, after, records: list[dict]) -> dict:
    _, before_actor, _ = build_actor_tools(env, before)
    _, after_actor, _ = build_actor_tools(env, after)
    observations = jnp.asarray(np.asarray([row["policy_state"]["actor_observation"] for row in records], np.float32))
    old = np.asarray(before_actor(before[1], observations))
    new = np.asarray(after_actor(after[1], observations))
    delta = new - old
    return {"rms": float(np.sqrt(np.mean(delta * delta))),
            "max": float(np.max(np.abs(delta))),
            "saturation_fraction": float(np.mean(np.abs(new) >= .95))}


def acceptance(baseline: dict, final: dict, drift: dict, finite: bool) -> dict:
    return {
        "P1_state_retention": final["P1_states"] >= baseline["P1_states"],
        "Final_branch_retention": final["final_branches"] >= baseline["final_branches"],
        "region_retention": all(final["regions"][region]["P1"] >= baseline["regions"][region]["P1"]
                                for region in ("early", "middle", "late")),
        "action_RMS": drift["rms"] <= .02,
        "action_max": drift["max"] <= .05,
        "finite": finite,
        "measurable_improvement": (final["P1_states"] > baseline["P1_states"]
                                   or final["final_branches"] > baseline["final_branches"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=str(DEFAULT_RUN))
    parser.add_argument("--tube", default=str(TUBE))
    args = parser.parse_args()
    root, tube_path = Path(args.run), Path(args.tube)
    if root.exists():
        raise SystemExit(f"refusing overwrite {root}")
    gate = json.loads(Path("docs/RUNTIME_GATE.json").read_text())
    fingerprint = source_fingerprint(Path.cwd())
    if gate.get("status") != "PASS" or gate.get("source_fingerprint") != fingerprint:
        raise SystemExit("runtime gate stale")
    tube = SnapshotBank.load(tube_path)
    if tube.metadata.get("artifact_role") != "certified_tube" or not tube.metadata.get("independent_audit"):
        raise SystemExit("Tube input is not independently certified")
    if file_sha256(tube_path) != "2034a5963b30c934795d36872085f7157c3e022c8efde69281ebab7592503593":
        raise SystemExit("unexpected Tube-v5 identity")
    dparams, policy_cfg, manifest = load_bundle(PI_D, verify_files=True)
    lparams, _, _ = load_bundle(PI_L, verify_files=True)
    if file_sha256(PI_D / "params.pkl") != EXPECTED["pi_D"] or file_sha256(PI_L / "params.pkl") != EXPECTED["pi_L"]:
        raise SystemExit("frozen policy identity mismatch")
    source_hash = file_sha256(tube_path)
    training_bank = build_training_bank(tube, source_hash)
    root.mkdir(parents=True)
    training_bank.save(root / "training_bank.pkl")
    cfg = load_config("configs/backward_descent_rsi_pilot_v1.json", {
        **policy_cfg, "training_stage": "flight", "use_bank_resets": True,
        "expert_chain_termination": True, "descent_local_reward_enable": True,
        "domain_randomization": False, "obs_noise_enable": False, "episode_length": 64,
    })
    if file_sha256(cfg.xml_path) != EXPECTED["xml"] or cfg.action_mapping_version != ACTION_MAPPING_VERSION:
        raise SystemExit("runtime model mismatch")
    save_config(cfg, root / "effective_config.json")
    save_json(root / "cost_estimate.json", {"estimated_seconds": 1800, "effective_PPO_steps": STEPS,
        "baseline_rollouts": len(tube.records) * 4, "post_rollouts": len(tube.records) * 4,
        "longer_block_authorized": False})
    save_json(root / "manifest.json", {
        "status": "FROZEN_BEFORE_TRAINING", "tube_sha256": source_hash,
        "pi_D": EXPECTED["pi_D"], "pi_L": EXPECTED["pi_L"], "C_L": EXPECTED["C_L"],
        "xml": EXPECTED["xml"], "seed": 0, "effective_steps": STEPS,
        "learning_rate": LR, "initial_policy": "frozen_pi_D_without_compact_adapter",
        "evaluation": "pi_D_candidate_to_frozen_pi_L Final-Recovery; Chain separate",
        "acceptance": "no fixed-Tube regression + bounded action drift + measurable improvement",
    })
    env = OrangeBikeDVGC(cfg, snapshot_bank=training_bank, cert_bank=SnapshotBank.load(C_L))
    eval_cfg = load_config("configs/backward_descent_rsi_pilot_v1.json", {
        **cfg.to_dict(), "use_bank_resets": False, "expert_chain_termination": False,
        "domain_randomization": False, "obs_noise_enable": False,
    })
    eval_env = OrangeBikeDVGC(eval_cfg, snapshot_bank=SnapshotBank(), cert_bank=SnapshotBank.load(C_L))
    reset = jax.jit(env.reset)(jax.random.PRNGKey(0))
    step = jax.jit(env.step)(reset, jnp.zeros((4,), jnp.float32))
    preflight = {
        "model_load": True, "reset_finite": bool(np.isfinite(np.asarray(reset.data.qpos)).all()),
        "step_finite": bool(np.isfinite(np.asarray(step.data.qpos)).all()),
        "training_records": len(training_bank.records), "weight_sum": float(sum(
            row["reset_weight"] for row in training_bank.records)),
        "regions": dict(Counter(row["descent_layer"] for row in training_bank.records)),
    }
    save_json(root / "preflight.json", preflight)
    baseline = evaluate_composite(eval_env, dparams, lparams, tube.records, EVAL_SEED)
    save_json(root / "baseline_evaluation.json", baseline)

    num_envs, batch_size, minibatches, evals = 50, 25, 2, 2
    validate_ppo_batch_layout(num_envs=num_envs, batch_size=batch_size, num_minibatches=minibatches)
    effective = ppo_effective_timesteps(STEPS, unroll_length=32, batch_size=batch_size,
                                        num_minibatches=minibatches, num_evals=evals)
    if effective != STEPS:
        raise SystemExit(f"unexpected effective budget: {effective}")
    progress = []
    def progress_fn(step_count, metrics):
        progress.append({"effective_steps": int(step_count), **{
            key: float(value) for key, value in metrics.items() if np.asarray(value).shape == ()}})
        save_json(root / "training_progress.json", {"status": "running", "progress": progress})
        print(f"[tube-v5-rsi] effective_steps={int(step_count)}", flush=True)
    train_fn = make_ppo_train_fn(
        timesteps=STEPS, episode_length=64, num_envs=num_envs, num_eval_envs=16,
        num_evals=evals, seed=0, learning_rate=LR,
        entropy_cost=float(manifest["ppo_hyperparameters"]["entropy_cost"]),
        reward_scaling=.1, checkpoint_dir=root / "orbax", unroll_length=32,
        batch_size=batch_size, num_minibatches=minibatches, num_updates_per_batch=2,
        discounting=.995, gae_lambda=.97, clipping_epsilon=.10, max_grad_norm=.75,
        restore_params=dparams, full_reset=True,
    )
    started = time.time()
    try:
        _, final_params, final_metrics = train_fn(environment=env, progress_fn=progress_fn, eval_env=env)
    except BaseException as error:
        save_json(root / "training_integrity.json", {"status": "FAIL", "error": str(error),
            "error_type": type(error).__name__, "PPO_authorization": False})
        raise
    finite = not any(not math.isfinite(value) for row in progress for value in row.values()
                     if isinstance(value, float))
    save_bundle(root / "checkpoint_1600", params=final_params, config=cfg, xml_path=cfg.xml_path,
                candidate_bank=root / "training_bank.pkl", downstream_bank=C_L,
                policy_version="descent-tube-v5-rsi-retention-1600",
                extra={"artifact_role": "bounded_tube_rsi_pilot", "effective_steps": STEPS,
                       "source_tube_sha256": source_hash, "initial_policy_hash": EXPECTED["pi_D"]})
    final = evaluate_composite(eval_env, final_params, lparams, tube.records, EVAL_SEED)
    drift = action_drift(eval_env, dparams, final_params, tube.records)
    checks = acceptance(baseline, final, drift, finite)
    accepted = all(checks.values())
    integrity = {"status": "PASS" if finite else "FAIL", "finite": finite,
                 "oom": False, "timeout": False, "effective_steps": STEPS,
                 "elapsed_seconds": time.time() - started, "runtime_fingerprint": fingerprint}
    save_json(root / "final_evaluation.json", final)
    save_json(root / "training_integrity.json", integrity)
    report = {
        "status": "ACCEPT" if accepted else "REJECT", "artifact_role": "bounded_tube_rsi_pilot",
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "tube_sha256": source_hash, "baseline": {key: baseline[key] for key in (
            "states", "P1_states", "final_branches", "chain_final_branches", "regions", "failure_reasons")},
        "final": {key: final[key] for key in (
            "states", "P1_states", "final_branches", "chain_final_branches", "regions", "failure_reasons")},
        "action_drift": drift, "checks": checks, "integrity": integrity,
        "final_metrics": final_metrics, "PPO_authorization": "next_1600_block" if accepted else False,
        "formal_tube_or_jel": False,
    }
    save_json(root / "DESCENT_TUBE_V5_RSI_RETENTION_PILOT_V1_REPORT.json", report)
    save_json(root / "completed.json", {"status": report["status"],
        "next": "cumulative_3200_block" if accepted else "retention_failure_diagnosis"})
    print(json.dumps({key: report[key] for key in (
        "status", "baseline", "final", "action_drift", "checks", "integrity", "PPO_authorization")}, indent=2))


if __name__ == "__main__":
    main()
