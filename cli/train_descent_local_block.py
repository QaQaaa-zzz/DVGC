"""Run one cumulative 25,600-step candidate-guided local descent PPO block."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config
from dvgc.env import OrangeBikeDVGC
from dvgc.policy import load_bundle, save_bundle
from dvgc.runtime import make_ppo_train_fn, save_json


def _ratios(rows, prefix, names):
    row = next((item for item in reversed(rows) if any(key.startswith(prefix) for key in item)), {})
    values = {name: float(row.get(prefix + name, 0.0)) for name in names}
    total = sum(values.values())
    return {name: value / total if total else 0.0 for name, value in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-policy", required=True)
    parser.add_argument("--bootstrap-bank", required=True)
    parser.add_argument("--candidate-bank", required=True)
    parser.add_argument("--entry-set", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--cumulative-steps", type=int, required=True)
    parser.add_argument("--restore-checkpoint", default="")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=None)
    args = parser.parse_args()
    if args.cumulative_steps not in tuple(range(25600,204801,25600)):
        raise SystemExit("cumulative steps must be a 25,600-step milestone through 204,800")
    run = Path(args.run)
    if run.exists():
        raise SystemExit(f"Run exists: {run}")
    params, policy_cfg, manifest = load_bundle(args.resume_policy, verify_files=True)
    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = manifest.get("ppo_hyperparameters", {}).get("learning_rate")
    if learning_rate is None or float(learning_rate) <= 0:
        raise SystemExit("A positive inherited or explicit learning rate is required")
    candidate_hash = file_sha256(args.candidate_bank)
    bootstrap_hash = file_sha256(args.bootstrap_bank)
    entry_hash = file_sha256(args.entry_set)
    original_policy_hash = file_sha256(Path(args.resume_policy) / "params.pkl")
    training_bank = SnapshotBank.load(args.bootstrap_bank)
    if training_bank.metadata.get("reset_source_protocol", {}).get("source_bank_sha256") != candidate_hash:
        raise SystemExit("Bootstrap reset bank does not match the immutable candidate pool")
    entry = SnapshotBank.load(args.entry_set)
    cfg = load_config(args.config, {
        **policy_cfg,
        "training_stage": "flight",
        "expert_chain_termination": True,
        "descent_local_reward_enable": True,
    })
    env = OrangeBikeDVGC(cfg, snapshot_bank=training_bank, cert_bank=entry)
    eval_cfg = load_config(args.config, {
        **cfg.to_dict(), "domain_randomization": False, "obs_noise_enable": False,
    })
    eval_env = OrangeBikeDVGC(eval_cfg, snapshot_bank=training_bank, cert_bank=entry)
    run.mkdir(parents=True)
    rows = []
    status = {
        "status": "running", "seed": args.seed, "cumulative_effective_steps": args.cumulative_steps,
        "candidate_bank_sha256": candidate_hash, "bootstrap_bank_sha256": bootstrap_hash,
        "entry_set_sha256": entry_hash, "initial_policy_hash": original_policy_hash,
        "restore_checkpoint": str(Path(args.restore_checkpoint).resolve()) if args.restore_checkpoint else None,
        "progress": rows,
    }
    save_json(run / "training_metrics.json", status)

    def progress(step, metrics):
        row = {"step": int(step), **{key: float(value) for key, value in metrics.items() if hasattr(value, "__float__")}}
        rows.append(row)
        save_json(run / "training_metrics.json", status)
        print(f"[descent-local] step={step}")

    kwargs = {"restore_checkpoint_path": args.restore_checkpoint} if args.restore_checkpoint else {"restore_params": params}
    train_fn = make_ppo_train_fn(
        timesteps=args.cumulative_steps, episode_length=int(cfg.episode_length), num_envs=160,
        num_eval_envs=128, num_evals=2, seed=args.seed, learning_rate=float(learning_rate),
        entropy_cost=.001, reward_scaling=.1, checkpoint_dir=run / "orbax", unroll_length=32,
        batch_size=80, num_minibatches=10, num_updates_per_batch=2, discounting=.995,
        gae_lambda=.97, clipping_epsilon=.10, max_grad_norm=.75, full_reset=True, **kwargs,
    )
    started = time.time()
    _, trained, final_metrics = train_fn(environment=env, progress_fn=progress, eval_env=eval_env)
    nonfinite = sorted({key for row in rows for key, value in row.items() if isinstance(value, float) and not math.isfinite(value)})
    if file_sha256(Path(args.resume_policy) / "params.pkl") != original_policy_hash:
        raise RuntimeError("Immutable source policy changed during local training")
    policy = run / "policy"
    save_bundle(
        policy, params=trained, config=cfg, xml_path=cfg.xml_path, candidate_bank=args.candidate_bank,
        downstream_bank=args.entry_set, policy_version=f"flight-descent-local-{args.cumulative_steps:06d}",
        extra={"stage": "flight", "expert_role": "candidate_guided_local_bootstrap", "seed": args.seed,
               "cumulative_effective_steps": args.cumulative_steps, "initial_policy_hash": original_policy_hash,
               "bootstrap_bank_sha256": bootstrap_hash, "ppo_hyperparameters": {"learning_rate": float(learning_rate),
               "entropy_cost": .001, "reward_scaling": .1, "discounting": .995, "gae_lambda": .97,
               "clipping_epsilon": .10, "max_grad_norm": .75, "num_updates_per_batch": 2}},
    )
    parent_names = [f"p{index:03d}" for index in range(len(env._reset_parent_ids))]
    report = {
        **{key: value for key, value in status.items() if key != "progress"},
        "status": "PASS" if not nonfinite else "FAIL", "elapsed_seconds": time.time() - started,
        "policy_hash": file_sha256(policy / "params.pkl"), "final_metrics": final_metrics,
        "reset_group_episode_ratio": _ratios(rows, "eval/episode_reset/episode/group/", ("provisional_safe", "boundary", "successful_anchor")),
        "reset_group_transition_ratio": _ratios(rows, "eval/episode_reset/transition/group/", ("provisional_safe", "boundary", "successful_anchor")),
        "reset_layer_episode_ratio": _ratios(rows, "eval/episode_reset/episode/layer/", ("late", "middle", "early")),
        "reset_layer_transition_ratio": _ratios(rows, "eval/episode_reset/transition/layer/", ("late", "middle", "early")),
        "reset_parent_episode_ratio": _ratios(rows, "eval/episode_reset/episode/parent/", parent_names),
        "reset_parent_transition_ratio": _ratios(rows, "eval/episode_reset/transition/parent/", parent_names),
        "reset_parent_index": {f"p{index:03d}": name for index, name in enumerate(env._reset_parent_ids)},
        "health": {"nonfinite_metric_keys": nonfinite, "oom": False, "timeout": False},
    }
    save_json(run / "report.json", report)
    status.update({"status": report["status"], "policy": str(policy.resolve()), "report": str((run / "report.json").resolve())})
    save_json(run / "training_metrics.json", status)
    print(json.dumps({key: value for key, value in report.items() if key not in ("final_metrics", "reset_parent_episode_ratio", "reset_parent_transition_ratio", "reset_parent_index")}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
