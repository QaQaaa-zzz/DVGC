#!/usr/bin/env python3
"""Collect a bounded handoff bank from a frozen Phase U checkpoint."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import jax
from jax import numpy as jp
from jit_dvgc.checkpoint import CheckpointIdentity, load_checkpoint
from jit_dvgc.config import load_config, file_sha256
from jit_dvgc.constants import ACTION_ORDER, ACTOR_FRAME_FIELDS, ACTOR_TASK_FIELDS
from jit_dvgc.env import TwoPhaseBikeEnv
from jit_dvgc.handoff_bank import BankCollector, collect_streaming_rollout, pytree_sha256
from jit_dvgc.handoff_snapshot import compatibility_identity
from jit_dvgc.ppo import make_checkpoint_policy

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-ticks", type=int, default=128)
    p.add_argument("--max-transitions", type=int, default=20_000)
    p.add_argument("--episodes", type=int, default=2)
    args = p.parse_args()
    config = load_config(args.config)
    env = TwoPhaseBikeEnv(config)
    identity = CheckpointIdentity(config.config_sha256, env._bundle.xml_sha256, ACTOR_FRAME_FIELDS, ACTOR_TASK_FIELDS, ACTION_ORDER)
    payload = load_checkpoint(args.checkpoint, expected=identity)
    policy_sha = pytree_sha256(payload.actor_params)
    out = Path(__file__).parents[1] / "runs" / "handoff_bank" / args.run_id
    collector = BankCollector(out, "bounded online handoff snapshot collection", str(args.checkpoint), config.config_sha256, env._bundle.xml_sha256, policy_sha, args.max_transitions)
    collector.max_ticks = args.max_ticks
    deterministic_policy = make_checkpoint_policy(env, payload, deterministic=True)
    reset_fn = jax.jit(env.reset_natural)
    policy_step = jax.jit(lambda s, k: env.step(s, deterministic_policy(s.obs, k)[0]))
    try:
        for episode in range(args.episodes):
            if collector.transitions >= args.max_transitions:
                break
            state = reset_fn(jax.random.key(args.seed + episode))
            collect_streaming_rollout(collector, state, seed=args.seed + episode,
                step=lambda s, ep=episode: policy_step(s, jax.random.fold_in(jax.random.key(args.seed + ep), int(s.info["episode_step"]))),
                capture=lambda s, parent, tick: env.capture_handoff_snapshot(s, policy_sha256=policy_sha, parent_trajectory=parent, parent_tick=tick),
                parent_trajectory=f"seed-{args.seed + episode}", max_ticks=min(args.max_ticks, args.max_transitions - collector.transitions))
        manifest = collector.close()
    except Exception as exc:
        manifest = collector.close(status="failed", failure=f"{type(exc).__name__}: {exc}")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["status"] == "closed" else 1

if __name__ == "__main__":
    raise SystemExit(main())
