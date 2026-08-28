#!/usr/bin/env python3
"""Apply the locked TRAIN V_up boundary recipe once to validation seed(s)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from jit_dvgc.env import TwoPhaseBikeEnv
from jit_dvgc.expert_freeze import load_frozen_manifest, verify_frozen_record
from jit_dvgc.ppo import make_checkpoint_policy
from jit_dvgc.upstream_boundary_lock import load_boundary_lock
from jit_dvgc.upstream_boundary_validation import (
    collect_locked_validation_candidates,
    select_locked_validation_anchors,
)


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--nominal-catalog", type=Path, required=True)
    parser.add_argument("--nominal-labels", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol-seed", type=int, default=820404)
    args = parser.parse_args()

    lock = load_boundary_lock(args.lock)
    anchors, audit = select_locked_validation_anchors(args.nominal_catalog, args.nominal_labels, args.lock)

    frozen = load_frozen_manifest(args.frozen_manifest)
    record = frozen["experts"]["pi_up_star"]
    config, payload = verify_frozen_record(record)
    if config.phase != "propulsion_ascent":
        raise ValueError("pi_up_star must use propulsion_ascent config")
    for key, actual in (
        ("frozen_pi_up_actor_sha256", record["actor_sha256"]),
        ("frozen_pi_up_payload_sha256", record["payload_sha256"]),
        ("frozen_pi_up_config_sha256", record["config_sha256"]),
        ("xml_sha256", record["xml_sha256"]),
    ):
        if str(lock[key]) != str(actual):
            raise ValueError(f"frozen pi_up_star identity mismatch for {key}")

    env = TwoPhaseBikeEnv(config)
    if str(lock["xml_sha256"]) != str(env._bundle.xml_sha256):
        raise ValueError("locked XML identity mismatch")
    policy = make_checkpoint_policy(env, payload, deterministic=True)
    step_fn = jax.jit(env.step)
    base_key = jax.random.PRNGKey(args.protocol_seed)

    def policy_action(state, variant_index: int, perturb_step: int):
        key = jax.random.fold_in(base_key, int(variant_index))
        key = jax.random.fold_in(key, int(perturb_step))
        result = policy(state.obs, key)
        action = result[0] if isinstance(result, tuple) else result
        return np.asarray(jax.device_get(action), dtype=np.float32)

    def capture(state, anchor):
        return env.capture_handoff_snapshot(
            state,
            policy_sha256=record["actor_sha256"],
            parent_trajectory=str(anchor.row["parent_group_id"]),
            parent_tick=int(anchor.row["tick"]),
            policy_identity="pi_up_star_locked_boundary_validation",
        )

    report = collect_locked_validation_candidates(
        anchors,
        args.output_dir,
        lock_path=args.lock,
        restore=env.restore_handoff_snapshot,
        policy_action=policy_action,
        step=step_fn,
        capture=capture,
        protocol_seed=args.protocol_seed,
    )
    _write_json(Path(args.output_dir) / "anchor_audit.json", audit)
    print(json.dumps({"anchor_audit": audit, "collection": {k: v for k, v in report.items() if k != "entries"}}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
