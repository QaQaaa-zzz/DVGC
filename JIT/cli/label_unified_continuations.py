#!/usr/bin/env python3
"""Label real-dynamics envelope candidates under one frozen unified policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax

from jit_dvgc.checkpoint import load_checkpoint
from jit_dvgc.config import file_sha256
from jit_dvgc.ppo import make_checkpoint_policy
from jit_dvgc.unified_continuation_labels import (
    DEFAULT_UNIFIED_CONTINUATION_MAX_TICKS,
    DEFAULT_UNIFIED_CONTINUATION_PROTOCOL_SEED,
    label_unified_continuations,
)
from jit_dvgc.unified_formal import build_unified_formal_environment
from jit_dvgc.unified_policy_freeze import load_frozen_unified_manifest
from jit_dvgc.unified_training import checkpoint_identity


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-policy", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=DEFAULT_UNIFIED_CONTINUATION_MAX_TICKS,
    )
    parser.add_argument(
        "--protocol-seed",
        type=int,
        default=DEFAULT_UNIFIED_CONTINUATION_PROTOCOL_SEED,
    )
    args = parser.parse_args()

    if args.max_ticks <= 0:
        parser.error("--max-ticks must be positive")
    if args.output_dir.exists():
        parser.error(f"--output-dir already exists: {args.output_dir}")

    frozen = load_frozen_unified_manifest(args.frozen_policy)
    record = frozen["policy"]
    if record["policy_role"] != "envelope_expansion_authority":
        raise ValueError("frozen unified policy is not an expansion authority")

    if jax.default_backend() != "gpu":
        raise RuntimeError("unified continuation labeling requires the visible JAX GPU")
    config, _artifact, env = build_unified_formal_environment(
        Path(record["formal_config"])
    )
    if config.config_sha256 != record["formal_config_sha256"]:
        raise ValueError("unified continuation formal config mismatch")
    if env._bundle.xml_sha256 != record["xml_sha256"]:
        raise ValueError("unified continuation runtime XML mismatch")
    if int(args.max_ticks) != int(config.ppo.episode_horizon):
        raise ValueError(
            "unified continuation labeling must use the frozen policy episode horizon"
        )

    payload = load_checkpoint(
        Path(record["checkpoint"]),
        expected=checkpoint_identity(config, env),
    )
    if int(payload.training_transitions) != int(record["source_training_transitions"]):
        raise ValueError("unified continuation checkpoint transition mismatch")
    if file_sha256(Path(record["checkpoint"]) / "payload.pkl") != record["payload_sha256"]:
        raise ValueError("unified continuation checkpoint payload SHA-256 mismatch")
    policy = jax.jit(make_checkpoint_policy(env, payload, deterministic=True))

    report = label_unified_continuations(
        args.catalog,
        args.output_dir,
        env=env,
        policy=policy,
        policy_record=record,
        frozen_manifest_sha256=file_sha256(args.frozen_policy),
        max_ticks=args.max_ticks,
        protocol_seed=args.protocol_seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
