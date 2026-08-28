#!/usr/bin/env python3
"""Compile and audit the fixed Round-1 reset mixture without stepping the env."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax

from jit_dvgc.unified_round1 import (
    audit_round1_reset_sampler,
    build_round1_environment,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=256)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError("Round-1 reset smoke requires the visible JAX GPU")
    if args.output.exists():
        raise FileExistsError(f"reset smoke output already exists: {args.output}")
    config, _artifact, env = build_round1_environment(args.config)
    report = audit_round1_reset_sampler(env, sample_count=args.samples)
    report = {
        **report,
        "config": str(args.config.resolve()),
        "config_sha256": config.config_sha256,
        "reset_mixture": config.raw["reset_mixture"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
