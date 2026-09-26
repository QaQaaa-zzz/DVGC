#!/usr/bin/env python3
"""Compile and audit the stable unified reset sampler without stepping the env."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from jit_dvgc.tube_rsi import PHASE_DOWNSTREAM, PHASE_UPSTREAM
from jit_dvgc.unified_formal import build_unified_formal_environment


def audit_reset_sampler(env, *, sample_count: int) -> dict[str, object]:
    if sample_count <= 1:
        raise ValueError("unified reset smoke requires multiple samples")
    reset = jax.jit(env.reset)
    counts = {"natural": 0, "soft_tube": 0}
    phase_counts = {"downstream": 0, "upstream": 0}
    for seed in range(9_500_001, 9_500_001 + int(sample_count)):
        state = reset(jax.random.PRNGKey(seed))
        jax.block_until_ready(state)
        qpos = np.asarray(jax.device_get(state.data.qpos))
        qvel = np.asarray(jax.device_get(state.data.qvel))
        if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
            raise ValueError("unified reset smoke produced a nonfinite state")
        if bool(np.asarray(jax.device_get(state.info["expert_switching_used"]))):
            raise ValueError("unified reset smoke used expert switching")
        natural = float(
            np.asarray(jax.device_get(state.metrics["reset/source_natural"]))
        )
        soft = float(
            np.asarray(jax.device_get(state.metrics["reset/source_soft_tube"]))
        )
        if (natural, soft) not in {(0.0, 1.0), (1.0, 0.0)}:
            raise ValueError("unified reset source flags are not one-hot")
        phase = int(np.asarray(jax.device_get(state.info["active_phase"])))
        if natural == 1.0:
            counts["natural"] += 1
            if phase != PHASE_UPSTREAM:
                raise ValueError("natural unified reset must start upstream")
        else:
            counts["soft_tube"] += 1
            if phase not in {PHASE_UPSTREAM, PHASE_DOWNSTREAM}:
                raise ValueError("Soft-Tube reset has an invalid phase")
            phase_counts["upstream" if phase == PHASE_UPSTREAM else "downstream"] += 1
    probability = float(env.natural_reset_probability)
    if probability == 0.0:
        if counts["natural"] != 0 or counts["soft_tube"] != sample_count:
            raise ValueError("Soft-Tube-only reset contract drift")
    elif counts["natural"] == 0 or counts["soft_tube"] == 0:
        raise ValueError("mixed reset smoke did not exercise both reset sources")
    return {
        "schema": "jit_pi_unified_reset_smoke_v1",
        "status": "completed",
        "sample_count": int(sample_count),
        "natural_count": counts["natural"],
        "soft_tube_count": counts["soft_tube"],
        "observed_natural_fraction": counts["natural"] / float(sample_count),
        "soft_tube_phase_counts": phase_counts,
        "configured_natural_probability": probability,
        "configured_soft_tube_probability": 1.0 - probability,
        "runtime_naccdmax": int(env._reset_data_naccdmax()),
        "environment_interactions": 0,
        "training_transitions": 0,
        "expert_switching_used": False,
        "validation_data_used": False,
        "test_data_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=256)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError("unified reset smoke requires the visible JAX GPU")
    if args.output.exists():
        raise FileExistsError(f"reset smoke output already exists: {args.output}")
    config, _artifact, env = build_unified_formal_environment(args.config)
    report = audit_reset_sampler(env, sample_count=args.samples)
    report = {
        **report,
        "config": str(args.config.resolve()),
        "config_sha256": config.config_sha256,
        "reset_mixture": config.reset_mixture.as_dict(),
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
