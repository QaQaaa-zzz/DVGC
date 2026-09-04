#!/usr/bin/env python3
"""Train a unified policy from frozen preceding-policy Actor parameters.

The historical filename is retained for command compatibility. Observation
normalizer and Actor come from the declared frozen unified policy; critic and
optimizer remain fresh.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from brax.training.agents.ppo import train as ppo_train

import jit_dvgc.unified_formal as flat_formal
import jit_dvgc.training.formal as canonical_formal
from jit_dvgc.unified_formal import (
    load_frozen_actor_restore_params,
    load_unified_actor_warm_start_config,
)


def _read_json(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _load_warm_target_config(path: Path):
    return load_unified_actor_warm_start_config(path)


def _load_pi0_restore_params(config_path: Path):
    """Compatibility alias used by historical wrappers and tests."""
    return load_frozen_actor_restore_params(config_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    raw = _read_json(args.config)
    declared_run = raw.get("run_declaration", {}).get("run_id")
    if declared_run != args.run_id:
        raise ValueError("run-id must match the warm-start config declaration")

    restore_params = load_frozen_actor_restore_params(args.config)

    def warm_trainer(**kwargs):
        if kwargs.get("restore_params") is not None:
            raise ValueError("unexpected pre-existing PPO restore_params")
        if kwargs.get("restore_value_fn") is not False:
            raise ValueError("actor-only warm-start requires a fresh critic")
        call = dict(kwargs)
        call["restore_params"] = restore_params
        call["restore_value_fn"] = False
        return ppo_train.train(**call)

    previous_flat = flat_formal.load_unified_formal_config
    previous_canonical = canonical_formal.load_unified_formal_config
    flat_formal.load_unified_formal_config = _load_warm_target_config
    canonical_formal.load_unified_formal_config = _load_warm_target_config
    try:
        result = canonical_formal.run_unified_formal(
            args.config,
            args.run_id,
            trainer=warm_trainer,
        )
    finally:
        flat_formal.load_unified_formal_config = previous_flat
        canonical_formal.load_unified_formal_config = previous_canonical

    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
