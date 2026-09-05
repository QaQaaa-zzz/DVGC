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

    load_unified_actor_warm_start_config(args.config)
    result = canonical_formal.run_unified_formal(args.config, args.run_id)

    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
