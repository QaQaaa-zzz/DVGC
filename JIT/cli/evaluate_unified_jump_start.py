#!/usr/bin/env python3
"""Evaluate a frozen unified policy from the fixed ground jump start."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.unified_natural_evaluation import (
    CANONICAL_ROLLOUT_SEED,
    run_canonical_jump_start_evaluation,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=CANONICAL_ROLLOUT_SEED)
    args = parser.parse_args()
    report = run_canonical_jump_start_evaluation(
        args.config,
        args.checkpoint,
        args.output_dir,
        rollout_seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
