#!/usr/bin/env python3
"""Evaluate a fixed completed unified policy from the canonical natural start."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.unified_natural_evaluation import run_canonical_natural_evaluation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_canonical_natural_evaluation(
        args.config,
        args.checkpoint,
        args.output_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
