#!/usr/bin/env python3
"""Build the real-rollout nominal centerline for trajectory-centered JIT."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.analysis.nominal_jump_centerline import build_nominal_jump_centerline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-evaluation-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_nominal_jump_centerline(
        canonical_evaluation_report=args.canonical_evaluation_report,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
