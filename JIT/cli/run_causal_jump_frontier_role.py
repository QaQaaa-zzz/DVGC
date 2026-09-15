#!/usr/bin/env python3
"""Run one ground-connected causal frontier logical role."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.causal_frontier_protocol import run_causal_frontier_role


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--role",
        choices=("train", "calibration", "acceptance"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_causal_frontier_role(
        plan_path=args.plan,
        role=args.role,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
