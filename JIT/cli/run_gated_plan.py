#!/usr/bin/env python3
"""Run a finite locked command plan after its external run has completed."""
import argparse
import json
import os
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"

from jit_dvgc.gated_execution import run_gated_plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    result = run_gated_plan(args.plan, args.output_dir, wait=args.wait, poll_seconds=args.poll_seconds)
    print(json.dumps(result, indent=2))
    return 0 if result["phase"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
