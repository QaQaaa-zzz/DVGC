#!/usr/bin/env python3
"""Prepare/run outcome-blind frontier roles for automatic envelope iterations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax

from jit_dvgc.iterative_frontier_protocol import prepare_frontier_plan, run_frontier_role


def main() -> int:
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="command", required=True)

    prepare = subs.add_parser("prepare-plan")
    prepare.add_argument("--selected-policy", type=Path, required=True)
    prepare.add_argument("--source-tube", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--max-parent-groups-per-phase", type=int, default=15)

    role = subs.add_parser("run-role")
    role.add_argument("--plan", type=Path, required=True)
    role.add_argument("--role", choices=("train", "calibration", "acceptance"), required=True)
    role.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "prepare-plan":
        result = prepare_frontier_plan(
            selected_policy=args.selected_policy,
            source_tube=args.source_tube,
            output=args.output,
            max_parent_groups_per_phase=args.max_parent_groups_per_phase,
        )
    else:
        if jax.default_backend() != "gpu":
            raise RuntimeError("iterative frontier rollout requires the visible JAX GPU")
        result = run_frontier_role(
            plan_path=args.plan,
            role=args.role,
            output_dir=args.output_dir,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
