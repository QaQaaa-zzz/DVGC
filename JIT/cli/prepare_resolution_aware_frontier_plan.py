#!/usr/bin/env python3
"""Revise an outcome-blind frontier plan around one nominal jump centerline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.acquisition.resolution_frontier import revise_frontier_plan_for_resolution_cells


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--source-tube", type=Path, required=True)
    parser.add_argument("--capability-geometry-summary", type=Path, required=True)
    parser.add_argument("--nominal-centerline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-parent-cells-per-phase", type=int, default=25)
    parser.add_argument("--proposal-frozen-policy", type=Path)
    parser.add_argument(
        "--continuation-frozen-policy", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--causal-lookback-m", type=float, action="append",
        help="predeclared causal lookback; repeat to replace the default grid",
    )
    parser.add_argument(
        "--strength", type=float, action="append",
        help="predeclared normalized action perturbation strength; repeat to replace the source grid",
    )
    args = parser.parse_args()
    result = revise_frontier_plan_for_resolution_cells(
        source_plan=args.source_plan,
        source_tube=args.source_tube,
        capability_geometry_summary=args.capability_geometry_summary,
        nominal_centerline=args.nominal_centerline,
        output=args.output,
        max_parent_cells_per_phase=args.max_parent_cells_per_phase,
        proposal_frozen_policy=args.proposal_frozen_policy,
        continuation_frozen_policies=tuple(args.continuation_frozen_policy),
        causal_lookbacks_m=(
            tuple(args.causal_lookback_m)
            if args.causal_lookback_m
            else (0.1, 0.2, 0.3)
        ),
        strengths=(tuple(args.strength) if args.strength else None),
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
