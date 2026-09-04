#!/usr/bin/env python3
"""Revise one still-outcome-blind frontier plan using physical capability cells."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.acquisition.resolution_frontier import (
    revise_frontier_plan_for_resolution_cells,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--source-tube", type=Path, required=True)
    parser.add_argument("--capability-geometry-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-parent-cells-per-phase", type=int, default=25)
    args = parser.parse_args()
    result = revise_frontier_plan_for_resolution_cells(
        source_plan=args.source_plan,
        source_tube=args.source_tube,
        capability_geometry_summary=args.capability_geometry_summary,
        output=args.output,
        max_parent_cells_per_phase=args.max_parent_cells_per_phase,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
