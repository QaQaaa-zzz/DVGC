#!/usr/bin/env python3
"""Build the task-semantic Jump-Tube view over a physical Tube analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.analysis.jump_tube_view import build_jump_tube_view


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capability-geometry-summary", type=Path, required=True)
    parser.add_argument("--nominal-centerline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-capability-geometry-summary", type=Path)
    args = parser.parse_args()
    result = build_jump_tube_view(
        capability_geometry_summary=args.capability_geometry_summary,
        nominal_centerline=args.nominal_centerline,
        output_dir=args.output_dir,
        source_capability_geometry_summary=args.source_capability_geometry_summary,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
