#!/usr/bin/env python3
"""Analyze one completed paired gate as envelope progression + policy realization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.analysis.capability_progression import analyze_capability_progression_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--retrospective",
        action="store_true",
        help=(
            "mark analysis as post-candidate method reinterpretation; such an artifact "
            "may describe evidence but may not formally select the candidate policy"
        ),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"capability progression output already exists: {args.output}")
    result = analyze_capability_progression_file(
        args.gate_summary,
        retrospective=args.retrospective,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
