#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.c1_engineering_selection import finalize_c1_engineering_selection


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize the exact user-authorized Iteration-1 64x64 C1 upstream engineering selection, "
            "then fit/calibrate downstream with the same architecture."
        )
    )
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--selected-upstream-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = finalize_c1_engineering_selection(
        train_root=args.train_root,
        calibration_root=args.calibration_root,
        selected_upstream_root=args.selected_upstream_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
