#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.iterative_weighting_compat import fit_and_calibrate_observed_cells


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit fixed-architecture C^k fields on TRAIN and calibrate on disjoint calibration rows."
    )
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = fit_and_calibrate_observed_cells(
        train_root=args.train_root,
        calibration_root=args.calibration_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
