#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.iterative_continuation_fields import MODEL_PROFILES
from jit_dvgc.iterative_weighting_compat import fit_and_calibrate_observed_cells


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit C^k fields on TRAIN and calibrate on the declared calibration rows."
    )
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-profile",
        choices=tuple(sorted(MODEL_PROFILES)),
        default="legacy_tiny_tanh",
        help=(
            "Network profile. legacy_tiny_tanh reproduces 76->8->1; "
            "standard_mlp_64x64_tanh uses 76->64->64->1; "
            "standard_mlp_128x128_tanh is the final same-data capacity escalation "
            "using 76->128->128->1. All other fit/calibration settings are unchanged."
        ),
    )
    args = parser.parse_args()
    result = fit_and_calibrate_observed_cells(
        train_root=args.train_root,
        calibration_root=args.calibration_root,
        output_dir=args.output_dir,
        model_profile=args.model_profile,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
