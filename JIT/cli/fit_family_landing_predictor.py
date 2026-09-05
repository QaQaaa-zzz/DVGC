#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.family_landing_predictor import (
    evaluate_locked_forward_scores,
    fit_family_landing_predictor,
    lock_forward_predictor_scores,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path)
    parser.add_argument("--calibration-root", type=Path)
    parser.add_argument("--acceptance-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--lock-forward-catalog", type=Path)
    parser.add_argument("--role", choices=("train", "calibration", "acceptance"))
    parser.add_argument("--field", type=Path)
    parser.add_argument("--field-manifest", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--audit-locked-scores", type=Path)
    parser.add_argument("--role-root", type=Path)
    args = parser.parse_args()
    if args.audit_locked_scores is not None:
        required = {
            "role": args.role,
            "role_root": args.role_root,
            "output_file": args.output_file,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error("forward audit requires: " + ", ".join(missing))
        result = evaluate_locked_forward_scores(
            scores_path=args.audit_locked_scores,
            role_root=args.role_root,
            role=args.role,
            output_path=args.output_file,
        )
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.lock_forward_catalog is not None:
        required = {
            "role": args.role,
            "field": args.field,
            "field_manifest": args.field_manifest,
            "calibration": args.calibration,
            "output_file": args.output_file,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error("forward scoring requires: " + ", ".join(missing))
        result = lock_forward_predictor_scores(
            catalog_path=args.lock_forward_catalog,
            role=args.role,
            field_path=args.field,
            field_manifest_path=args.field_manifest,
            calibration_path=args.calibration,
            output_path=args.output_file,
        )
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    required = {
        "train_root": args.train_root,
        "calibration_root": args.calibration_root,
        "acceptance_root": args.acceptance_root,
        "output_dir": args.output_dir,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error("predictor fitting requires: " + ", ".join(missing))
    result = fit_family_landing_predictor(
        train_root=args.train_root,
        calibration_root=args.calibration_root,
        acceptance_root=args.acceptance_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
