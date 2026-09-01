#!/usr/bin/env python3
"""Build TRAIN-only learned Soft-Tube artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.soft_tube import SoftTubeInputs, build_soft_tube


_LEGACY_OPTIONS = (
    "frozen-experts",
    "up-model-dir",
    "down-model-dir",
    "up-nominal-labels",
    "up-nominal-catalog",
    "up-nominal-protocol",
    "up-boundary-labels",
    "up-boundary-catalog",
    "up-boundary-protocol",
    "down-labels",
    "down-catalog",
    "down-protocol",
    "output-dir",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        help="build a policy-conditioned core-retaining Tube iteration from a locked config",
    )
    for option in _LEGACY_OPTIONS:
        parser.add_argument(f"--{option}", type=Path)
    args = parser.parse_args()

    if args.config is not None:
        supplied_legacy = [
            option
            for option in _LEGACY_OPTIONS
            if getattr(args, option.replace("-", "_")) is not None
        ]
        if supplied_legacy:
            raise ValueError("--config mode cannot be mixed with legacy Soft-Tube inputs")
        from jit_dvgc.core_retaining_tube_iteration import build_core_retaining_tube

        artifact = build_core_retaining_tube(args.config)
    else:
        missing = [
            option
            for option in _LEGACY_OPTIONS
            if getattr(args, option.replace("-", "_")) is None
        ]
        if missing:
            parser.error(
                "legacy mode requires all Soft-Tube inputs; missing: "
                + ", ".join(f"--{name}" for name in missing)
            )
        inputs = SoftTubeInputs(
            frozen_experts=args.frozen_experts,
            up_model_dir=args.up_model_dir,
            down_model_dir=args.down_model_dir,
            up_nominal_labels=args.up_nominal_labels,
            up_nominal_catalog=args.up_nominal_catalog,
            up_nominal_protocol=args.up_nominal_protocol,
            up_boundary_labels=args.up_boundary_labels,
            up_boundary_catalog=args.up_boundary_catalog,
            up_boundary_protocol=args.up_boundary_protocol,
            down_labels=args.down_labels,
            down_catalog=args.down_catalog,
            down_protocol=args.down_protocol,
        )
        artifact = build_soft_tube(inputs, args.output_dir)

    print(
        json.dumps(
            {
                "output_dir": str(artifact.root),
                "manifest": artifact.manifest,
                "diagnostics": artifact.diagnostics,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
