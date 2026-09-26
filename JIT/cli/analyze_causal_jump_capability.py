#!/usr/bin/env python3
"""Build role-separated ground-connected Jump-Capability evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.analysis.causal_jump_capability import (
    build_causal_jump_capability_evidence,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nominal-centerline", type=Path, required=True)
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("JIT/configs/phase_u_continuation_smoke.json"),
    )
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--acceptance-root", type=Path, required=True)
    parser.add_argument("--source-causal-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_causal_jump_capability_evidence(
        nominal_centerline=args.nominal_centerline,
        model_config=args.model_config,
        train_root=args.train_root,
        calibration_root=args.calibration_root,
        acceptance_root=args.acceptance_root,
        output_dir=args.output_dir,
        source_causal_summary=args.source_causal_summary,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
