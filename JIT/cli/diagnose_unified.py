#!/usr/bin/env python3
"""Run bounded TRAIN-only unified diagnostics and paired iteration gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.analysis import run_paired_policy_gate, run_unified_fixed_panel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gate-config",
        type=Path,
        help="run the predeclared paired core-preservation/boundary-gain gate",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--samples-per-phase", type=int, default=8)
    args = parser.parse_args()

    if args.gate_config is not None:
        if any(value is not None for value in (args.config, args.checkpoint, args.output_dir)):
            parser.error("--gate-config cannot be combined with panel --config/--checkpoint/--output-dir")
        report = run_paired_policy_gate(args.gate_config)
    else:
        if args.config is None or args.checkpoint is None or args.output_dir is None:
            parser.error("fixed-panel mode requires --config, --checkpoint, and --output-dir")
        report = run_unified_fixed_panel(
            args.config,
            args.checkpoint,
            args.output_dir,
            samples_per_phase=args.samples_per_phase,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
