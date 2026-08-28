#!/usr/bin/env python3
"""Run the bounded TRAIN-only unified checkpoint panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.unified_diagnostic import run_unified_fixed_panel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-phase", type=int, default=8)
    args = parser.parse_args()
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
