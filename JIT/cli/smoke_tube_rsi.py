#!/usr/bin/env python3
"""Run the bounded unified Tube-RSI restore/step smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.config import load_config
from jit_dvgc.tube import load_soft_tube, run_tube_rsi_smoke
from jit_dvgc.unified_env import UnifiedTubeRSIEnv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--up-config", type=Path, required=True)
    parser.add_argument("--down-config", type=Path, required=True)
    parser.add_argument("--soft-tube", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-phase", type=int, default=8)
    args = parser.parse_args()
    env = UnifiedTubeRSIEnv(
        load_config(args.up_config),
        load_config(args.down_config),
        load_soft_tube(args.soft_tube),
    )
    report = run_tube_rsi_smoke(
        env, args.output_dir, samples_per_phase=args.samples_per_phase
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
