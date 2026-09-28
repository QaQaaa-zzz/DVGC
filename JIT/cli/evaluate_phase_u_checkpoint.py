#!/usr/bin/env python3
"""Evaluate one saved Phase U checkpoint panel in an isolated GPU process."""

import argparse
import json
from pathlib import Path

from jit_dvgc.phase_u_offline_evaluation import evaluate_saved_phase_u_panel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--panel", choices=("natural", "airborne_rsi"), required=True)
    args = parser.parse_args()
    count = evaluate_saved_phase_u_panel(
        args.config, args.run_dir, args.step, args.panel
    )
    print(json.dumps({"step": args.step, "panel": args.panel, "environment_transitions": count}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
