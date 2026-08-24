#!/usr/bin/env python3
"""Stable CLI for implemented Propulsion-Ascent smoke and formal training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.ppo import run_phase_u_smoke


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--formal", action="store_true")
    parser.add_argument("--restore-checkpoint", type=Path)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.phase != "propulsion_ascent":
        parser.error("only propulsion_ascent is implemented")
    if args.restore_checkpoint is not None and not args.formal:
        parser.error("--restore-checkpoint is only valid with --formal")
    if args.smoke:
        report = run_phase_u_smoke(args.config, args.run_id)
    else:
        from jit_dvgc.formal_training import run_phase_u_formal

        report = run_phase_u_formal(
            args.config,
            args.run_id,
            restore_checkpoint=args.restore_checkpoint,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
