#!/usr/bin/env python3
"""Stable CLI for the implemented Phase U engineering smoke only."""

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
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.phase != "propulsion_ascent":
        parser.error("only propulsion_ascent is implemented")
    if not args.smoke:
        parser.error("--smoke is required; formal training is not implemented")
    report = run_phase_u_smoke(args.config, args.run_id)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
