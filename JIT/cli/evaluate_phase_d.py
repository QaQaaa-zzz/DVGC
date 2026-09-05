#!/usr/bin/env python3
"""Frozen Phase D fixed-index evaluation panel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.phase_d_evaluation import run_phase_d_panel


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--eval-seeds", type=int, nargs="+", required=True)
    p.add_argument("--max-ticks", type=int, default=100)
    p.add_argument("--run-id", required=True)
    args = p.parse_args()
    if args.phase != "descent_recovery":
        p.error("only descent_recovery is supported")
    report = run_phase_d_panel(args.config, args.checkpoint, args.catalog, eval_seeds=tuple(args.eval_seeds), max_ticks=args.max_ticks, run_id=args.run_id)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
