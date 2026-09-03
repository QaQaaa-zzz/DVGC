#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax

from jit_dvgc.iterative_acceptance_gate import lock_baseline, run_candidate_gate


def main() -> int:
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="command", required=True)

    lock = subs.add_parser("lock-baseline")
    lock.add_argument("--selected-policy", type=Path, required=True)
    lock.add_argument("--source-tube", type=Path, required=True)
    lock.add_argument("--acceptance-root", type=Path, required=True)
    lock.add_argument("--output-dir", type=Path, required=True)

    gate = subs.add_parser("run-candidate")
    gate.add_argument("--baseline-lock", type=Path, required=True)
    gate.add_argument("--candidate-frozen-policy", type=Path, required=True)
    gate.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError("iterative acceptance gate requires the visible JAX GPU")
    if args.command == "lock-baseline":
        result = lock_baseline(
            selected_policy=args.selected_policy,
            source_tube=args.source_tube,
            acceptance_root=args.acceptance_root,
            output_dir=args.output_dir,
        )
    else:
        result = run_candidate_gate(
            baseline_lock=args.baseline_lock,
            candidate_frozen_policy=args.candidate_frozen_policy,
            output_dir=args.output_dir,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
