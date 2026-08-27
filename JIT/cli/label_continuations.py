#!/usr/bin/env python3
"""Generate policy-bound continuation labels for formal feasibility data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.continuation_labels import (
    DEFAULT_TEST_SEEDS,
    DEFAULT_TRAIN_SEEDS,
    DEFAULT_VALIDATION_SEEDS,
    label_downstream_continuations,
)
from jit_dvgc.upstream_labels import label_upstream_continuations


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=("up", "down"), required=True)
    p.add_argument("--frozen-manifest", type=Path, required=True)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--branches", type=int, default=1)
    p.add_argument("--max-ticks", type=int)
    p.add_argument("--protocol-seed", type=int)
    p.add_argument(
        "--train-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_TRAIN_SEEDS),
    )
    p.add_argument(
        "--validation-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_VALIDATION_SEEDS),
    )
    p.add_argument(
        "--test-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_TEST_SEEDS),
    )
    p.add_argument("--stochastic-policy", action="store_true")
    args = p.parse_args()

    common = dict(
        branches=args.branches,
        stochastic_policy=args.stochastic_policy,
        train_seeds=tuple(args.train_seeds),
        validation_seeds=tuple(args.validation_seeds),
        test_seeds=tuple(args.test_seeds),
    )
    if args.target == "down":
        report = label_downstream_continuations(
            args.frozen_manifest,
            args.catalog,
            args.output_dir,
            max_ticks=100 if args.max_ticks is None else args.max_ticks,
            protocol_seed=(
                820301 if args.protocol_seed is None else args.protocol_seed
            ),
            **common,
        )
    else:
        report = label_upstream_continuations(
            args.frozen_manifest,
            args.catalog,
            args.output_dir,
            max_ticks=400 if args.max_ticks is None else args.max_ticks,
            protocol_seed=(
                820302 if args.protocol_seed is None else args.protocol_seed
            ),
            **common,
        )

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
