#!/usr/bin/env python3
"""Label a TRAIN-only reachable V_up boundary bank."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.continuation_labels import DEFAULT_TRAIN_SEEDS
from jit_dvgc.upstream_boundary_labels import label_train_boundary_continuations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-ticks", type=int, default=400)
    parser.add_argument("--protocol-seed", type=int, default=820402)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=list(DEFAULT_TRAIN_SEEDS))
    args = parser.parse_args()
    report = label_train_boundary_continuations(
        args.frozen_manifest,
        args.catalog,
        args.output_dir,
        max_ticks=args.max_ticks,
        protocol_seed=args.protocol_seed,
        train_seeds=tuple(args.train_seeds),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
