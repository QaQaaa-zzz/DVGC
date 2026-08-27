#!/usr/bin/env python3
"""Generate policy-bound continuation labels for formal feasibility data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.continuation_labels import label_downstream_continuations


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=("down",), default="down")
    p.add_argument("--frozen-manifest", type=Path, required=True)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--branches", type=int, default=1)
    p.add_argument("--max-ticks", type=int, default=100)
    p.add_argument("--protocol-seed", type=int, default=820301)
    p.add_argument("--split-seed", type=int, default=820301)
    p.add_argument("--stochastic-policy", action="store_true")
    args = p.parse_args()
    report = label_downstream_continuations(
        args.frozen_manifest,
        args.catalog,
        args.output_dir,
        branches=args.branches,
        max_ticks=args.max_ticks,
        protocol_seed=args.protocol_seed,
        stochastic_policy=args.stochastic_policy,
        split_seed=args.split_seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
