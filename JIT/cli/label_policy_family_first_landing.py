#!/usr/bin/env python3
"""Label one causal catalog by first landing under pi_0/pi_1/pi_2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.policy_family_landing import (
    label_policy_family_first_landing,
    merge_policy_family_evaluator_shards,
    run_policy_family_evaluator_shard,
)
from jit_dvgc.unified_policy_freeze import load_frozen_unified_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--acquisition-frozen-policy", type=Path, required=True)
    parser.add_argument(
        "--evaluator-frozen-policy", type=Path, action="append", required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-ticks", type=int, default=400)
    parser.add_argument("--protocol-seed", type=int, default=9_521_201)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--merge-shard-dir", type=Path, action="append")
    args = parser.parse_args()
    if args.merge_shard_dir:
        if len(args.evaluator_frozen_policy) != 1:
            parser.error("shard merge requires exactly one evaluator frozen policy")
        evaluator = load_frozen_unified_manifest(args.evaluator_frozen_policy[0])[
            "policy"
        ]
        result = merge_policy_family_evaluator_shards(
            catalog_path=args.catalog,
            shard_dirs=args.merge_shard_dir,
            output_dir=args.output_dir,
            evaluator_name=str(evaluator["name"]),
        )
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.shard_index is not None or args.shard_count is not None:
        if args.shard_index is None or args.shard_count is None:
            parser.error("evaluator shard requires both --shard-index and --shard-count")
        if len(args.evaluator_frozen_policy) != 1:
            parser.error("evaluator shard requires exactly one evaluator frozen policy")
        result = run_policy_family_evaluator_shard(
            catalog_path=args.catalog,
            acquisition_frozen_policy=args.acquisition_frozen_policy,
            evaluator_frozen_policy=args.evaluator_frozen_policy[0],
            output_dir=args.output_dir,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            max_ticks=args.max_ticks,
            protocol_seed=args.protocol_seed,
        )
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    result = label_policy_family_first_landing(
        catalog_path=args.catalog,
        acquisition_frozen_policy=args.acquisition_frozen_policy,
        evaluator_frozen_policies=args.evaluator_frozen_policy,
        output_dir=args.output_dir,
        max_ticks=args.max_ticks,
        protocol_seed=args.protocol_seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
