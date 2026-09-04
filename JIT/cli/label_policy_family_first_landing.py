#!/usr/bin/env python3
"""Label one causal catalog by first landing under pi_0/pi_1/pi_2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.policy_family_landing import label_policy_family_first_landing


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
    args = parser.parse_args()
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
