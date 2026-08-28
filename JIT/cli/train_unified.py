#!/usr/bin/env python3
"""Run the bounded fresh single-policy Tube-RSI PPO pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.unified_training import run_unified_pilot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    result = run_unified_pilot(args.config, args.run_id)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
