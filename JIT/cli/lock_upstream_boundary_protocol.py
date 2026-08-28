#!/usr/bin/env python3
"""Lock the accepted TRAIN V_up boundary recipe before validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.upstream_boundary_lock import write_boundary_lock


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-protocol", type=Path, required=True)
    parser.add_argument("--train-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = write_boundary_lock(args.train_protocol, args.train_analysis, args.output)
    print(json.dumps(lock, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
