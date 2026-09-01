#!/usr/bin/env python3
"""Run a schema-selected fresh single-policy Tube-RSI PPO job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.training import (
    FORMAL_SCHEMA,
    PILOT_SCHEMA,
    read_json,
    run_unified_formal,
    run_unified_pilot,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    schema = read_json(args.config).get("schema")
    if schema == PILOT_SCHEMA:
        result = run_unified_pilot(args.config, args.run_id)
    elif schema == FORMAL_SCHEMA:
        result = run_unified_formal(args.config, args.run_id)
    else:
        raise ValueError(f"unsupported unified training schema: {schema}")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
