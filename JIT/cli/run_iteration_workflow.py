#!/usr/bin/env python3
"""Plan or explicitly execute a resumable JIT envelope-iteration workflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.workflow import run_workflow


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="execute pending stages; without this flag only print the resolved plan",
    )
    args = parser.parse_args()
    result = run_workflow(args.config, execute=args.execute)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
