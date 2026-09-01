#!/usr/bin/env python3
"""Execute the locked Iteration-0 group-disjoint expansion validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax

from jit_dvgc.expansion_validation_protocol import audit_expansion_validation_protocol
from jit_dvgc.expansion_validation_runtime import execute_expansion_validation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="run the full zero-interaction artifact/leakage audit and exit",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "reuse only a completed acquisition under the exact runtime protocol and "
            "resume sequential validation labels"
        ),
    )
    args = parser.parse_args()

    if args.audit_only:
        print(json.dumps(audit_expansion_validation_protocol(args.config), indent=2, sort_keys=True))
        return 0

    if jax.default_backend() != "gpu":
        raise RuntimeError("expansion validation runtime requires the visible JAX GPU")
    report = execute_expansion_validation(args.config, resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
