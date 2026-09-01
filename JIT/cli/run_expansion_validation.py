#!/usr/bin/env python3
"""Execute locked Iteration-0 continuation/expansion validation protocols."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax

from jit_dvgc.expansion_validation_runtime import execute_expansion_validation
from jit_dvgc.expansion_validation_runtime_preflight import (
    audit_expansion_validation_runtime_preflight,
)
from jit_dvgc.fresh_shared_continuation_validation import (
    CONFIG_SCHEMA as FRESH_SHARED_CONFIG_SCHEMA,
    audit_fresh_shared_validation_preflight,
    execute_fresh_shared_validation,
)


def _schema(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("expansion validation config must be a JSON object")
    return str(payload.get("schema", ""))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="run the zero-interaction artifact/protocol preflight and exit",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume only under the exact already-written runtime protocol",
    )
    args = parser.parse_args()

    fresh = _schema(args.config) == FRESH_SHARED_CONFIG_SCHEMA
    preflight = (
        audit_fresh_shared_validation_preflight(args.config)
        if fresh
        else audit_expansion_validation_runtime_preflight(args.config)
    )
    if args.audit_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0
    expected_status = (
        "fresh_validation_preflight_ready" if fresh else "runtime_preflight_ready"
    )
    if preflight.get("status") != expected_status:
        raise RuntimeError("expansion validation runtime preflight did not close")

    if jax.default_backend() != "gpu":
        raise RuntimeError("expansion validation runtime requires the visible JAX GPU")
    report = (
        execute_fresh_shared_validation(args.config, resume=args.resume)
        if fresh
        else execute_expansion_validation(args.config, resume=args.resume)
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
