#!/usr/bin/env python3
"""Refine the downstream TRAIN transition band on the contiguous 17..32 grid."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax

from jit_dvgc.downstream_transition_refinement import (
    audit_downstream_transition_refinement,
    search_downstream_transition_refinement,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume only fully completed duration steps under the exact same protocol",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="validate the declaration without constructing MJX or spending interactions",
    )
    parser.add_argument(
        "--repair-source-head",
        help=(
            "with --resume, allow only the declared failed run's repository HEAD "
            "to advance to the current repair HEAD"
        ),
    )
    args = parser.parse_args()

    if args.repair_source_head is not None and not args.resume:
        parser.error("--repair-source-head requires --resume")
    if args.audit_only and args.repair_source_head is not None:
        parser.error("--audit-only does not perform repair resume")

    if args.audit_only:
        print(
            json.dumps(
                audit_downstream_transition_refinement(args.config),
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if jax.default_backend() != "gpu":
        raise RuntimeError("downstream transition-band refinement requires visible JAX GPU")
    report = search_downstream_transition_refinement(
        args.config,
        resume=args.resume,
        repair_source_head=args.repair_source_head,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
