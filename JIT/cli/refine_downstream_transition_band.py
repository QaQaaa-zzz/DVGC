#!/usr/bin/env python3
"""Refine the downstream TRAIN transition band on the contiguous 17..32 grid."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax

from jit_dvgc.downstream_transition_refinement import (
    load_downstream_refinement_config,
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
    args = parser.parse_args()

    config = load_downstream_refinement_config(args.config)
    if args.audit_only:
        print(
            json.dumps(
                {
                    "status": "config_valid",
                    "schema": config["schema"],
                    "iteration": config["iteration"],
                    "output_dir": config["output_dir"],
                    "duration_grid": config["fixed_acquisition"]["duration_grid"],
                    "terminal_clipping": config["fixed_acquisition"]["terminal_clipping"],
                    "readiness": config["readiness"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if jax.default_backend() != "gpu":
        raise RuntimeError("downstream transition-band refinement requires visible JAX GPU")
    report = search_downstream_transition_refinement(args.config, resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
