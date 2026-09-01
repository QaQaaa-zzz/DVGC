#!/usr/bin/env python3
"""Run the predeclared automated TRAIN transition-band search for frozen pi_k."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax

from jit_dvgc.acquisition.transition_band import (
    load_transition_band_search_config,
    search_unified_transition_band,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume only from fully completed prior shells under the exact same protocol",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="validate the search declaration without constructing runtime or spending interactions",
    )
    args = parser.parse_args()

    config = load_transition_band_search_config(args.config)
    if args.audit_only:
        print(
            json.dumps(
                {
                    "status": "config_valid",
                    "schema": config["schema"],
                    "iteration": config["iteration"],
                    "output_dir": config["output_dir"],
                    "outer_shells": config["outer_shells"],
                    "readiness": config["readiness"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if jax.default_backend() != "gpu":
        raise RuntimeError("automated transition-band search requires the visible JAX GPU")
    report = search_unified_transition_band(args.config, resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
