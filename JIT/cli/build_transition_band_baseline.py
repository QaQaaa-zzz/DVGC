#!/usr/bin/env python3
"""Build an accumulated completed-shell TRAIN baseline for automated search."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.transition_band_baseline import (
    build_transition_band_baseline,
    load_transition_band_baseline_config,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="validate the baseline declaration without writing an artifact",
    )
    args = parser.parse_args()
    config = load_transition_band_baseline_config(args.config)
    if args.audit_only:
        print(
            json.dumps(
                {
                    "status": "config_valid",
                    "schema": config["schema"],
                    "output_dir": config["output_dir"],
                    "expected_protocol_sha256": config["expected_protocol_sha256"],
                    "expected_output": config["expected_output"],
                    "source_names": [row["name"] for row in config["protocol"]["sources"]],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if Path(config["output_dir"]).exists():
        parser.error(f"baseline output already exists: {config['output_dir']}")
    report = build_transition_band_baseline(args.config)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
