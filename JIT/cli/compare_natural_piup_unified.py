#!/usr/bin/env python3
"""Compare frozen pi_up_star and pi_unified from the same natural reset."""
from __future__ import annotations

import argparse
from pathlib import Path

from jit_dvgc.natural_start_expert_compare import run_natural_start_expert_compare


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-config", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--round0-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_natural_start_expert_compare(
        formal_config=args.formal_config,
        frozen_manifest=args.frozen_manifest,
        round0_dir=args.round0_dir,
        output_dir=args.output_dir,
    )
    print(args.output_dir / "report.json")
    print(report["classification"])
    print(report["round1_recommendation"])


if __name__ == "__main__":
    main()
