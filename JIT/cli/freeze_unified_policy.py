#!/usr/bin/env python3
"""Freeze one completed unified checkpoint as an envelope-expansion authority."""
from __future__ import annotations

import argparse
from pathlib import Path

from jit_dvgc.training.freeze import freeze_unified_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--formal-report", type=Path)
    args = parser.parse_args()

    manifest = freeze_unified_policy(
        args.output_dir,
        config_path=args.config,
        checkpoint=args.checkpoint,
        iteration=args.iteration,
        formal_report=args.formal_report,
    )
    print(args.output_dir / "frozen_unified_policy.json")
    print(manifest["policy"]["name"])
    print(manifest["policy"]["payload_sha256"])


if __name__ == "__main__":
    main()
