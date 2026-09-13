#!/usr/bin/env python3
"""Repackage completed or failed results without importing GPU code or rerunning jobs."""
import argparse
from pathlib import Path
from jit_dvgc.result_bundle import bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--full", action="store_true", help="Include detailed JSON and paper figure formats")
    args = parser.parse_args()
    path = bundle(args.output_dir, full=args.full)
    print(f"Return this file: {path} ({path.stat().st_size / 1_000_000:.2f} MB)")


if __name__ == "__main__":
    main()
