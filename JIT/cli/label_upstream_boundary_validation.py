#!/usr/bin/env python3
"""Label the locked validation V_up boundary bank."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.upstream_boundary_validation_labels import label_validation_boundary_continuations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-ticks", type=int, default=400)
    parser.add_argument("--protocol-seed", type=int, default=820405)
    args = parser.parse_args()
    report = label_validation_boundary_continuations(
        args.frozen_manifest,
        args.catalog,
        args.lock,
        args.output_dir,
        max_ticks=args.max_ticks,
        protocol_seed=args.protocol_seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
