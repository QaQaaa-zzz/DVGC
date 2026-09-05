#!/usr/bin/env python3
"""Project a Soft Tube into the resolution-aware physical capability space."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.analysis.capability_tube import build_capability_tube_geometry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tube", type=Path, required=True)
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("JIT/configs/phase_u_continuation_smoke.json"),
        help="authoritative JIT config used only to resolve the fixed XML/model indices",
    )
    parser.add_argument("--source-tube", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_capability_tube_geometry(
        tube=args.tube,
        model_config=args.model_config,
        source_tube=args.source_tube,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
