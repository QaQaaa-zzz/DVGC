#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.iterative_tube_engineering_override import build_iterative_tube_engineering_override


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Tube2 from the exact user-authorized 64x64 C1 engineering selection."
    )
    parser.add_argument("--source-tube", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--fields-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_iterative_tube_engineering_override(
        source_tube=args.source_tube,
        train_root=args.train_root,
        fields_root=args.fields_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
