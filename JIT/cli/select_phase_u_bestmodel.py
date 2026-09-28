#!/usr/bin/env python3
"""Select best Phase U saved checkpoint from its complete fixed panels."""

import argparse
import json
from pathlib import Path

from jit_dvgc.phase_u_bestmodel import update_best_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = update_best_model(args.run_dir)
    if result is None:
        parser.error("no complete paired fixed evaluation panel is available")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
