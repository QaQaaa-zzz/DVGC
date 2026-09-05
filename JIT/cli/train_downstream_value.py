#!/usr/bin/env python3
"""Train the first-pass supervised V_down from frozen-expert continuation labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.downstream_value import train_downstream_value_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=830002)
    args = parser.parse_args()
    report = train_downstream_value_model(
        args.labels,
        args.frozen_manifest,
        args.output_dir,
        hidden_sizes=tuple(args.hidden_sizes),
        steps=args.steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
