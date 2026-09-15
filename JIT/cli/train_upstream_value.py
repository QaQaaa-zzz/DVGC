#!/usr/bin/env python3
"""Train the first-pass supervised V_up from TRAIN and held-out validation labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.upstream_value import train_upstream_value_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nominal-labels", type=Path, required=True)
    parser.add_argument("--boundary-train-labels", type=Path, required=True)
    parser.add_argument("--boundary-validation-labels", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=830001)
    args = parser.parse_args()
    report = train_upstream_value_model(
        args.nominal_labels,
        args.boundary_train_labels,
        args.boundary_validation_labels,
        args.lock,
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
