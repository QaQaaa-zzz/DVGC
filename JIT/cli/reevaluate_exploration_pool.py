#!/usr/bin/env python3
"""Evaluate a bounded declared TRAIN candidate-pool selection under frozen policies."""
import argparse
from pathlib import Path
from jit_dvgc.exploration_reevaluation import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--order', nargs='+', required=True)
    parser.add_argument('--horizon', type=int, required=True)
    parser.add_argument('--budget', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--round-index', type=int, required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--keys', nargs='+', help='Explicit canonical keys; includes witnessed entries if requested')
    args = parser.parse_args()
    run(args.pool, args.bank, order=args.order, horizon=args.horizon, budget=args.budget,
        output=args.output, round_index=args.round_index, limit=args.limit, keys=args.keys)


if __name__ == '__main__':
    main()
