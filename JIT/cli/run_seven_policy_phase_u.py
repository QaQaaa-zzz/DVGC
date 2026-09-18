#!/usr/bin/env python3
"""Stable entry point for the finite seven-policy Phase U development experiment."""
import argparse
import json
from pathlib import Path

from jit_dvgc.seven_policy_phase_u import prepare_experiment, run_experiment, report_experiment, train_arm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--sources', type=Path, required=True)
    prepare.add_argument('--historical-config', type=Path, required=True)
    prepare.add_argument('--previous', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True)
    prepare.add_argument('--training-seeds', type=int, nargs=3, required=True)
    prepare.add_argument('--condition-seeds', type=int, nargs=4, required=True)
    prepare.add_argument('--python')
    for name in ('run', 'report', '_train-arm'):
        child = commands.add_parser(name)
        child.add_argument('--spec', type=Path, required=True)
        if name == '_train-arm':
            child.add_argument('--arm', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare_experiment(args.sources, args.historical_config, args.previous, args.output,
            training_seeds=args.training_seeds, condition_seeds=args.condition_seeds, python=args.python)
    elif args.command == '_train-arm':
        result = train_arm(args.spec, args.arm)
    elif args.command == 'run':
        result = run_experiment(args.spec)
    else:
        result = report_experiment(args.spec)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
