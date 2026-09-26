#!/usr/bin/env python3
"""Prepare isolated fixed-policy learned/random pulse arms; no simulation."""
import argparse
import json
import os
from pathlib import Path

os.environ['JAX_PLATFORMS'] = 'cpu'

from jit_dvgc.pulse_mechanism import prepare_mechanism


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial-frozen-policy', type=Path, required=True)
    parser.add_argument('--task-template-bank', type=Path, required=True,
                        help='Geometry/start identities only; no policies, labels or support are imported')
    parser.add_argument('--jump-start-state-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--num-envs', type=int, default=128)
    parser.add_argument('--rounds', type=int, default=24)
    parser.add_argument('--amplitudes', type=float, nargs='+', default=[.10, .15])
    parser.add_argument('--max-actual-interactions', type=int, default=2_500_000)
    parser.add_argument('--recovery-seconds', type=float, default=.5)
    parser.add_argument('--python', default='/home/qy/mujoco_playground/.venv/bin/python')
    args = parser.parse_args()
    plan = prepare_mechanism(**vars(args))
    print(json.dumps({'status': 'prepared_not_launched',
                      'plan': str(args.output.resolve() / 'plan.json'),
                      'maximum_interactions': plan['max_interactions']}, indent=2))


if __name__ == '__main__':
    main()
