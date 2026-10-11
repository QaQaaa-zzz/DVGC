#!/usr/bin/env python3
"""Prepare, audit, execute or report the explicitly bounded T1 paired pilot."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'audit', 'dry-run', 'run', 'report', 'worker'))
    for field in ('parent', 't0-gate', 'output', 'plan', 'request', 'repository'):
        parser.add_argument('--' + field, type=Path)
    parser.add_argument('--tensorboard-port', type=int, default=6032)
    args = parser.parse_args()
    required = ('parent', 't0_gate', 'output', 'repository') if args.mode == 'prepare' else (
        ('request',) if args.mode == 'worker' else ('plan',))
    for field in required:
        if getattr(args, field) is None:
            parser.error('--' + field.replace('_', '-') + ' is required for ' + args.mode)
    if args.mode != 'worker':
        os.environ['JAX_PLATFORMS'] = 'cpu'
    if args.mode == 'worker':
        from jit_dvgc.retention_t1_runtime import worker
        result = worker(args.request)
    else:
        from jit_dvgc import retention_t1
        if args.mode == 'prepare':
            result = retention_t1.prepare(args.parent, args.t0_gate, args.output,
                args.repository, tensorboard_port=args.tensorboard_port)
        else:
            result = getattr(retention_t1, args.mode.replace('-', '_'))(args.plan)
    print(json.dumps(result, indent=2, default=str, allow_nan=False))


if __name__ == '__main__':
    main()
