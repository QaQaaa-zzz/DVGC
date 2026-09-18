#!/usr/bin/env python3
"""Render diagnostic liftoff-aligned pulse witnesses without simulation."""
import argparse
from jit_dvgc.analysis.liftoff_alignment import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lineage', required=True)
    parser.add_argument('--output', required=True, help='New output directory; never overwrite originals')
    parser.add_argument('--per-round', type=int, default=24, help='Successful candidates per round; 0 = all')
    parser.add_argument('--clearance', type=float, default=.01, help='Diagnostic double-wheel clearance in metres')
    parser.add_argument('--hold-frames', type=int, default=3)
    args = parser.parse_args()
    result = build(args.lineage, args.output, per_round=args.per_round,
                   threshold=args.clearance, hold=args.hold_frames)
    print(f'Aligned {result["aligned"]}/{result["selected"]}; output: {args.output}')


if __name__ == '__main__':
    main()
