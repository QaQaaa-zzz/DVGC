#!/usr/bin/env python3
"""Nine real pi_1 positive-knee trajectories near the observed TRAIN gap; no PPO."""
import argparse
from pathlib import Path
from jit_dvgc.boundary_refinement import run, DEFAULT_OUTPUT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',default='0')
    parser.add_argument('--previous',type=Path)
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--budget',type=int,default=2_000_000)
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[2]
    result=run(repo,args.output_dir or repo/DEFAULT_OUTPUT, previous=args.previous,
               baseline=args.baseline,gpu=args.gpu,budget=args.budget)
    return 0 if result['status']=='completed' else 2


if __name__=='__main__':raise SystemExit(main())
