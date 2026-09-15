#!/usr/bin/env python3
"""Search low successful real trajectories; no state-height injection or PPO."""
import argparse
from pathlib import Path
from jit_dvgc.lower_boundary import run,DEFAULT_OUTPUT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',default='0')
    parser.add_argument('--source',type=Path)
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--budget',type=int,default=3_500_000)
    args=parser.parse_args();repo=Path(__file__).resolve().parents[2]
    result=run(repo,args.output_dir or repo/DEFAULT_OUTPUT,source=args.source,
        baseline=args.baseline,gpu=args.gpu,budget=args.budget)
    return 0 if result['status']=='completed' else 2


if __name__=='__main__':raise SystemExit(main())
