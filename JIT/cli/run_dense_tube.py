#!/usr/bin/env python3
"""Run a bounded real-frame 5 cm Tube pilot with measured label acceleration."""
import argparse
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',default='0')
    p.add_argument('--baseline',type=Path)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--budget',type=int,default=2_000_000)
    p.add_argument('--worker',help=argparse.SUPPRESS)
    p.add_argument('--destination',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--policy',help=argparse.SUPPRESS)
    p.add_argument('--backend',default='serial',help=argparse.SUPPRESS)
    p.add_argument('--catalog',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--shard-index',type=int,default=0,help=argparse.SUPPRESS)
    p.add_argument('--shard-count',type=int,default=1,help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.worker:
        from jit_dvgc.dense_tube_runtime import worker
        return worker(a)
    from jit_dvgc.dense_tube import run,DEFAULT_OUTPUT
    repo=Path(__file__).resolve().parents[2]
    result=run(repo,a.output_dir or repo/DEFAULT_OUTPUT,baseline=a.baseline,gpu=a.gpu,budget=a.budget)
    return 0 if result['status']=='completed' else 2

if __name__=='__main__':raise SystemExit(main())
