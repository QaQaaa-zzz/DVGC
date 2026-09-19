#!/usr/bin/env python3
"""Validate local snapshots and prepare witnessed TRAIN support; no simulation/PPO."""
import os
os.environ['JAX_PLATFORMS']='cpu'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
from pathlib import Path
from jit_dvgc.complementary_support import run,DEFAULT_SOURCE,DEFAULT_OUTPUT


def main():
    repo=Path(__file__).resolve().parents[2]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=repo/DEFAULT_SOURCE)
    parser.add_argument('--output-dir',type=Path,default=repo/DEFAULT_OUTPUT)
    args=parser.parse_args()
    return 0 if run(args.source,args.output_dir)['status']=='completed' else 2


if __name__=='__main__':raise SystemExit(main())
