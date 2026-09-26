#!/usr/bin/env python3
"""Reuse pi_4 observations and run its frozen pi_2 proposer control."""
import argparse
import os
from pathlib import Path
os.environ['JAX_PLATFORMS']='cpu'
from jit_dvgc.paired_discovery import run,DEFAULT_OUTPUT

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',default='0');p.add_argument('--source',type=Path);p.add_argument('--output-dir',type=Path)
    a=p.parse_args();repo=Path(__file__).resolve().parents[2]
    r=run(repo,a.output_dir or repo/DEFAULT_OUTPUT,a.source,a.gpu)
    raise SystemExit(0 if r['status']=='completed' else 2)
