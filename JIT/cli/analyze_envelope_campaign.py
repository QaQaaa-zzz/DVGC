#!/usr/bin/env python3
"""Rebuild verified campaign reports with zero simulation/training interactions."""
import argparse
import os
from pathlib import Path
os.environ['JAX_PLATFORMS']='cpu'
from jit_dvgc.campaign_analysis import run

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output-dir',type=Path)
    args=p.parse_args()
    output=args.output_dir or args.source.with_name(args.source.name+'_analysis_v1')
    raise SystemExit(0 if run(args.source,output)['status']=='analysis_completed' else 2)
