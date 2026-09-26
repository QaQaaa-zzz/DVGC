#!/usr/bin/env python3
"""Retry publication of an existing compact bundle; no experiment rerun."""
import argparse
from pathlib import Path
from jit_dvgc.result_publishing import publish_safely

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args()
    result=publish_safely(Path(__file__).resolve().parents[2],args.output_dir)
    raise SystemExit(0 if result['status']=='published' else 2)
