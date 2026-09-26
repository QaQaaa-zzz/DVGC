#!/usr/bin/env python3
"""Run four frozen forward proposers and compare observed Tube discovery."""
import argparse
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',default='0')
    p.add_argument('--baseline',type=Path)
    p.add_argument('--previous-pilot',type=Path)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--budget-per-proposer',type=int,default=2_000_000)
    a=p.parse_args()
    from jit_dvgc.discovery_comparison import run,DEFAULT_OUTPUT
    repo=Path(__file__).resolve().parents[2]
    result=run(repo,a.output_dir or repo/DEFAULT_OUTPUT,baseline=a.baseline,
               gpu=a.gpu,budget_per_proposer=a.budget_per_proposer,previous_pilot=a.previous_pilot)
    return 0 if result['status']=='completed' else 2

if __name__=='__main__':raise SystemExit(main())
