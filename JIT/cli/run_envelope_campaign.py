#!/usr/bin/env python3
"""Run bounded autonomous witnessed-support learning and discovery rounds."""
import argparse
import os
os.environ["JAX_PLATFORMS"]="cpu"
from pathlib import Path
from jit_dvgc.envelope_campaign import run,DEFAULT_OUTPUT

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--all-proposers',action='store_true',help='Reuse completed pi_2/pi_4 evidence; explore all policies and train pi_5 onward')
    p.add_argument('--gpu',default='0');p.add_argument('--output-dir',type=Path)
    p.add_argument('--max-rounds',type=int,default=3);p.add_argument('--budget',type=int,default=40_000_000)
    p.add_argument('--ppo-steps',type=int,default=512_000)
    p.add_argument('--checkpoint-steps',type=int,nargs='+',help='Opt-in aligned milestones ending at --ppo-steps, e.g. 32000 64000 128000; one live trainer, no automatic extension')
    p.add_argument('--panel-samples-per-phase',type=int,default=2)
    p.add_argument('--panel-horizon',type=int,default=400)
    p.add_argument('--patience',type=int,default=2);p.add_argument('--min-gain',type=int,default=5)
    a=p.parse_args();repo=Path(__file__).resolve().parents[2]
    result=run(repo,a.output_dir or repo/DEFAULT_OUTPUT,gpu=a.gpu,max_rounds=a.max_rounds,
        budget=a.budget,ppo_steps=a.ppo_steps,patience=a.patience,min_gain=a.min_gain,all_proposers=a.all_proposers,
        checkpoints=a.checkpoint_steps,panel_samples_per_phase=a.panel_samples_per_phase,panel_horizon=a.panel_horizon)
    raise SystemExit(2 if result['status']=='engineering_error' else 0)
