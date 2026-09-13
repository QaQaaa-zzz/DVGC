#!/usr/bin/env python3
"""Fit an explicit witnessed TRAIN export; no simulation, PPO, or campaign launch.

Inputs: residual_warmstart_dataset_v1 JSON referencing immutable absolute-path
residual_action_witnesses_v1 sources. Historical snapshots alone cannot supply
missing action conditioning observations. Use the module's documented contract;
never synthesize labels from unknown suffixes. Output is a frozen supervised
warm-start artifact, not evidence of improved learned exploration.
"""
import argparse
import json
from pathlib import Path

from jit_dvgc.residual_exploration import fit, save_artifact


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,required=True)
    parser.add_argument('--max-steps',type=int,required=True,
                        help='Explicit optimizer budget; environmental interactions are zero')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--learning-rate',type=float,default=.001)
    args=parser.parse_args(argv)
    if args.output.exists():
        parser.error('output already exists; preserve immutable explorer artifact')
    dataset=json.loads(args.dataset.read_text())
    variables,report=fit(dataset,steps=args.steps,max_steps=args.max_steps,
                         seed=args.seed,learning_rate=args.learning_rate)
    identity=save_artifact(args.output,variables,dataset,report)
    print(json.dumps(dict(explorer_sha256=identity,output=str(args.output),**report),indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
