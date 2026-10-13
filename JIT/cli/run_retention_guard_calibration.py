#!/usr/bin/env python3
"""Execute only a separately reserved fixed20-attempt calibration and GPU micro."""
import argparse
import json
from jit_dvgc.retention_guard_calibration import run

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-plan',required=True)
    parser.add_argument('--authorization',required=True)
    parser.add_argument('--execute',action='store_true',required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.stage_plan,args.authorization),indent=2,allow_nan=False))
