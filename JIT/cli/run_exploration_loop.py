#!/usr/bin/env python3
"""Run the declared finite TRAIN exploration/learning/delayed-feedback loop."""
import argparse
import json
import os
import signal
os.environ["JAX_PLATFORMS"] = "cpu"
from pathlib import Path
from jit_dvgc.exploration_loop import run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    def terminate(signum, frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,terminate)
    result=run(args.spec,args.output)
    print(json.dumps(result,indent=2))
    if result['phase']!='completed':raise SystemExit(2)


if __name__=='__main__':
    main()
