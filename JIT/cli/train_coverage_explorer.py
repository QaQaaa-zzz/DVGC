#!/usr/bin/env python3
"""Run one bounded privileged PPO explorer against its own frozen-pi coverage."""
from pathlib import Path
import argparse
from jit_dvgc.exploration_training import run
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();run(a.spec,a.output)
