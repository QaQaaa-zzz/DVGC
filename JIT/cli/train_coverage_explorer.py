#!/usr/bin/env python3
"""Train bounded residual disturbances on frozen pi using candidate suffix novelty."""
from pathlib import Path
import argparse
from jit_dvgc.exploration_training import run
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();run(a.spec,a.output)
