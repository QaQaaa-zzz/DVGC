#!/usr/bin/env python3
"""Evaluate a declared old/pending panel under source and trained checkpoints."""
import argparse
from pathlib import Path
from jit_dvgc.retention_experiment import evaluate

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--spec', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
evaluate(args.spec, args.output)
