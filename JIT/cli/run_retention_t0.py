#!/usr/bin/env python3
"""Saved-data CPU diagnostic; no training or simulation entrypoint."""
import argparse
from pathlib import Path

from jit_dvgc.retention_t0_diagnostics import run

parser = argparse.ArgumentParser(description=__doc__)
for name in ('control', 'intervention', 'output'):
    parser.add_argument('--' + name, type=Path, required=True)
args = parser.parse_args()
result = run(args.control, args.intervention, args.output)
raise SystemExit(0 if result['t1_allowed'] else 2)
