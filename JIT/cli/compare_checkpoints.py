#!/usr/bin/env python3
"""Predeclare or execute one bounded frozen-checkpoint exploration comparison."""
import argparse
import json
from pathlib import Path
from jit_dvgc.checkpoint_comparison import prepare,run
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--spec',type=Path);p.add_argument('--gpu',default='0')
a=p.parse_args();result=prepare(a.spec,a.output) if a.spec else run(a.output,gpu=a.gpu)
print(json.dumps({k:v for k,v in result.items() if k not in ('analysis','source_files','input_files')},indent=2))
raise SystemExit(0 if result.get('status','completed')=='completed' else 2)
