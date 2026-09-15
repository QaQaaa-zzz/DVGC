#!/usr/bin/env python3
"""Run the finite short-pulse loop or one declared runtime stage."""
import argparse
import json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--mode',choices=['loop','baseline','collect','evaluate','update','seed_support'],required=True)
p.add_argument('--spec',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.mode=='seed_support':
    from jit_dvgc.current_policy_iteration import seed_support
    seed_support(json.loads(a.spec.read_text()),a.output)
elif a.mode=='loop':
    from jit_dvgc.pulse_exploration import run
    run(a.spec,a.output)
else:
    from jit_dvgc import pulse_exploration_runtime as runtime
    getattr(runtime,a.mode)(json.loads(a.spec.read_text()),a.output)
