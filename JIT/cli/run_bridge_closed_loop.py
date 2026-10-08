"""Prepare, inspect, or explicitly execute a bounded Actor/E/G continuation."""
import argparse
import json
from jit_dvgc.generative_bridge.closed_loop import prepare_closed_loop,prepare_continuation,run_closed_loop
from jit_dvgc.generative_bridge.production import read
from jit_dvgc.generative_bridge.protocol import atomic_json
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--plan');p.add_argument('--previous');p.add_argument('--output');p.add_argument('--repository')
p.add_argument('--neighborhood',action='store_true')
p.add_argument('--continue-from');p.add_argument('--rounds',type=int)
p.add_argument('--continuous',action='store_true')
p.add_argument('--profile');p.add_argument('--allow-stopped-parent',action='store_true')
mode=p.add_mutually_exclusive_group();mode.add_argument('--dry-run',action='store_true');mode.add_argument('--execute',action='store_true')
a=p.parse_args()
if a.plan:
    plan=read(a.plan)
    if a.dry_run:
        if not plan.get('profile'):p.error('dry-run requires an explicit continuation profile')
        from jit_dvgc.generative_bridge.continuation_profile import inspect_plan
        result=inspect_plan(plan);print(json.dumps(result,indent=2))
    elif plan.get('profile') and not a.execute:
        p.error('new-profile execution requires --execute; --dry-run never simulates')
    else:run_closed_loop(plan)
elif a.execute or a.dry_run:p.error('--execute/--dry-run require --plan')
elif a.continue_from:
    print(prepare_continuation(a.continue_from,a.output,a.repository,rounds=a.rounds,continuous=a.continuous,
        profile=a.profile,allow_stopped_parent=a.allow_stopped_parent)['output'])
else:
    if a.profile or a.allow_stopped_parent:p.error('profile requires --continue-from')
    print(prepare_closed_loop(a.previous,a.output,a.repository,neighborhood=a.neighborhood)['output'])
