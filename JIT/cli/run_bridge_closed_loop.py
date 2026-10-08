"""Prepare or execute a bounded experimental Actor/E/G continuation."""
import argparse
from jit_dvgc.generative_bridge.closed_loop import prepare_closed_loop,prepare_continuation,run_closed_loop
from jit_dvgc.generative_bridge.production import read
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--plan');p.add_argument('--previous');p.add_argument('--output');p.add_argument('--repository')
p.add_argument('--neighborhood',action='store_true')
p.add_argument('--continue-from');p.add_argument('--rounds',type=int)
a=p.parse_args()
if a.plan:run_closed_loop(read(a.plan))
elif a.continue_from:print(prepare_continuation(a.continue_from,a.output,a.repository,rounds=a.rounds)['output'])
else:print(prepare_closed_loop(a.previous,a.output,a.repository,neighborhood=a.neighborhood)['output'])
