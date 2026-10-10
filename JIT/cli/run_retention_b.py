#!/usr/bin/env python3
import argparse
from pathlib import Path
p=argparse.ArgumentParser(description='One finite BC+keep B arm; never PPO.')
s=p.add_subparsers(dest='command',required=True)
a=s.add_parser('prepare');a.add_argument('--output',type=Path,required=True);a.add_argument('--repository',type=Path,required=True);a.add_argument('--previous-zero',type=Path)
for name in ('run','audit','dry-run','report','worker-reset','worker-B'):
 a=s.add_parser(name);a.add_argument('--plan',type=Path,required=True)
 if name=='run':a.add_argument('--execute',action='store_true',required=True)
a=p.parse_args()
from jit_dvgc import retention_b
if a.command=='prepare':print(retention_b.prepare(a.output,a.repository,a.previous_zero))
elif a.command=='run':retention_b.run(a.plan)
elif a.command=='audit':print(retention_b.audit(a.plan))
elif a.command=='dry-run':print(retention_b.audit(a.plan)['budget'])
elif a.command=='report':print(retention_b.report(a.plan))
elif a.command=='worker-reset':retention_b.reset_worker(a.plan)
else:retention_b.B_worker(a.plan)
