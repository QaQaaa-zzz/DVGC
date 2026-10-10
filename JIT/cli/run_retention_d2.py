#!/usr/bin/env python3
import argparse
from pathlib import Path
p=argparse.ArgumentParser(description='Prepare D2 or execute one bounded engineering stage; no formal arm launcher.')
s=p.add_subparsers(dest='command',required=True)
a=s.add_parser('prepare');a.add_argument('--evidence',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--repository',type=Path,required=True)
for name in ('anchors','micro'):
 a=s.add_parser(name);a.add_argument('--plan',type=Path,required=True);a.add_argument('--execute',action='store_true',required=True)
a=p.parse_args()
from jit_dvgc import retention_d2
if a.command=='prepare':print(retention_d2.prepare(a.evidence,a.output,a.repository))
else:getattr(retention_d2,a.command)(a.plan)
