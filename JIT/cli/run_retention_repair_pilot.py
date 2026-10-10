#!/usr/bin/env python3
"""Prepare, validate, explicitly run, or report a bounded retention diagnostic."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
a=sub.add_parser('prepare');a.add_argument('--stage',choices=['D0'],required=True);a.add_argument('--output',type=Path,required=True)
for name in ('audit','run','report'):
    a=sub.add_parser(name);a.add_argument('--plan',type=Path,required=True)
    if name=='run':a.add_argument('--execute',action='store_true',required=True)
    if name=='report':a.add_argument('--output',type=Path)
a=p.parse_args()
if a.command=='prepare':
    from jit_dvgc.retention_repair import prepare
    print(prepare(a.output))
elif a.command=='audit':
    from jit_dvgc.retention_repair import audit
    print(json.dumps(audit(a.plan),indent=2))
elif a.command=='run':
    from jit_dvgc.retention_repair_execution import run
    run(a.plan)
else:
    from jit_dvgc.retention_repair_report import report
    print(report(a.plan,a.output))
