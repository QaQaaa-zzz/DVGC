#!/usr/bin/env python3
"""Prepare/audit/execute exactly one retention stage; report saved data only."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
a=sub.add_parser('prepare');a.add_argument('--stage',choices=['V','D1','D2'],required=True)
a.add_argument('--evidence-dir',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--repository',type=Path,required=True)
a=sub.add_parser('prepare-resume');a.add_argument('--previous',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--repository',type=Path,required=True)
a=sub.add_parser('worker');a.add_argument('--request',type=Path,required=True)
for name in ('audit','run','report'):
    a=sub.add_parser(name);a.add_argument('--plan',type=Path,required=True)
    if name=='run':a.add_argument('--execute',action='store_true',required=True)
    if name=='report':a.add_argument('--output',type=Path)
a=p.parse_args()
if a.command=='prepare':
    from jit_dvgc.retention_next import prepare
    print(prepare(a.stage,a.evidence_dir,a.output,a.repository))
elif a.command=='prepare-resume':
    from jit_dvgc.retention_next import prepare_resume
    print(prepare_resume(a.previous,a.output,a.repository))
elif a.command=='worker':
    from jit_dvgc.retention_next_runtime import worker
    worker(a.request)
elif a.command=='audit':
    from jit_dvgc.retention_next import audit
    print(json.dumps(audit(a.plan),indent=2))
elif a.command=='run':
    from jit_dvgc.retention_next_runtime import run
    run(a.plan)
else:
    from jit_dvgc.retention_next_report import report
    print(report(a.plan,a.output))
