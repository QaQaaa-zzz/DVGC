#!/usr/bin/env python3
"""Prepare a v1.1 contract or run an explicitly bound, finite production pilot."""
import argparse
import json
from pathlib import Path
import yaml
from jit_dvgc.generative_bridge.reporting import prepare
from jit_dvgc.generative_bridge.protocol import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['audit','prepare','bind','run-pilot','worker'])
    p.add_argument('--spec',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--pointer',type=Path)
    p.add_argument('--source-round',type=Path)
    p.add_argument('--repository',type=Path)
    p.add_argument('--max-interactions',type=int)
    p.add_argument('--max-supervised-updates',type=int)
    p.add_argument('--max-wall-seconds',type=float)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--retry-generator',action='store_true')
    args=p.parse_args()
    if args.command!='worker':
        import os
        os.environ['JAX_PLATFORMS']='cpu'
    if args.command=='bind':
        if any(getattr(args,k) is None for k in ('pointer','source_round','output','repository','max_interactions','max_supervised_updates','max_wall_seconds')):
            p.error('bind requires explicit source pointer/round, repository, output and all budgets')
        from jit_dvgc.generative_bridge.production import prepare_bound_run
        report=prepare_bound_run(pointer=args.pointer,source_round=args.source_round,output=args.output,
            repository=args.repository,max_interactions=args.max_interactions,
            max_supervised_updates=args.max_supervised_updates,max_wall_seconds=args.max_wall_seconds)
        print(json.dumps({'status':'bound_not_started','path':str(args.output/'production.json'),'budgets':report['budgets']},indent=2));return
    if args.spec is None:p.error('--spec required')
    spec=yaml.safe_load(args.spec.read_text())
    if args.command=='worker':
        from jit_dvgc.generative_bridge.worker import run_generator
        run_generator(spec);return
    if args.command=='run-pilot':
        if not args.execute or spec.get('schema')!='jit_bridge_bound_production_v1_1' or spec.get('execute') is not True:
            p.error('run-pilot requires --execute and a source-bound production manifest')
        from jit_dvgc.generative_bridge.production import ProductionRunner,start_notifications
        start_notifications(spec)
        runner=ProductionRunner(spec);runner.retry_generator=args.retry_generator
        runner.run();return
    if args.execute:p.error('audit/prepare never execute; use a bound run-pilot')
    if args.output is None:p.error('--output required')
    report=prepare(spec);args.output.mkdir(parents=True,exist_ok=False)
    atomic_json(args.output/'prepare_report.json',report)
    atomic_json(args.output/'cost_ledger.json',report['cost_ledger'])
    atomic_json(args.output/'comparison_plan.json',report['comparison_arms'])
    print(json.dumps({'status':report['status'],'execute':False,'output':str(args.output),'missing':report['missing']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
