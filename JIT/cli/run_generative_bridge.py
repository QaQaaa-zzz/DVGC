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
    p.add_argument('command',choices=['audit','prepare','bind','run-pilot','recover','prepare-series','run-series','prepare-v12','run-v12','recover-v12','worker'])
    p.add_argument('--spec',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--previous',type=Path)
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--replay-diagnostic',type=Path)
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
    if args.command=='prepare-series':
        if any(getattr(args,k) is None for k in ('previous','output','repository')):
            p.error('prepare-series requires previous/output/repository')
        from jit_dvgc.generative_bridge.series import prepare_series
        report=prepare_series(args.previous,args.output,args.repository,rounds=args.rounds)
        print(json.dumps({'phase':'prepared','budgets':report['budgets']}));return
    if args.command=='recover-v12':
        if any(v is None for v in (args.previous,args.output,args.repository)):p.error('recover-v12 needs --previous --output --repository')
        from jit_dvgc.generative_bridge.recovery import prepare_campaign_recovery
        prepare_campaign_recovery(args.previous,args.output,args.repository,max_wall_seconds=args.max_wall_seconds or 86400)
        print(json.dumps({'phase':'prepared','output':str(args.output)}));return
    if args.command=='recover':
        if any(getattr(args,k) is None for k in ('previous','output','repository')):
            p.error('recover requires --previous, --output and --repository')
        from jit_dvgc.generative_bridge.recovery import prepare_recovery
        prepare_recovery(args.previous,args.output,args.repository,replay_diagnostic=args.replay_diagnostic)
        print(json.dumps({'phase':'prepared','output':str(args.output)}));return
    if args.spec is None:p.error('--spec required')
    spec=json.loads(args.spec.read_text()) if args.spec.suffix=='.json' else yaml.safe_load(args.spec.read_text())
    if args.command=='run-series':
        if not args.execute or spec.get('schema')!='jit_bridge_fixed_actor_series_v1':
            p.error('run-series requires declared fixed Actor series and --execute')
        from jit_dvgc.generative_bridge.series import run_series
        run_series(spec);return
    if args.command=='prepare-v12':
        from jit_dvgc.generative_bridge.campaign import prepare_campaign
        if any(v is None for v in (args.previous,args.output,args.repository)):p.error('prepare-v12 needs audit directory --previous, --output, --repository')
        result=prepare_campaign(args.spec,args.previous,args.output,args.repository)
        print(json.dumps({'phase':'prepared','path':str(args.output/'production.json')}));return
    if args.command=='run-v12':
        if not args.execute or spec.get('schema')!='jit_bridge_campaign_v1_2' or spec.get('execute') is not True:p.error('explicit source-bound v1.2 execution required')
        from jit_dvgc.generative_bridge.campaign import CampaignRunner
        from jit_dvgc.generative_bridge.production import start_notifications
        start_notifications(spec);CampaignRunner(spec).run();return
    if args.command=='worker':
        # Native faults must preserve Python thread stacks in the gated worker log.
        import faulthandler
        faulthandler.enable(all_threads=True)
        from jit_dvgc.generative_bridge.worker import run_generator,run_warmup
        if spec.get('mode')=='incremental' and spec.get('generator_update_policy')=='last_valid':
            from jit_dvgc.generative_bridge.worker_lifecycle import run_generator_process
            run_generator_process(spec)
        else:
            (run_warmup if spec.get('mode')=='warmup' else run_generator)(spec)
        return
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
