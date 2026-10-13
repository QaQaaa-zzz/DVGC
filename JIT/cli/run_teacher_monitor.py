#!/usr/bin/env python3
"""Prepare/audit or physically execute one fixed nontraining DEV teacher cell."""
import argparse
import json
from jit_dvgc.generative_bridge.teacher_monitor import prepare_panel,audit_panel,evaluate_checkpoint,aggregate_reports


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='mode',required=True)
    prepare=sub.add_parser('prepare');prepare.add_argument('--source-plan',required=True)
    prepare.add_argument('--output',required=True);prepare.add_argument('--dev-fixture',required=True)
    prepare.add_argument('--proposal-seed',type=int,default=1010246101)
    audit=sub.add_parser('audit');audit.add_argument('--panel',required=True)
    aggregate=sub.add_parser('aggregate');aggregate.add_argument('--reports',nargs='+',required=True)
    aggregate.add_argument('--output',required=True)
    run=sub.add_parser('evaluate');run.add_argument('--panel',required=True)
    run.add_argument('--generator',required=True);run.add_argument('--output',required=True)
    run.add_argument('--round',type=int,default=0);run.add_argument('--baseline')
    run.add_argument('--max-wall-seconds',type=float,default=3600)
    for name in ('tail','student'):
        run.add_argument('--'+name+'-bank');run.add_argument('--'+name+'-proposer')
    args=parser.parse_args()
    if args.mode=='prepare':
        value=prepare_panel(args.source_plan,args.output,dev_fixture=args.dev_fixture,proposal_seed=args.proposal_seed)
        print(json.dumps(dict(n=value['n'],role=value['role'],teacher_max_control_steps=value['teacher_max_control_steps'])))
    elif args.mode=='audit':
        value,rows=audit_panel(args.panel);print(json.dumps(dict(phase='verified',n=len(rows),role=value['role'])))
    elif args.mode=='aggregate':
        value=aggregate_reports(args.reports,args.output)
        print(json.dumps(dict(rounds=value['rounds'],n=value['n'],output=args.output,environment_interactions=0)))
    else:
        models={}
        for name in ('tail','student'):
            bank=getattr(args,name+'_bank');proposer=getattr(args,name+'_proposer')
            if bool(bank)!=bool(proposer):parser.error(name+' bank and proposer must be specified together')
            models[name]=dict(bank=bank,proposer=proposer) if bank else None
        value=evaluate_checkpoint(args.panel,args.generator,args.output,round_index=args.round,
            baseline=args.baseline,tail_model=models['tail'],student_model=models['student'],max_wall_seconds=args.max_wall_seconds)
        print(json.dumps(value['summary']))


if __name__=='__main__':main()
