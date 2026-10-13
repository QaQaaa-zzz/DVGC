#!/usr/bin/env python3
"""FrontierBridge stage; prepare/audit/dry-run perform zero physical work."""
import argparse
import json
from pathlib import Path
from jit_dvgc import retention_consolidation as stage


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('prepare')
    for key in ('config','output','repository'):p.add_argument('--'+key,type=Path,required=True)
    for name in ('audit','dry-run','run','worker','report'):
        p=commands.add_parser(name);p.add_argument('--plan',type=Path,required=True)
        if name in ('run','worker'):
            p.add_argument('--execute',action='store_true',required=True)
            p.add_argument('--authorization',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='prepare':print(stage.prepare(args.config,args.output,args.repository))
    else:
        function=getattr(stage,args.command.replace('-','_'))
        result=function(args.plan,args.authorization) if args.command in ('run','worker') else function(args.plan)
        print(json.dumps(result,indent=2,default=str,allow_nan=False))


if __name__=='__main__':main()
