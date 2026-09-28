#!/usr/bin/env python3
"""Read-only audit / prepare entry. No implicit training or data recapture."""
import argparse
import json
from pathlib import Path
import yaml
from jit_dvgc.generative_bridge.reporting import prepare
from jit_dvgc.generative_bridge.protocol import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['audit','prepare','run-pilot'])
    p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    spec=yaml.safe_load(args.spec.read_text())
    report=prepare(spec)
    if args.execute:
        # This release is intentionally not a production launch authorization.
        # Never turn an unresolved source into the latest checkpoint by guessing.
        from jit_dvgc.generative_bridge.contracts import validate_spec
        validate_spec(spec,execute=True)
        raise ValueError('production execution requires a resolved source-bound stage plan and GPU validation; preparation only')
    args.output.mkdir(parents=True,exist_ok=False)
    atomic_json(args.output/'prepare_report.json',report)
    atomic_json(args.output/'cost_ledger.json',report['cost_ledger'])
    atomic_json(args.output/'comparison_plan.json',report['comparison_arms'])
    print(json.dumps({'status':report['status'],'execute':False,'output':str(args.output),
                      'missing':report['missing']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
