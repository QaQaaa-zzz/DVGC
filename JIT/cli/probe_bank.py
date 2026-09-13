#!/usr/bin/env python3
"""Lock complementary probes, collect arrivals, and execute bounded suffix labels."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from jit_dvgc.probe_bank import (lock_probe_bank, acquire_probe_catalog,
                               prepare_probe_labels, run_probe_labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("lock", "acquire"):
        command = sub.add_parser(name)
        command.add_argument("--spec", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    prepare = sub.add_parser("prepare-labels")
    prepare.add_argument("--bank", type=Path, required=True)
    prepare.add_argument("--catalog", type=Path, action="append", required=True)
    prepare.add_argument("--role", choices=("train", "calibration", "acceptance"), required=True)
    prepare.add_argument("--seed", type=int, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run-labels")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--python", default=sys.executable)
    prepare_exists = sub.add_parser("prepare-existence")
    prepare_exists.add_argument("--bank",type=Path,required=True)
    prepare_exists.add_argument("--catalog",type=Path,required=True)
    prepare_exists.add_argument("--output",type=Path,required=True)
    prepare_exists.add_argument("--order",nargs='+',required=True)
    prepare_exists.add_argument("--seed",type=int,required=True)
    prepare_exists.add_argument("--budget",type=int,required=True)
    prepare_exists.add_argument("--backend",choices=['serial','vectorized'],default='serial')
    prepare_exists.add_argument("--batch-size",type=int,default=1)
    prepare_exists.add_argument("--indices",type=Path)
    prepare_exists.add_argument("--max-candidates-per-process",type=int,default=4096)
    execute_exists = sub.add_parser("run-existence")
    execute_exists.add_argument("--plan",type=Path,required=True)
    execute_exists.add_argument("--output",type=Path,required=True)
    execute_exists.add_argument("--gpu",default='0')
    args = parser.parse_args()
    if args.command == 'prepare-existence':
        from jit_dvgc.continuation.existence import prepare
        result=prepare(args.bank,args.catalog,args.output,order=args.order,seed=args.seed,budget=args.budget,
                       backend=args.backend,batch_size=args.batch_size,
                       indices=json.loads(args.indices.read_text()) if args.indices else None,
                       max_candidates_per_process=args.max_candidates_per_process)
        print(json.dumps(result,indent=2));return 0
    if args.command == 'run-existence':
        from jit_dvgc.continuation.existence import run
        result=run(args.plan,args.output,gpu=args.gpu)
        print(json.dumps({k:v for k,v in result.items() if k!='entries'},indent=2));return 0 if result['status']=='completed' else 2
    if args.command == "lock":
        result = lock_probe_bank(json.loads(args.spec.read_text()), args.output)
    elif args.command == "acquire":
        result = acquire_probe_catalog(args.spec, args.output)
    elif args.command == "prepare-labels":
        result = prepare_probe_labels(args.bank, args.catalog, args.output, role=args.role, seed=args.seed)
    else:
        result = run_probe_labels(args.plan, args.output, python=args.python)
    print(json.dumps({k: v for k, v in result.items() if k != "entries"}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
