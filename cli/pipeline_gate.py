"""Emit compact PASS/FAIL decisions consumed by the remaining-stage controller."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dvgc.bank import SnapshotBank
from dvgc.pipeline import audit_decision, certification_decision, training_decision


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def write(output, report):
    path=Path(output)
    if path.exists(): raise SystemExit(f"Gate output already exists: {path}")
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(report,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(report,indent=2)); raise SystemExit(0 if report["status"]=="PASS" else 2)


def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest="kind",required=True)
    p=sub.add_parser("candidate"); p.add_argument("--build",required=True); p.add_argument("--audit",required=True); p.add_argument("--output",required=True)
    p=sub.add_parser("training"); p.add_argument("--analysis",required=True); p.add_argument("--evaluation",required=True); p.add_argument("--reference-evaluation",default=""); p.add_argument("--minimum-final",type=float,required=True); p.add_argument("--maximum-timeout",type=float,default=.05); p.add_argument("--maximum-final-drop",type=float,default=.05); p.add_argument("--output",required=True)
    p=sub.add_parser("certification"); p.add_argument("--report",required=True); p.add_argument("--phase",required=True); p.add_argument("--minimum-safe",type=int,default=4); p.add_argument("--output",required=True)
    p=sub.add_parser("audit"); p.add_argument("--report",required=True); p.add_argument("--minimum-precision",type=float,default=.95); p.add_argument("--maximum-timeout",type=float,default=.05); p.add_argument("--output",required=True)
    p=sub.add_parser("evaluation"); p.add_argument("--report",required=True); p.add_argument("--minimum-final",type=float,default=0.0); p.add_argument("--maximum-timeout",type=float,default=.10); p.add_argument("--output",required=True)
    p=sub.add_parser("bank-count"); p.add_argument("--bank",required=True); p.add_argument("--phase",required=True)
    args=parser.parse_args()
    if args.kind=="candidate":
        build,audit=load(args.build),load(args.audit); reasons=[]
        if build.get("status")!="PASS": reasons.append("candidate build failed")
        if audit.get("status")!="PASS": reasons.append("candidate quality audit failed")
        report={"status":"PASS" if not reasons else "FAIL","phase":audit.get("phase"),"build":{k:build.get(k) for k in ("target","attempts","accepted_new","duplicates","deduplication_rate","proposal_physical_failure_rate","proposal_timeout_rate")},"quality":{k:audit.get(k) for k in ("candidate_count","bootstrap_eligible","training_only","contact_audit","rollout_audit","quality_flags")},"reasons":reasons}; write(args.output,report)
    if args.kind=="training":
        report=training_decision(load(args.analysis),load(args.evaluation),minimum_final=args.minimum_final,maximum_timeout=args.maximum_timeout,reference_evaluation=load(args.reference_evaluation) if args.reference_evaluation else None,maximum_final_drop=args.maximum_final_drop); write(args.output,report)
    if args.kind=="certification": write(args.output,certification_decision(load(args.report),args.phase,args.minimum_safe))
    if args.kind=="audit": write(args.output,audit_decision(load(args.report),minimum_precision=args.minimum_precision,maximum_timeout=args.maximum_timeout))
    if args.kind=="evaluation":
        source=load(args.report); final=float(source.get("final_recovery_rate",0.0)); timeout=float(source.get("timeout_rate",1.0)); reasons=[]
        if final<float(args.minimum_final): reasons.append("Final-Recovery below evaluation gate")
        if timeout>float(args.maximum_timeout): reasons.append("timeout above evaluation gate")
        write(args.output,{"status":"PASS" if not reasons else "FAIL","policy_version":source.get("policy_version"),"episodes":source.get("episodes"),"final_recovery_rate":final,"chain_success_rate":source.get("chain_rate"),"physical_failure_rate":source.get("physical_failure_rate"),"timeout_rate":timeout,"termination_reason_counts":source.get("termination_reason_counts",{}),"minimum_final":args.minimum_final,"maximum_timeout":args.maximum_timeout,"reasons":reasons})
    bank=SnapshotBank.load(args.bank); print(len(bank.records_for_phase(args.phase,include_training_only=False)))


if __name__=="__main__": main()
