"""Emit compact PASS/FAIL decisions consumed by the remaining-stage controller."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dvgc.bank import SnapshotBank
from dvgc.pipeline import audit_decision, certification_decision, curriculum_decision, training_decision


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
    p=sub.add_parser("curriculum"); p.add_argument("--evaluation",required=True); p.add_argument("--landing-retention",required=True); p.add_argument("--landing-reference",required=True); p.add_argument("--minimum-chain-lcb",type=float,required=True); p.add_argument("--minimum-final",type=float,required=True); p.add_argument("--maximum-landing-drop",type=float,default=.05); p.add_argument("--output",required=True)
    p=sub.add_parser("handoff"); p.add_argument("--landing-diagnostic",required=True); p.add_argument("--flight-diagnostic",required=True); p.add_argument("--entry-calibration",required=True); p.add_argument("--minimum-precision",type=float,default=.95); p.add_argument("--output",required=True)
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
    if args.kind=="curriculum": write(args.output,curriculum_decision(load(args.evaluation),load(args.landing_retention),load(args.landing_reference),minimum_chain_lcb=args.minimum_chain_lcb,minimum_final=args.minimum_final,maximum_landing_drop=args.maximum_landing_drop))
    if args.kind=="handoff":
        landing,flight,cal=load(args.landing_diagnostic),load(args.flight_diagnostic),load(args.entry_calibration); reasons=[]
        final=sum(v for k,v in landing["chain_final_table"].items() if k.endswith("final1"))
        if int(landing.get("downstream_final_safe_count",0))<=0: reasons.append("Landing entry set is empty")
        if int(landing.get("chain_trigger_count",0))<=0: reasons.append("Chain event is not reachable")
        if final and int(landing.get("missed_success",final))>=final: reasons.append("all Landing-policy Final successes are missed")
        if int(landing.get("false_progress",0))!=0: reasons.append("Landing-policy false progress present")
        if float(cal.get("independent_audit_precision",0))<float(args.minimum_precision): reasons.append("entry matcher precision below gate")
        if float(flight.get("chain_reward_total",0)) and int(flight.get("chain_trigger_count",0))<=0: reasons.append("Flight Chain reward/event mismatch")
        write(args.output,{"status":"PASS" if not reasons else "FAIL","entry_safe_count":landing.get("downstream_final_safe_count"),"landing_chain_final_table":landing.get("chain_final_table"),"flight_chain_final_table":flight.get("chain_final_table"),"landing_chain_reward":landing.get("chain_reward_total"),"flight_chain_reward":flight.get("chain_reward_total"),"entry_precision":cal.get("independent_audit_precision"),"entry_recall":cal.get("independent_audit_recall"),"reasons":reasons})
    bank=SnapshotBank.load(args.bank); print(len(bank.records_for_phase(args.phase,include_training_only=False)))


if __name__=="__main__": main()
