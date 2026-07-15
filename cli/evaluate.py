"""Evaluate a frozen policy on fixed candidates or natural-start full jumps."""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path
import jax
import numpy as np
from dvgc.bank import SnapshotBank
from dvgc.config import load_config
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.rollout import restore_snapshot, frozen_rollout
from dvgc.runtime import build_inference


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--policy",required=True); p.add_argument("--stage",required=True,choices=["landing","flight","takeoff","approach","full"]); p.add_argument("--bank",default=""); p.add_argument("--downstream-bank",default=""); p.add_argument("--episodes",type=int,default=100); p.add_argument("--seed",type=int,default=2000000); p.add_argument("--output",required=True); a=p.parse_args()
    output=Path(a.output)
    if output.exists(): raise SystemExit(f"Evaluation output already exists: {output}")
    params,cfg_dict,manifest=load_bundle(a.policy,verify_files=True); cfg=load_config(overrides={**cfg_dict,"training_stage":a.stage,"domain_randomization":False,"obs_noise_enable":False,"use_bank_resets":False})
    bank=SnapshotBank.load(a.bank) if a.bank else SnapshotBank(); downstream=SnapshotBank.load(a.downstream_bank) if a.downstream_bank else SnapshotBank(); env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank(),cert_bank=downstream); inference=build_inference(env,params,deterministic=True)
    rows=bank.records_for_phase(a.stage,include_training_only=False) if a.stage!="full" and a.bank else []
    if a.bank and not rows: raise SystemExit(f"Bank has no certifiable {a.stage} records")
    step_fn=jax.jit(env.step)
    out=[]
    for i in range(a.episodes):
        key=jax.random.PRNGKey(a.seed+i); row=rows[i%len(rows)] if rows else None
        state=restore_snapshot(env,row,key) if row else env.reset(key)
        _,result=frozen_rollout(env,inference,state,key,horizon=int(cfg.branch_horizon),action_noise_std=0.0,step_fn=step_fn)
        result["episode"]=i; result["seed"]=a.seed+i
        if row:
            result["candidate_id"]=row["id"]
            result["candidate_kind"]=row.get("candidate_kind","unknown")
        result["termination_reason"]=END_REASON.get(result["end_code"],f"unknown_{result['end_code']}")
        out.append(result)
    reasons=Counter(x["termination_reason"] for x in out)
    def summarize(values):
        return {"episodes":len(values),"chain_rate":float(np.mean([x["chain"] for x in values])),"final_recovery_rate":float(np.mean([x["final"] for x in values])),"physical_failure_rate":float(np.mean([x["terminated"] and not x["final"] for x in values])),"timeout_rate":float(np.mean([x["truncated"] for x in values]))}
    grouped={kind:summarize([x for x in out if x.get("candidate_kind")==kind]) for kind in sorted({x.get("candidate_kind") for x in out if x.get("candidate_kind")})}
    report={"policy_version":manifest["policy_version"],"stage":a.stage,"episodes":len(out),"chain_rate":float(np.mean([x["chain"] for x in out])),"final_recovery_rate":float(np.mean([x["final"] for x in out])),"physical_failure_rate":float(np.mean([x["terminated"] and not x["final"] for x in out])),"timeout_rate":float(np.mean([x["truncated"] for x in out])),"mean_steps":float(np.mean([x["steps"] for x in out])),"termination_reason_counts":dict(sorted(reasons.items())),"grouped_candidate_metrics":grouped,"rows":out}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps({key:value for key,value in report.items() if key!="rows"},indent=2))
if __name__=="__main__": main()
