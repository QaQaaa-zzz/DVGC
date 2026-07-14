"""Independent Tube audit with a disjoint seed namespace."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import jax
import numpy as np
from dvgc.bank import SnapshotBank
from dvgc.config import load_config
from dvgc.env import OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.rollout import restore_snapshot, frozen_rollout
from dvgc.runtime import build_inference


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy",required=True); p.add_argument("--bank",required=True); p.add_argument("--downstream-bank",default="")
    p.add_argument("--phase",required=True,choices=["landing","flight","takeoff","approach"]); p.add_argument("--output",required=True)
    p.add_argument("--seed",type=int,default=1000000); p.add_argument("--branches",type=int,default=16); p.add_argument("--limit",type=int,default=0)
    a=p.parse_args(); params,cfg_dict,manifest=load_bundle(a.policy,verify_files=True)
    cfg=load_config(overrides={**cfg_dict,"training_stage":a.phase,"domain_randomization":False,"obs_noise_enable":False})
    bank=SnapshotBank.load(a.bank); downstream=SnapshotBank.load(a.downstream_bank) if a.downstream_bank else SnapshotBank()
    if a.phase!="landing" and not a.downstream_bank: raise SystemExit("--downstream-bank is mandatory outside Landing")
    env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank(),cert_bank=downstream); inference=build_inference(env,params,deterministic=True)
    rows=[r for r in bank.records_for_phase(a.phase,include_training_only=False) if r["final"]["branches"]>0]
    rows=rows[:a.limit or None]; audit=[]; branch_chain=[]; branch_final=[]
    for i,row in enumerate(rows):
        cs=fs=0
        for b in range(a.branches):
            key=jax.random.PRNGKey(a.seed+i*10000+b); state=restore_snapshot(env,row,key)
            _,out=frozen_rollout(env,inference,state,key,horizon=int(cfg.branch_horizon),action_noise_std=float(cfg.action_noise_std))
            cs+=out["chain"]; fs+=out["final"]; branch_chain.append(out["chain"]); branch_final.append(out["final"])
        p=fs/a.branches; audit.append({"id":row["id"],"predicted_label":row["final"]["label"],"predicted_mean":row["final"]["posterior"]["mean"],"audit_chain":cs/a.branches,"audit_final":p})
    pred_safe=np.asarray([r["predicted_label"]=="safe" for r in audit]); recoverable=np.asarray([r["audit_final"]>=cfg.safe_threshold for r in audit]); probs=np.asarray([r["predicted_mean"] for r in audit],float); obs=np.asarray([r["audit_final"] for r in audit],float)
    precision=float(recoverable[pred_safe].mean()) if pred_safe.any() else float("nan")
    recall=float(pred_safe[recoverable].mean()) if recoverable.any() else float("nan")
    coverage=float(pred_safe.mean()) if len(pred_safe) else 0.0
    brier=float(np.mean((probs-obs)**2)) if len(obs) else float("nan")
    bins=np.linspace(0,1,6); ece=0.0
    for lo,hi in zip(bins[:-1],bins[1:]):
        mask=(probs>=lo)&(probs<(hi if hi<1 else hi+1e-9))
        if mask.any(): ece+=float(mask.mean()*abs(probs[mask].mean()-obs[mask].mean()))
    bc=np.asarray(branch_chain,bool); bf=np.asarray(branch_final,bool)
    report={"policy_version":manifest["policy_version"],"phase":a.phase,"seed_namespace":"audit","states":len(audit),"branches_per_state":a.branches,"tube_precision":precision,"recoverable_recall":recall,"candidate_mass_coverage":coverage,"brier":brier,"ece_5bin":ece,"false_progress_rate":float(np.mean(bc & ~bf)) if len(bc) else float("nan"),"missed_success_rate":float(np.mean(~bc & bf)) if len(bc) else float("nan"),"rows":audit}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps({k:v for k,v in report.items() if k!="rows"},indent=2))
if __name__=="__main__": main()
