"""Frozen-policy dual certification: recursive Chain and end-to-end Final Recovery."""
from __future__ import annotations
import argparse, json, uuid
from pathlib import Path
import jax
import numpy as np
from dvgc.bank import SnapshotBank, beta_posterior, posterior_label
from dvgc.config import load_config
from dvgc.env import OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.rollout import restore_snapshot, frozen_rollout
from dvgc.runtime import build_inference


def protocol(cfg):
    return {"alpha0":cfg.beta_alpha0,"beta0":cfg.beta_beta0,"q_low":cfg.posterior_q_low,"q_high":cfg.posterior_q_high,"min_branches":cfg.min_branches,"safe_threshold":cfg.safe_threshold,"dead_threshold":cfg.dead_threshold,"boundary_max_width":cfg.boundary_max_width}


def decided(s,f,cfg):
    n=s+f; p=beta_posterior(s,f,alpha0=cfg.beta_alpha0,beta0=cfg.beta_beta0,q_low=cfg.posterior_q_low,q_high=cfg.posterior_q_high)
    return posterior_label(p,n,min_branches=cfg.min_branches,safe_threshold=cfg.safe_threshold,dead_threshold=cfg.dead_threshold,boundary_max_width=cfg.boundary_max_width)!="unknown"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy",required=True); p.add_argument("--candidate-bank",required=True)
    p.add_argument("--downstream-bank",default=""); p.add_argument("--phase",required=True,choices=["landing","flight","takeoff","approach"])
    p.add_argument("--output-bank",required=True); p.add_argument("--seed",type=int,default=0)
    p.add_argument("--namespace",default="build"); p.add_argument("--limit",type=int,default=0)
    a=p.parse_args(); params,cfg_dict,manifest=load_bundle(a.policy,verify_files=True)
    cfg=load_config(overrides={**cfg_dict,"training_stage":a.phase,"domain_randomization":False,"obs_noise_enable":False})
    candidates=SnapshotBank.load(a.candidate_bank); downstream=SnapshotBank.load(a.downstream_bank) if a.downstream_bank else SnapshotBank()
    if a.phase!="landing" and not a.downstream_bank: raise SystemExit("--downstream-bank is mandatory outside Landing")
    variants=[]
    scales=[(.95,.90,.95,1.0),(1.0,1.0,1.0,1.0),(1.05,1.10,1.05,1.0)]
    for mass,fric,force,gravity in scales:
        vc=load_config(overrides={**cfg.to_dict(),"mass_scale":mass,"friction_scale":fric,"actuator_force_scale":force,"gravity_scale":gravity})
        variants.append(OrangeBikeDVGC(vc,snapshot_bank=SnapshotBank(),cert_bank=downstream))
    inference=build_inference(variants[0],params,deterministic=True)
    rows=candidates.records_for_phase(a.phase,include_training_only=False); rows=rows[:a.limit or None]
    tube_version=f"{a.phase}-{uuid.uuid4().hex[:10]}"; results=[]
    for ri,row in enumerate(rows):
        cs=cf=fs=ff=0
        for b in range(int(cfg.max_branches)):
            env=variants[b%len(variants)]; key=jax.random.PRNGKey(a.seed+ri*10000+b)
            state=restore_snapshot(env,row,key)
            _,out=frozen_rollout(env,inference,state,key,horizon=int(cfg.branch_horizon),action_noise_std=float(cfg.action_noise_std))
            cs+=out["chain"]; cf+=1-out["chain"]; fs+=out["final"]; ff+=1-out["final"]
            if b+1>=int(cfg.min_branches) and decided(cs,cf,cfg) and decided(fs,ff,cfg): break
        candidates.update_certification(row["id"],chain_successes=cs,chain_failures=cf,final_successes=fs,final_failures=ff,policy_version=manifest["policy_version"],estimator_version=manifest.get("estimator_version","event_filter_v1"),tube_version=tube_version,protocol=protocol(cfg),seed_namespace=f"{a.namespace}:{a.phase}")
        results.append({"id":row["id"],"chain":cs,"final":fs,"branches":cs+cf})
        print(f"[cert] {ri+1}/{len(rows)} chain={cs}/{cs+cf} final={fs}/{fs+ff}")
    candidates.metadata.update({"last_policy_version":manifest["policy_version"],"last_tube_version":tube_version,"downstream_bank":a.downstream_bank})
    candidates.save(a.output_bank)
    report={"phase":a.phase,"tube_version":tube_version,"summary":candidates.summary(),"results":results}
    Path(a.output_bank).with_suffix(".cert.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report["summary"],indent=2))
if __name__=="__main__": main()
