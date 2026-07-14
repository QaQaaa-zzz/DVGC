"""Frozen-policy dual certification: recursive Chain and end-to-end Final Recovery."""
from __future__ import annotations
import argparse, json, uuid
from pathlib import Path
import jax
from dvgc.bank import SnapshotBank, beta_posterior, posterior_label
from dvgc.certification import DYNAMICS_VARIANTS, branch_evidence, branch_seed, summarize_branches
from dvgc.config import load_config
from dvgc.env import OrangeBikeDVGC
from dvgc.policy import load_bundle, verify_manifest_artifact
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
    if manifest.get("stage") not in (None,a.phase): raise SystemExit(f"Policy stage {manifest.get('stage')} cannot certify {a.phase}")
    if Path(a.candidate_bank).resolve()==Path(a.output_bank).resolve(): raise SystemExit("--output-bank must differ from --candidate-bank so policy provenance remains immutable")
    try:
        verify_manifest_artifact(manifest,"candidate_bank",a.candidate_bank)
        verify_manifest_artifact(manifest,"downstream_bank",a.downstream_bank or None,required=(a.phase!="landing"))
    except ValueError as exc: raise SystemExit(str(exc)) from exc
    cfg=load_config(overrides={**cfg_dict,"training_stage":a.phase,"domain_randomization":False,"obs_noise_enable":False})
    candidates=SnapshotBank.load(a.candidate_bank); downstream=SnapshotBank.load(a.downstream_bank) if a.downstream_bank else SnapshotBank()
    if a.phase!="landing" and not a.downstream_bank: raise SystemExit("--downstream-bank is mandatory outside Landing")
    variants=[]
    for spec in DYNAMICS_VARIANTS:
        overrides={key:value for key,value in spec.items() if key!="id"}
        vc=load_config(overrides={**cfg.to_dict(),**overrides})
        env=OrangeBikeDVGC(vc,snapshot_bank=SnapshotBank(),cert_bank=downstream)
        variants.append((spec["id"],env,jax.jit(env.step)))
    inference=build_inference(variants[0][1],params,deterministic=True)
    rows=candidates.records_for_phase(a.phase,include_training_only=False); rows=rows[:a.limit or None]
    if not rows: raise SystemExit(f"Candidate bank has no certifiable {a.phase} records")
    tube_version=f"{a.phase}-{uuid.uuid4().hex[:10]}"; namespace=f"{a.namespace}:{a.phase}"; results=[]; all_branches=[]
    candidates.invalidate_phase(a.phase,reason=f"fresh certification for {manifest['policy_version']}")
    for ri,row in enumerate(rows):
        cs=cf=fs=ff=0; branches=[]
        for b in range(int(cfg.max_branches)):
            variant_id,env,step_fn=variants[b%len(variants)]; seed=branch_seed(a.seed,ri,b); key=jax.random.PRNGKey(seed)
            state=restore_snapshot(env,row,key)
            _,out=frozen_rollout(env,inference,state,key,horizon=int(cfg.branch_horizon),action_noise_std=float(cfg.action_noise_std),step_fn=step_fn)
            evidence=branch_evidence(branch_index=b,seed=seed,seed_namespace=namespace,dynamics_variant=variant_id,outcome=out)
            branches.append(evidence); all_branches.append(evidence)
            cs+=out["chain"]; cf+=1-out["chain"]; fs+=out["final"]; ff+=1-out["final"]
            if b+1>=int(cfg.min_branches) and decided(cs,cf,cfg) and decided(fs,ff,cfg): break
        candidates.update_certification(row["id"],chain_successes=cs,chain_failures=cf,final_successes=fs,final_failures=ff,policy_version=manifest["policy_version"],estimator_version=manifest.get("estimator_version","event_filter_v1"),tube_version=tube_version,protocol=protocol(cfg),seed_namespace=namespace,branch_evidence=branches)
        terminal=summarize_branches(branches)
        results.append({"id":row["id"],"chain":cs,"final":fs,"branches":cs+cf,"branch_evidence":branches,"terminal_summary":terminal})
        print(f"[cert] {ri+1}/{len(rows)} chain={cs}/{cs+cf} final={fs}/{fs+ff} physical_failure={terminal['physical_failures']} timeout={terminal['timeouts']} horizon={terminal['horizon_exhaustions']}")
    candidates.metadata.update({"last_policy_version":manifest["policy_version"],"last_tube_version":tube_version,"downstream_bank":a.downstream_bank,"construction_seed":int(a.seed),"construction_seed_namespace":namespace,"dynamics_variants":[dict(spec) for spec in DYNAMICS_VARIANTS]})
    candidates.save(a.output_bank)
    report={"phase":a.phase,"tube_version":tube_version,"seed_namespace":namespace,"summary":candidates.summary(),"terminal_summary":summarize_branches(all_branches),"results":results}
    Path(a.output_bank).with_suffix(".cert.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report["summary"],indent=2))
if __name__=="__main__": main()
