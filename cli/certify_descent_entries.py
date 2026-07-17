"""Independently certify C_D proposals under frozen pi_F,D -> pi_L."""
from __future__ import annotations

import argparse, json, uuid
from pathlib import Path

import jax

from dvgc.bank import SnapshotBank, beta_posterior, posterior_label
from dvgc.certification import DYNAMICS_VARIANTS, branch_evidence, branch_seed, summarize_branches
from dvgc.composite import CanonicalEntryMatcher, composite_rollout
from dvgc.config import file_sha256, load_config
from dvgc.descent_entry import descent_entry_feature
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.rollout import restore_snapshot
from dvgc.runtime import build_inference, save_json

def qualified_descent_success(outcome): return bool(outcome["chain"] and outcome["final"])


def label_decided(successes,failures,cfg):
    p=beta_posterior(successes,failures,alpha0=cfg.beta_alpha0,beta0=cfg.beta_beta0,q_low=cfg.posterior_q_low,q_high=cfg.posterior_q_high)
    return posterior_label(p,successes+failures,min_branches=cfg.min_branches,safe_threshold=cfg.safe_threshold,dead_threshold=cfg.dead_threshold,boundary_max_width=cfg.boundary_max_width)!="unknown"


def current_label(successes,failures,cfg):
    p=beta_posterior(successes,failures,alpha0=cfg.beta_alpha0,beta0=cfg.beta_beta0,q_low=cfg.posterior_q_low,q_high=cfg.posterior_q_high)
    return posterior_label(p,successes+failures,min_branches=cfg.min_branches,safe_threshold=cfg.safe_threshold,dead_threshold=cfg.dead_threshold,boundary_max_width=cfg.boundary_max_width)


def protocol(cfg):
    return {"alpha0":cfg.beta_alpha0,"beta0":cfg.beta_beta0,"q_low":cfg.posterior_q_low,"q_high":cfg.posterior_q_high,"min_branches":cfg.min_branches,"safe_threshold":cfg.safe_threshold,"dead_threshold":cfg.dead_threshold,"boundary_max_width":cfg.boundary_max_width}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--descent-policy",required=True); p.add_argument("--candidate-source-policy",default=""); p.add_argument("--landing-policy",required=True); p.add_argument("--candidate-bank",required=True); p.add_argument("--landing-entry-set",required=True); p.add_argument("--output",required=True); p.add_argument("--config",default="configs/default.json"); p.add_argument("--seed",type=int,required=True); p.add_argument("--namespace",required=True); p.add_argument("--audit-only",action="store_true"); p.add_argument("--confirm-safe-to-max",action="store_true"); p.add_argument("--start-index",type=int,default=0); p.add_argument("--end-index",type=int,default=None); a=p.parse_args(); out=Path(a.output)
    if out.exists(): raise SystemExit(f"Output exists: {out}")
    dp,dc,dm=load_bundle(a.descent_policy,verify_files=True); lp,_,lm=load_bundle(a.landing_policy,verify_files=True); source=SnapshotBank.load(a.candidate_bank)
    source_policy_hash=source.metadata.get("descent_policy_hash"); current_policy_hash=file_sha256(Path(a.descent_policy)/"params.pkl")
    if source_policy_hash!=current_policy_hash:
        if not a.candidate_source_policy or source_policy_hash!=file_sha256(Path(a.candidate_source_policy)/"params.pkl"): raise SystemExit("C_D proposal source-policy provenance mismatch")
    if source.metadata.get("landing_entry_set_sha256")!=file_sha256(a.landing_entry_set): raise SystemExit("C_D proposal C_L provenance mismatch")
    base=load_config(a.config,{**dc,"training_stage":"flight","expert_chain_termination":False,"domain_randomization":False,"obs_noise_enable":False,"use_bank_resets":False})
    variants=[]
    for spec in DYNAMICS_VARIANTS:
        cfg=load_config(a.config,{**base.to_dict(),**{k:v for k,v in spec.items() if k!="id"}}); entry=SnapshotBank.load(a.landing_entry_set); env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank(),cert_bank=entry)
        inf={"flight":build_inference(env,dp,deterministic=True),"landing":build_inference(env,lp,deterministic=True)}; variants.append((spec["id"],env,jax.jit(env.step),inf,CanonicalEntryMatcher(env,"flight",a.landing_entry_set)))
    all_rows=source.records_for_phase("flight",include_training_only=False); end=len(all_rows) if a.end_index is None else a.end_index
    if not (0<=a.start_index<end<=len(all_rows)): raise SystemExit(f"Invalid candidate index range [{a.start_index},{end}) for {len(all_rows)} states")
    if not a.audit_only and (a.start_index!=0 or end!=len(all_rows)): raise SystemExit("Candidate slicing is restricted to immutable audit-only runs")
    indexed_rows=list(enumerate(all_rows))[a.start_index:end]; results=[]; all_evidence=[]; tube_version=f"descent-entry-{uuid.uuid4().hex[:10]}"; work=SnapshotBank(source.records,source.metadata)
    for record in work.records:
        record["entry_feature"] = descent_entry_feature(record["physical_feature"], base).astype("float32")
    if not a.audit_only: work.invalidate_phase("flight",reason=f"C_D certification under {dm['policy_version']} -> {lm['policy_version']}")
    for local_index,(i,row) in enumerate(indexed_rows):
        successes=failures=chain_successes=chain_failures=raw_final_successes=0; evidence=[]
        for b in range(int(base.max_branches)):
            variant,env,step,inf,matcher=variants[b%len(variants)]; seed=branch_seed(a.seed,i,b); key=jax.random.PRNGKey(seed)
            _,outcome=composite_rollout(env,("flight","landing"),inf,{"flight":matcher},restore_snapshot(env,row,key),key,horizon=int(base.branch_horizon),step_fn=step,action_noise_std=float(base.action_noise_std))
            ev=branch_evidence(branch_index=b,seed=seed,seed_namespace=f"{a.namespace}:descent_entry",dynamics_variant=variant,outcome=outcome); qualified=qualified_descent_success(outcome); ev["end_code"]=int(outcome["end_code"]); ev["end_reason"]=END_REASON.get(int(outcome["end_code"]),"unknown"); ev["raw_composite_final_recovery"]=bool(outcome["final"]); ev["descent_entry_final_success"]=qualified
            if outcome["final"] and not outcome["chain"]: ev["final_recovery"]=False; ev["terminal_cause"]="handoff_missed_final"
            evidence.append(ev); all_evidence.append(ev)
            chain_successes+=int(outcome["chain"]); chain_failures+=int(not outcome["chain"]); raw_final_successes+=int(outcome["final"]); successes+=int(qualified); failures+=int(not qualified)
            if not a.audit_only and b+1>=int(base.min_branches) and label_decided(successes,failures,base) and label_decided(chain_successes,chain_failures,base):
                provisional_safe=current_label(successes,failures,base)=="safe" or current_label(chain_successes,chain_failures,base)=="safe"
                if not (a.confirm_safe_to_max and provisional_safe and b+1<int(base.max_branches)): break
        if not a.audit_only: work.update_certification(row["id"],chain_successes=chain_successes,chain_failures=chain_failures,final_successes=successes,final_failures=failures,policy_version=dm["policy_version"],estimator_version=dm.get("estimator_version","event_filter_v1"),tube_version=tube_version,protocol=protocol(base),seed_namespace=f"{a.namespace}:descent_entry",branch_evidence=evidence)
        results.append({"id":row["id"],"candidate_index":i,"source_id":row.get("entry_source_id"),"proposal_step":row.get("proposal_step"),"chain":chain_successes,"raw_final":raw_final_successes,"final":successes,"branches":successes+failures,"final_rate":successes/(successes+failures),"branch_evidence":evidence})
        print(f"[C_D {'audit' if a.audit_only else 'cert'}] {local_index+1}/{len(indexed_rows)} global={i} chain={chain_successes}/{chain_successes+chain_failures} final={successes}/{successes+failures}")
    summary=summarize_branches(all_evidence); common={"status":"PASS","audit_only":a.audit_only,"confirm_safe_to_max":a.confirm_safe_to_max,"seed":a.seed,"seed_namespace":f"{a.namespace}:descent_entry","candidate_bank_sha256":file_sha256(a.candidate_bank),"candidate_source_policy_hash":source_policy_hash,"landing_entry_set_sha256":file_sha256(a.landing_entry_set),"descent_policy_hash":current_policy_hash,"landing_policy_hash":file_sha256(Path(a.landing_policy)/"params.pkl"),"states":len(indexed_rows),"total_states":len(all_rows),"start_index":a.start_index,"end_index":end,"terminal_summary":summary,"rows":results}
    if a.audit_only: save_json(out,common)
    else:
        work.metadata.update({"entry_bank_role":"certified_descent_handoff_candidates","last_policy_version":dm["policy_version"],"last_tube_version":tube_version,"construction_seed":a.seed,"construction_seed_namespace":f"{a.namespace}:descent_entry","landing_policy_version":lm["policy_version"],"landing_policy_hash":common["landing_policy_hash"],"landing_entry_set_sha256":common["landing_entry_set_sha256"],"dynamics_variants":[dict(x) for x in DYNAMICS_VARIANTS]}); work.save(out); common.update({"tube_version":tube_version,"summary":work.summary(),"bank_sha256":file_sha256(out)}); save_json(out.with_suffix(".cert.json"),common)
    print(json.dumps({k:v for k,v in common.items() if k not in ("rows",)},indent=2))


if __name__=="__main__": main()
