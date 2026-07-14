"""Independent Tube audit with a disjoint seed namespace."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import jax
from dvgc.bank import SnapshotBank
from dvgc.audit import build_audit_report
from dvgc.certification import DYNAMICS_VARIANTS, assert_disjoint_branch_seeds, branch_evidence, branch_seed, summarize_branches
from dvgc.config import load_config
from dvgc.env import OrangeBikeDVGC
from dvgc.policy import load_bundle, verify_manifest_artifact
from dvgc.rollout import restore_snapshot, frozen_rollout
from dvgc.runtime import build_inference


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy",required=True); p.add_argument("--bank",required=True); p.add_argument("--downstream-bank",default="")
    p.add_argument("--phase",required=True,choices=["landing","flight","takeoff","approach"]); p.add_argument("--output",required=True)
    p.add_argument("--seed",type=int,default=1000000); p.add_argument("--namespace",default="audit"); p.add_argument("--branches",type=int,default=16); p.add_argument("--limit",type=int,default=0)
    p.add_argument("--start-index",type=int,default=0)
    a=p.parse_args(); params,cfg_dict,manifest=load_bundle(a.policy,verify_files=True)
    output=Path(a.output)
    if output.exists(): raise SystemExit(f"Audit output already exists: {output}")
    if manifest.get("stage") not in (None,a.phase): raise SystemExit(f"Policy stage {manifest.get('stage')} cannot audit {a.phase}")
    try: verify_manifest_artifact(manifest,"downstream_bank",a.downstream_bank or None,required=(a.phase!="landing"))
    except ValueError as exc: raise SystemExit(str(exc)) from exc
    cfg=load_config(overrides={**cfg_dict,"training_stage":a.phase,"domain_randomization":False,"obs_noise_enable":False})
    bank=SnapshotBank.load(a.bank); downstream=SnapshotBank.load(a.downstream_bank) if a.downstream_bank else SnapshotBank()
    if a.phase!="landing" and not a.downstream_bank: raise SystemExit("--downstream-bank is mandatory outside Landing")
    try: bank.validate_certification_provenance(a.phase,policy_version=manifest["policy_version"],estimator_version=manifest.get("estimator_version","event_filter_v1"))
    except ValueError as exc: raise SystemExit(str(exc)) from exc
    variants=[]
    for spec in DYNAMICS_VARIANTS:
        overrides={key:value for key,value in spec.items() if key!="id"}
        vc=load_config(overrides={**cfg.to_dict(),**overrides})
        env=OrangeBikeDVGC(vc,snapshot_bank=SnapshotBank(),cert_bank=downstream)
        variants.append((spec["id"],env,jax.jit(env.step)))
    inference=build_inference(variants[0][1],params,deterministic=True)
    all_rows=[r for r in bank.records_for_phase(a.phase,include_training_only=False) if r["final"]["branches"]>0]
    start=int(a.start_index)
    if start<0 or start>=len(all_rows): raise SystemExit(f"--start-index must be in [0,{len(all_rows)-1}]")
    stop=min(len(all_rows),start+(int(a.limit) if a.limit else len(all_rows)))
    indexed_rows=list(enumerate(all_rows))[start:stop]
    namespace=f"{a.namespace}:{a.phase}"; audit=[]
    for local_index,(i,row) in enumerate(indexed_rows):
        audit_seeds=[branch_seed(a.seed,i,b) for b in range(a.branches)]
        if str(row.get("seed_namespace"))==namespace: raise SystemExit(f"Audit namespace reuses Tube construction namespace: {namespace}")
        try: assert_disjoint_branch_seeds(row.get("certification_branches",[]),audit_seeds)
        except ValueError as exc: raise SystemExit(str(exc)) from exc
        cs=fs=0; branches=[]
        for b in range(a.branches):
            variant_id,env,step_fn=variants[b%len(variants)]; seed=audit_seeds[b]; key=jax.random.PRNGKey(seed); state=restore_snapshot(env,row,key)
            _,out=frozen_rollout(env,inference,state,key,horizon=int(cfg.branch_horizon),action_noise_std=float(cfg.action_noise_std),step_fn=step_fn)
            evidence=branch_evidence(branch_index=b,seed=seed,seed_namespace=namespace,dynamics_variant=variant_id,outcome=out)
            branches.append(evidence); cs+=out["chain"]; fs+=out["final"]
        p=fs/a.branches; terminal=summarize_branches(branches); audit.append({"id":row["id"],"predicted_label":row["final"]["label"],"predicted_mean":row["final"]["posterior"]["mean"],"audit_chain":cs/a.branches,"audit_final":p,"terminal_summary":terminal,"branches":branches})
        audit[-1]["state_index"]=i
        print(f"[audit] {local_index+1}/{len(indexed_rows)} global={i+1}/{len(all_rows)} chain={cs}/{a.branches} final={fs}/{a.branches} physical_failure={terminal['physical_failures']} timeout={terminal['timeouts']} horizon={terminal['horizon_exhaustions']}")
    report=build_audit_report(audit,policy_version=manifest["policy_version"],phase=a.phase,seed_namespace=namespace,branches_per_state=a.branches,safe_threshold=float(cfg.safe_threshold),dynamics_variants=DYNAMICS_VARIANTS)
    report.update({"state_index_start":start,"state_index_end_exclusive":stop,"total_bank_states":len(all_rows)})
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps({k:v for k,v in report.items() if k!="rows"},indent=2))
if __name__=="__main__": main()
