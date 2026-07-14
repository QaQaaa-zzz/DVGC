"""Independent Tube audit with a disjoint seed namespace."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import jax
import numpy as np
from dvgc.bank import SnapshotBank
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
    a=p.parse_args(); params,cfg_dict,manifest=load_bundle(a.policy,verify_files=True)
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
    rows=[r for r in bank.records_for_phase(a.phase,include_training_only=False) if r["final"]["branches"]>0]
    rows=rows[:a.limit or None]; namespace=f"{a.namespace}:{a.phase}"; audit=[]; all_branches=[]
    for i,row in enumerate(rows):
        audit_seeds=[branch_seed(a.seed,i,b) for b in range(a.branches)]
        if str(row.get("seed_namespace"))==namespace: raise SystemExit(f"Audit namespace reuses Tube construction namespace: {namespace}")
        try: assert_disjoint_branch_seeds(row.get("certification_branches",[]),audit_seeds)
        except ValueError as exc: raise SystemExit(str(exc)) from exc
        cs=fs=0; branches=[]
        for b in range(a.branches):
            variant_id,env,step_fn=variants[b%len(variants)]; seed=audit_seeds[b]; key=jax.random.PRNGKey(seed); state=restore_snapshot(env,row,key)
            _,out=frozen_rollout(env,inference,state,key,horizon=int(cfg.branch_horizon),action_noise_std=float(cfg.action_noise_std),step_fn=step_fn)
            evidence=branch_evidence(branch_index=b,seed=seed,seed_namespace=namespace,dynamics_variant=variant_id,outcome=out)
            branches.append(evidence); all_branches.append(evidence); cs+=out["chain"]; fs+=out["final"]
        p=fs/a.branches; terminal=summarize_branches(branches); audit.append({"id":row["id"],"predicted_label":row["final"]["label"],"predicted_mean":row["final"]["posterior"]["mean"],"audit_chain":cs/a.branches,"audit_final":p,"terminal_summary":terminal,"branches":branches})
        print(f"[audit] {i+1}/{len(rows)} chain={cs}/{a.branches} final={fs}/{a.branches} physical_failure={terminal['physical_failures']} timeout={terminal['timeouts']} horizon={terminal['horizon_exhaustions']}")
    pred_safe=np.asarray([r["predicted_label"]=="safe" for r in audit]); recoverable=np.asarray([r["audit_final"]>=cfg.safe_threshold for r in audit]); probs=np.asarray([r["predicted_mean"] for r in audit],float); obs=np.asarray([r["audit_final"] for r in audit],float)
    precision=float(recoverable[pred_safe].mean()) if pred_safe.any() else float("nan")
    recall=float(pred_safe[recoverable].mean()) if recoverable.any() else float("nan")
    coverage=float(pred_safe.mean()) if len(pred_safe) else 0.0
    brier=float(np.mean((probs-obs)**2)) if len(obs) else float("nan")
    bins=np.linspace(0,1,6); ece=0.0
    for lo,hi in zip(bins[:-1],bins[1:]):
        mask=(probs>=lo)&(probs<(hi if hi<1 else hi+1e-9))
        if mask.any(): ece+=float(mask.mean()*abs(probs[mask].mean()-obs[mask].mean()))
    terminal=summarize_branches(all_branches)
    report={"policy_version":manifest["policy_version"],"phase":a.phase,"seed_namespace":namespace,"states":len(audit),"branches_per_state":a.branches,"tube_precision":precision,"recoverable_recall":recall,"candidate_mass_coverage":coverage,"brier":brier,"ece_5bin":ece,"false_progress_rate":terminal["false_progress_rate"],"missed_success_rate":terminal["missed_success_rate"],"physical_failure_rate":terminal["physical_failure_rate"],"timeout_rate":terminal["timeout_rate"],"horizon_exhaustion_rate":terminal["horizon_exhaustion_rate"],"terminal_summary":terminal,"dynamics_variants":[dict(spec) for spec in DYNAMICS_VARIANTS],"rows":audit}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps({k:v for k,v in report.items() if k!="rows"},indent=2))
if __name__=="__main__": main()
