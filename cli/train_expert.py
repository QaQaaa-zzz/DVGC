"""Train an owned stage expert to an immutable downstream canonical entry."""
from __future__ import annotations
import argparse,json,math,time
from pathlib import Path
import jax
from dvgc.bank import SnapshotBank
from dvgc.bounded import evaluate_records
from dvgc.config import file_sha256,load_config
from dvgc.curriculum import FLIGHT_RESET_STAGES,select_flight_reset_records
from dvgc.env import OrangeBikeDVGC
from dvgc.expert_training import action_drift,evaluate_flight_composite
from dvgc.experts import StageExpertRegistry
from dvgc.policy import load_bundle,save_bundle
from dvgc.runtime import make_ppo_train_fn,save_json

def resolve_learning_rate(manifest, explicit):
    if explicit is not None: return float(explicit)
    try: return float(manifest["ppo_hyperparameters"]["learning_rate"])
    except (KeyError,TypeError,ValueError) as exc: raise ValueError("Expert continuation requires --learning-rate when the source manifest has no PPO metadata") from exc


def ratios(rows,prefix):
    row=next((r for r in reversed(rows) if any(k.startswith(prefix) for k in r)),{}); values={name:float(row.get(prefix+name,0.0)) for name in ("flight_curriculum","canonical_entry_rehearsal","landing_tube_rehearsal","natural")}; total=sum(values.values()); return {k:(v/total if total else 0.0) for k,v in values.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--stage",choices=["flight"],required=True); p.add_argument("--curriculum",choices=FLIGHT_RESET_STAGES,required=True); p.add_argument("--bank",required=True); p.add_argument("--entry-set",required=True); p.add_argument("--registry",required=True); p.add_argument("--resume",required=True); p.add_argument("--run",required=True); p.add_argument("--config",default="configs/default.json"); p.add_argument("--runtime-gate",default="docs/RUNTIME_GATE.json"); p.add_argument("--seed",type=int,default=0); p.add_argument("--learning-rate",type=float,default=None); p.add_argument("--initial-composite-evaluation",default=""); p.add_argument("--landing-baseline",default=""); a=p.parse_args()
    run=Path(a.run)
    if run.exists(): raise SystemExit(f"Run exists: {run}")
    registry=StageExpertRegistry.load(a.registry); gate=json.loads(Path(a.runtime_gate).read_text()); initial_spec=registry.specs[a.stage]
    if gate.get("status")!="PASS" or gate.get("source_fingerprint")!=registry.runtime_source_fingerprint: raise SystemExit("Expert registry runtime gate is stale")
    if file_sha256(Path(a.resume)/"params.pkl")!=initial_spec.policy_hash: raise SystemExit("Resume policy does not match owned expert registry")
    if file_sha256(a.entry_set)!=initial_spec.downstream_entry_set_sha256: raise SystemExit("Canonical entry set hash mismatch")
    landing_spec=registry.specs["landing"]; landing_hash_before=file_sha256(Path(landing_spec.checkpoint_path)/"params.pkl"); landing_params,landing_cfg,_=load_bundle(landing_spec.checkpoint_path,verify_files=True)
    params,cfg_dict,manifest=load_bundle(a.resume,verify_files=True); learning_rate=resolve_learning_rate(manifest,a.learning_rate)
    cfg=load_config(a.config,{**cfg_dict,"training_stage":a.stage,"expert_chain_termination":True}); bank=SnapshotBank.load(a.bank); records=bank.records_for_phase("flight",include_training_only=False); training_records=select_flight_reset_records(records,a.curriculum); training_bank=SnapshotBank(training_records,bank.metadata); entry=SnapshotBank.load(a.entry_set)
    env=OrangeBikeDVGC(cfg,snapshot_bank=training_bank,cert_bank=entry); eval_env=OrangeBikeDVGC(load_config(a.config,{**cfg.to_dict(),"domain_randomization":False,"obs_noise_enable":False}),snapshot_bank=training_bank,cert_bank=entry)
    run.mkdir(parents=True); blocks=run/"blocks"; blocks.mkdir(); rows=[]; metrics={"status":"preflight","stage":a.stage,"curriculum":a.curriculum,"seed":a.seed,"effective_steps":102400,"block_steps":25600,"entry_set_sha256":file_sha256(a.entry_set),"candidate_bank_sha256":file_sha256(a.bank),"initial_policy_hash":initial_spec.policy_hash,"landing_policy_hash_before":landing_hash_before,"progress":rows}; save_json(run/"training_metrics.json",metrics)
    baseline=json.loads(Path(a.initial_composite_evaluation).read_text()) if a.initial_composite_evaluation else evaluate_flight_composite(params,cfg_dict,landing_params,records,a.entry_set,seed=8100000,controller_stack_hash=initial_spec.controller_stack_hash)
    if baseline.get("candidate_bank_sha256",file_sha256(a.bank))!=file_sha256(a.bank) or baseline.get("entry_set_sha256",file_sha256(a.entry_set))!=file_sha256(a.entry_set): raise SystemExit("Initial composite evaluation provenance mismatch")
    save_json(run/"initial_composite_evaluation.json",baseline)
    if a.landing_baseline: landing_baseline=json.loads(Path(a.landing_baseline).read_text())
    else:
        landing_bank=SnapshotBank.load("artifacts/landing_candidates.pkl"); landing_baseline=evaluate_records(landing_params,landing_cfg,"landing",landing_bank.records_for_phase("landing",include_training_only=False),SnapshotBank(),8200000)
    save_json(run/"frozen_landing_baseline.json",landing_baseline)
    class GateStop(RuntimeError):
        def __init__(self,message,passed): super().__init__(message); self.passed=passed
    pending={}; after_progress=lambda step:None; best_final=float(baseline["composite_final_rate"]); stagnant=0
    def progress(step,values):
        row={"step":int(step),**{k:float(v) for k,v in values.items() if hasattr(v,"__float__")}}; rows.append(row); metrics["status"]="running"; save_json(run/"training_metrics.json",metrics); print(f"[expert] step={step}"); after_progress(int(step))
    def policy_callback(step,make_policy,current):
        if step: pending[int(step)]=current
    def process_block(step,current):
        nonlocal best_final,stagnant
        if step not in (25600,51200,76800,102400): raise GateStop(f"unexpected block step {step}",False)
        index=step//25600; root=blocks/f"block_{index}_{step:06d}"; root.mkdir(); policy=root/"policy"; save_bundle(policy,params=current,config=cfg,xml_path=cfg.xml_path,candidate_bank=a.bank,downstream_bank=a.entry_set,policy_version=f"flight-expert-{a.curriculum}-{step:06d}",extra={"stage":"flight","expert_role":"provisional_support_discovery","seed":a.seed,"cumulative_effective_steps":step,"initial_policy_hash":initial_spec.policy_hash,"ppo_hyperparameters":{"learning_rate":learning_rate,"entropy_cost":.001,"reward_scaling":.1,"discounting":.995,"gae_lambda":.97,"clipping_epsilon":.10,"max_grad_norm":.75,"num_updates_per_batch":2}})
        block_registry=StageExpertRegistry.build({"landing":landing_spec.checkpoint_path,"flight":policy},{"flight":a.entry_set},runtime_source_fingerprint=registry.runtime_source_fingerprint); block_registry.save(root/"expert_registry.json")
        report=evaluate_flight_composite(current,cfg.to_dict(),landing_params,records,a.entry_set,seed=8300000+index*1000,controller_stack_hash=block_registry.specs["flight"].controller_stack_hash); drift=action_drift(current,params,cfg.to_dict(),training_records,8400000+index*1000); landing_hash_after=file_sha256(Path(landing_spec.checkpoint_path)/"params.pkl")
        nonfinite_metrics=[k for row in rows for k,v in row.items() if k!="step" and isinstance(v,float) and not math.isfinite(v)]
        improved=report["composite_final_rate"]>best_final; best_final=max(best_final,report["composite_final_rate"]); stagnant=stagnant+1 if report["chain_rate"]==0 and not improved else 0
        target="descent" if a.curriculum in ("late_descent","descent") else a.curriculum
        if target=="full": target_before={"chain_rate":baseline["chain_rate"],"composite_final_rate":baseline["composite_final_rate"]}; target_after={"chain_rate":report["chain_rate"],"composite_final_rate":report["composite_final_rate"]}
        else: target_before=baseline["subintervals"][target]; target_after=report["subintervals"][target]
        target_improved=target_after["chain_rate"]>target_before["chain_rate"] or target_after["composite_final_rate"]>target_before["composite_final_rate"]
        reasons=[]
        if landing_hash_after!=landing_hash_before: reasons.append("frozen Landing policy hash changed")
        if file_sha256(a.entry_set)!=initial_spec.downstream_entry_set_sha256: reasons.append("canonical entry set hash changed")
        if report["timeout_rate"]>0: reasons.append("composite timeout")
        if any(not math.isfinite(float(v)) for group in (drift["kl"],drift["action_l2"]) for v in group.values()): reasons.append("nonfinite action drift")
        if stagnant>=2: reasons.append("Chain zero and composite Final did not improve for two blocks")
        passed=report["chain_rate"]>0 and report["composite_final_rate"]>.075 and target_improved and report["timeout_rate"]==0 and not reasons
        status="PASS" if passed else ("STOP" if reasons or step==102400 else "CONTINUE")
        payload={"status":status,"block":index,"cumulative_effective_steps":step,"composite":report,"initial_composite_final_rate":baseline["composite_final_rate"],"initial_chain_rate":baseline["chain_rate"],"curriculum_target":target,"curriculum_target_before":target_before,"curriculum_target_after":target_after,"curriculum_target_improved":target_improved,"action_drift":drift,"reset_episode_ratio":ratios(rows,"episode/reset/episode/"),"completed_ppo_transition_ratio":ratios(rows,"episode/reset/transition/"),"landing_policy_hash_before":landing_hash_before,"landing_policy_hash_after":landing_hash_after,"c_l_sha256":file_sha256(a.entry_set),"candidate_bank_sha256":file_sha256(a.bank),"runtime_source_fingerprint":registry.runtime_source_fingerprint,"controller_stack_hash":block_registry.specs["flight"].controller_stack_hash,"health":{"nonfinite_metric_count":len(nonfinite_metrics),"nonfinite_metric_keys":sorted(set(nonfinite_metrics)),"timeout_rate":report["timeout_rate"],"provenance_current":not reasons},"reasons":reasons}; save_json(root/"report.json",payload); metrics.setdefault("blocks",[]).append({"step":step,"status":status,"report":str((root/"report.json").resolve())}); save_json(run/"training_metrics.json",metrics); print(f"[expert-gate] block={index} chain={report['chain_rate']:.4f} final={report['composite_final_rate']:.4f} status={status}")
        if passed: raise GateStop("Flight expert gate passed",True)
        if reasons or step==102400: raise GateStop("; ".join(reasons) or "Flight expert gate not reached",False)
    def after_progress(step):
        if step and step in pending: process_block(step,pending.pop(step))
    train_fn=make_ppo_train_fn(timesteps=102400,episode_length=int(cfg.episode_length),num_envs=160,num_eval_envs=128,num_evals=5,seed=a.seed,learning_rate=learning_rate,entropy_cost=.001,reward_scaling=.1,checkpoint_dir=run/"orbax",unroll_length=32,batch_size=80,num_minibatches=10,num_updates_per_batch=2,discounting=.995,gae_lambda=.97,clipping_epsilon=.10,max_grad_norm=.75,restore_params=params,policy_params_fn=policy_callback,full_reset=True)
    try: train_fn(environment=env,progress_fn=progress,eval_env=eval_env)
    except GateStop as exc:
        metrics.update({"status":"gate_pass" if exc.passed else "gate_stop","message":str(exc),"landing_policy_hash_after":file_sha256(Path(landing_spec.checkpoint_path)/"params.pkl"),"finished_at":time.time()}); save_json(run/"training_metrics.json",metrics); raise SystemExit(0 if exc.passed else 2)

if __name__=="__main__": main()
