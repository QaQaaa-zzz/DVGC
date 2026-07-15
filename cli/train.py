"""Train one backward-bootstrap stage without modifying Tube labels."""
from __future__ import annotations
import argparse, copy, datetime as dt, time
from pathlib import Path
import jax
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config, save_config
from dvgc.curriculum import FLIGHT_RESET_STAGES, select_flight_reset_records
from dvgc.env import OrangeBikeDVGC
from dvgc.policy import load_bundle, save_bundle
from dvgc.runtime import make_ppo_train_fn, ppo_effective_timesteps, save_json, validate_ppo_batch_layout


LANDING_ENTROPY_COST = 1e-4
DEFAULT_ENTROPY_COST = 1e-3


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True, choices=["landing","flight","takeoff","approach","full"])
    p.add_argument("--bank", required=True)
    p.add_argument("--downstream-bank", default="")
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--run", required=True)
    p.add_argument("--timesteps", type=int, default=1_000_000)
    # The authoritative Warp model exceeds its aggregate contact capacity at
    # 1024 parallel Landing environments even when each individual state is
    # valid.  This 320 x (80 * 4) layout is the validated formal default.
    p.add_argument("--num-envs", type=int, default=320)
    p.add_argument("--num-eval-envs", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default="")
    p.add_argument("--require-final-safe-rsi", action="store_true")
    p.add_argument("--batch-size", type=int, default=80)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--entropy-cost", type=float, default=None)
    p.add_argument("--flight-reset-stage", choices=FLIGHT_RESET_STAGES, default="full")
    p.add_argument("--downstream-rehearsal-mass", type=float, default=None)
    a=p.parse_args()
    run=Path(a.run)
    if run.exists(): raise SystemExit(f"Run directory already exists: {run}")
    validate_ppo_batch_layout(num_envs=a.num_envs,batch_size=a.batch_size,num_minibatches=a.num_minibatches)
    effective_timesteps=ppo_effective_timesteps(
        a.timesteps,unroll_length=32,batch_size=a.batch_size,
        num_minibatches=a.num_minibatches,num_evals=11,
    )
    overrides={"training_stage":a.stage}
    if a.downstream_rehearsal_mass is not None:
        if not 0.0<=a.downstream_rehearsal_mass<1.0: raise SystemExit("--downstream-rehearsal-mass must be in [0,1)")
        overrides["downstream_rehearsal_mass"]=a.downstream_rehearsal_mass
    cfg=load_config(a.config, overrides)
    bank=SnapshotBank.load(a.bank); downstream=SnapshotBank.load(a.downstream_bank) if a.downstream_bank else SnapshotBank()
    restore_params=None; resume_manifest={}
    if a.resume:
        restore_params,resume_cfg,resume_manifest=load_bundle(a.resume,verify_files=True)
        if resume_manifest.get("xml_sha256") != file_sha256(cfg.xml_path):
            raise SystemExit("Resume policy was trained with a different XML model")
        if resume_manifest.get("action_mapping_version") != cfg.action_mapping_version:
            raise SystemExit("Resume policy uses a different action mapping")
        if int(resume_cfg.get("actor_history_steps",-1)) != int(cfg.actor_history_steps):
            raise SystemExit("Resume policy uses an incompatible Actor observation history")

    certified_rows=[r for r in bank.records_for_phase(a.stage,include_training_only=False) if r["final"]["branches"]>0] if a.stage!="full" else []
    safe_count=sum(r["final"]["label"]=="safe" for r in certified_rows)
    boundary_count=sum(r["final"]["label"]=="boundary" for r in certified_rows)
    if certified_rows:
        if not a.resume: raise SystemExit("Policy-conditioned Tube resets require --resume with the certified policy")
        try: bank.validate_certification_provenance(a.stage,policy_version=resume_manifest["policy_version"],estimator_version=resume_manifest.get("estimator_version","event_filter_v1"))
        except ValueError as exc: raise SystemExit(str(exc)) from exc
    if a.require_final_safe_rsi:
        if a.stage=="full": raise SystemExit("--require-final-safe-rsi is only valid for backward-bootstrap phases")
        if safe_count<int(cfg.tube_activation_min_safe):
            raise SystemExit(f"Final-safe RSI requires at least {int(cfg.tube_activation_min_safe)} safe {a.stage} records, found {safe_count}")
        reset_mode="final_safe_boundary_tube_rsi"
    elif certified_rows:
        reset_mode="certified_bank_fallback" if safe_count<int(cfg.tube_activation_min_safe) else "final_safe_boundary_tube_rsi"
    else:
        reset_mode="geometric_bootstrap"
    reset_protocol={"mode":reset_mode,"bank":str(Path(a.bank).resolve()),"certified_records":len(certified_rows),"final_safe_records":safe_count,"boundary_records":boundary_count,"initial_policy_version":resume_manifest.get("policy_version")}
    # Rehearse the already certified downstream phase while extending the same
    # shared Actor backward.  These copies are training-only and can never be
    # certified as current-stage states.
    current_records=bank.records
    if a.stage=="flight":
        current_records=select_flight_reset_records(bank.records_for_phase("flight",include_training_only=False),a.flight_reset_stage)
    elif a.flight_reset_stage!="full":
        raise SystemExit("--flight-reset-stage is only valid for Flight")
    train_records=[copy.deepcopy(r) for r in current_records]
    if a.downstream_bank:
        for row in downstream.records:
            if row["final"]["label"] in ("safe","boundary") and not row.get("training_only",False):
                rehearsal=copy.deepcopy(row); rehearsal["source_phase"]=a.stage; rehearsal["training_only"]=True; rehearsal["candidate_kind"]="downstream_rehearsal"; train_records.append(rehearsal)
    reset_protocol.update({"flight_reset_stage":a.flight_reset_stage if a.stage=="flight" else None,"current_stage_reset_records":len(current_records),"full_candidate_records":len(bank.records_for_phase(a.stage,include_training_only=False)) if a.stage!="full" else 0,"downstream_rehearsal_mass":float(cfg.downstream_rehearsal_mass)})
    training_bank=SnapshotBank(train_records,bank.metadata)
    env=OrangeBikeDVGC(cfg,snapshot_bank=training_bank,cert_bank=downstream)
    eval_cfg=load_config(a.config,{"training_stage":a.stage,"domain_randomization":False,"obs_noise_enable":False})
    eval_env=OrangeBikeDVGC(eval_cfg,snapshot_bank=bank,cert_bank=downstream)
    run.mkdir(parents=True,exist_ok=False); save_config(cfg,run/"config.json")
    metrics_path=run/"training_metrics.json"
    learning_rate=float(a.learning_rate)
    entropy_cost=(LANDING_ENTROPY_COST if a.stage=="landing" else DEFAULT_ENTROPY_COST) if a.entropy_cost is None else float(a.entropy_cost)
    if learning_rate<=0: raise SystemExit("--learning-rate must be positive")
    if entropy_cost<0: raise SystemExit("--entropy-cost must be non-negative")
    train_fn=make_ppo_train_fn(timesteps=a.timesteps,episode_length=int(cfg.episode_length),num_envs=a.num_envs,num_eval_envs=a.num_eval_envs,num_evals=11,seed=a.seed,learning_rate=learning_rate,entropy_cost=entropy_cost,reward_scaling=0.1,checkpoint_dir=run/"orbax",unroll_length=32,batch_size=a.batch_size,num_minibatches=a.num_minibatches,num_updates_per_batch=2,discounting=.995,gae_lambda=.97,clipping_epsilon=.10,max_grad_norm=.75,restore_params=restore_params)
    rows=[]
    started_at=time.time()
    metric_log={"stage":a.stage,"seed":a.seed,"requested_timesteps":a.timesteps,"effective_timesteps":effective_timesteps,"ppo_layout":{"num_envs":a.num_envs,"num_eval_envs":a.num_eval_envs,"unroll_length":32,"batch_size":a.batch_size,"num_minibatches":a.num_minibatches,"num_evals":11},"ppo_hyperparameters":{"learning_rate":learning_rate,"entropy_cost":entropy_cost,"reward_scaling":0.1,"num_updates_per_batch":2,"discounting":.995,"gae_lambda":.97,"clipping_epsilon":.10,"max_grad_norm":.75},"reset_protocol":reset_protocol,"status":"initialized","started_at":started_at,"progress":rows}
    save_json(metrics_path,metric_log)
    def progress(step,metrics):
        row={"step":int(step),"recorded_at":time.time(),**{k:float(v) for k,v in metrics.items() if hasattr(v,"__float__")}}; rows.append(row)
        metric_log["status"]="running"; save_json(metrics_path,metric_log)
        print(f"[train] stage={a.stage} step={step:,}")
    try:
        _,params,final_metrics=train_fn(environment=env,progress_fn=progress,eval_env=eval_env)
    except BaseException as exc:
        metric_log.update({"status":"failed","error_type":type(exc).__name__,"error":str(exc),"finished_at":time.time(),"elapsed_seconds":time.time()-started_at})
        save_json(metrics_path,metric_log)
        raise
    metric_log.update({"status":"completed","final_metrics":final_metrics,"finished_at":time.time(),"elapsed_seconds":time.time()-started_at})
    save_json(metrics_path,metric_log)
    version=f"{a.stage}-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    save_bundle(run/"policy",params=params,config=cfg,xml_path=cfg.xml_path,candidate_bank=a.bank,downstream_bank=(a.downstream_bank or None),policy_version=version,extra={"stage":a.stage,"seed":a.seed,"timesteps":a.timesteps,"reset_protocol":reset_protocol,"ppo_hyperparameters":metric_log["ppo_hyperparameters"]})
    print(run/"policy")
if __name__=="__main__": main()
