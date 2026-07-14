"""Train one backward-bootstrap stage without modifying Tube labels."""
from __future__ import annotations
import argparse, copy, datetime as dt
from pathlib import Path
import jax
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config, save_config
from dvgc.env import OrangeBikeDVGC
from dvgc.policy import load_bundle, save_bundle
from dvgc.runtime import make_ppo_train_fn, save_json, validate_ppo_batch_layout


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True, choices=["landing","flight","takeoff","approach","full"])
    p.add_argument("--bank", required=True)
    p.add_argument("--downstream-bank", default="")
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--run", required=True)
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--num-eval-envs", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default="")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-minibatches", type=int, default=4)
    a=p.parse_args()
    validate_ppo_batch_layout(num_envs=a.num_envs,batch_size=a.batch_size,num_minibatches=a.num_minibatches)
    cfg=load_config(a.config, {"training_stage":a.stage})
    bank=SnapshotBank.load(a.bank); downstream=SnapshotBank.load(a.downstream_bank) if a.downstream_bank else SnapshotBank()
    # Rehearse the already certified downstream phase while extending the same
    # shared Actor backward.  These copies are training-only and can never be
    # certified as current-stage states.
    train_records=[copy.deepcopy(r) for r in bank.records]
    if a.downstream_bank:
        for row in downstream.records:
            if row["final"]["label"] in ("safe","boundary") and not row.get("training_only",False):
                rehearsal=copy.deepcopy(row); rehearsal["source_phase"]=a.stage; rehearsal["training_only"]=True; rehearsal["candidate_kind"]="downstream_rehearsal"; train_records.append(rehearsal)
    training_bank=SnapshotBank(train_records,bank.metadata)
    env=OrangeBikeDVGC(cfg,snapshot_bank=training_bank,cert_bank=downstream)
    eval_cfg=load_config(a.config,{"training_stage":a.stage,"domain_randomization":False,"obs_noise_enable":False})
    eval_env=OrangeBikeDVGC(eval_cfg,snapshot_bank=bank,cert_bank=downstream)
    restore_params=None
    if a.resume:
        restore_params,resume_cfg,resume_manifest=load_bundle(a.resume,verify_files=False)
        if resume_manifest.get("xml_sha256") != file_sha256(cfg.xml_path):
            raise SystemExit("Resume policy was trained with a different XML model")
        if resume_manifest.get("action_mapping_version") != cfg.action_mapping_version:
            raise SystemExit("Resume policy uses a different action mapping")
        if int(resume_cfg.get("actor_history_steps",-1)) != int(cfg.actor_history_steps):
            raise SystemExit("Resume policy uses an incompatible Actor observation history")
    run=Path(a.run); run.mkdir(parents=True,exist_ok=True); save_config(cfg,run/"config.json")
    metrics_path=run/"training_metrics.json"
    train_fn=make_ppo_train_fn(timesteps=a.timesteps,episode_length=int(cfg.episode_length),num_envs=a.num_envs,num_eval_envs=a.num_eval_envs,num_evals=11,seed=a.seed,learning_rate=1e-4,entropy_cost=1e-3,reward_scaling=0.1,checkpoint_dir=run/"orbax",unroll_length=32,batch_size=a.batch_size,num_minibatches=a.num_minibatches,num_updates_per_batch=2,discounting=.995,gae_lambda=.97,clipping_epsilon=.10,max_grad_norm=.75,restore_params=restore_params)
    rows=[]
    metric_log={"stage":a.stage,"seed":a.seed,"requested_timesteps":a.timesteps,"status":"initialized","progress":rows}
    save_json(metrics_path,metric_log)
    def progress(step,metrics):
        row={"step":int(step),**{k:float(v) for k,v in metrics.items() if hasattr(v,"__float__")}}; rows.append(row)
        metric_log["status"]="running"; save_json(metrics_path,metric_log)
        print(f"[train] stage={a.stage} step={step:,}")
    try:
        _,params,final_metrics=train_fn(environment=env,progress_fn=progress,eval_env=eval_env)
    except BaseException as exc:
        metric_log.update({"status":"failed","error_type":type(exc).__name__,"error":str(exc)})
        save_json(metrics_path,metric_log)
        raise
    metric_log.update({"status":"completed","final_metrics":final_metrics})
    save_json(metrics_path,metric_log)
    version=f"{a.stage}-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    save_bundle(run/"policy",params=params,config=cfg,xml_path=cfg.xml_path,candidate_bank=a.bank,downstream_bank=(a.downstream_bank or None),policy_version=version,extra={"stage":a.stage,"seed":a.seed,"timesteps":a.timesteps})
    print(run/"policy")
if __name__=="__main__": main()
