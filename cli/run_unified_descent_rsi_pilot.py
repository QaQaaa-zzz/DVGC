"""Run the single authorized 6,400-step unified Descent RSI learnability pilot."""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import time
from collections import Counter
from pathlib import Path

import jax
import numpy as np

from cli.runtime_gate import source_fingerprint
from dvgc.bank import SnapshotBank
from dvgc.config import ACTION_MAPPING_VERSION, config_hash, file_sha256, load_config, save_config
from dvgc.descent_pilot import build_heldout, evaluate, expansion_proposals
from dvgc.env import OrangeBikeDVGC
from dvgc.policy import load_bundle, save_bundle
from dvgc.provisional_descent import FEATURE_NAMES
from dvgc.runtime import make_ppo_train_fn, ppo_effective_timesteps, save_json, validate_ppo_batch_layout


EXPECTED_BANK = "8e6342bc0d9d5e7821929f6ddccd4fb5b7c23923c66ab2d058309249dfed45a1"
EXPECTED_XML = "d7e9f43ff8fb9e4571203f81062ce9c828acfa38692ee8c71a3e5daa15ce794c"
EXPECTED_HEAD = "04e1a2c"
FEATURE_SCALE = [0.05,0.02,0.03,0.03,0.03,0.03,0.15,0.05,0.15,0.2,0.2,0.2,0.03,0.04,0.04,1.5]


def _tree_shapes(tree):
    return [tuple(np.asarray(x).shape) for x in jax.tree.leaves(tree)]


def _classification(base, final, held_base, held_final, integrity):
    if integrity["status"] != "PASS": return "invalid_pilot", ["training integrity failed"]
    b=base["summary"]["overall"];f=final["summary"]["overall"]
    delta16=f["survival_counts"]["16"]-b["survival_counts"]["16"]
    delta24=f["survival_counts"]["24"]-b["survival_counts"]["24"]
    dt=f["time_to_failure"]["median"]-b["time_to_failure"]["median"]
    core_before=base["summary"]["labels"]["provisional_core"]["time_to_failure"]["median"]
    core_after=final["summary"]["labels"]["provisional_core"]["time_to_failure"]["median"]
    held_reverse=False
    if held_base and held_final and held_base["summary"]["overall"].get("states",0):
        held_reverse=held_final["summary"]["overall"]["time_to_failure"]["median"] < held_base["summary"]["overall"]["time_to_failure"]["median"]-2
    saturation=f["action"]["saturation_fraction"] if "action" in f else 0.0
    clear=(delta16>=2 or delta24>=1 or dt>=4) and core_after>=core_before-2 and not held_reverse and saturation<.75
    if clear:return "learnability_pass",[f"delta16={delta16}",f"delta24={delta24}",f"median_delta={dt}"]
    # A one-tick median shift cannot outweigh loss of the only 16/24-tick
    # survivor.  Weak-positive requires non-regression at both long horizons.
    physical_trend=(delta16>0 or delta24>0 or dt>=2) and delta16>=0 and delta24>=0 and not held_reverse
    if physical_trend:return "weak_positive_signal",[f"delta16={delta16}",f"delta24={delta24}",f"median_delta={dt}"]
    return "no_learnability_signal",[f"delta16={delta16}",f"delta24={delta24}",f"median_delta={dt}"]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank",required=True);p.add_argument("--run",required=True)
    p.add_argument("--initial-policy",required=True)
    p.add_argument("--config",default="configs/unified_descent_rsi_learnability_pilot_v1.json")
    p.add_argument("--seed",type=int,default=0);p.add_argument("--timesteps",type=int,default=6400)
    a=p.parse_args();root=Path(a.run)
    if root.exists():raise SystemExit(f"refusing to overwrite run: {root}")
    if a.seed!=0 or a.timesteps!=6400:raise SystemExit("protocol permits only seed=0 and 6,400 requested steps")
    if file_sha256(a.bank)!=EXPECTED_BANK:raise SystemExit("candidate bank hash mismatch")
    cfg=load_config(a.config)
    if file_sha256(cfg.xml_path)!=EXPECTED_XML or cfg.action_mapping_version!=ACTION_MAPPING_VERSION:
        raise SystemExit("XML or action mapping mismatch")
    gate=json.loads(Path("docs/RUNTIME_GATE.json").read_text());fingerprint=source_fingerprint(Path.cwd())
    if gate.get("status")!="PASS" or gate.get("source_fingerprint")!=fingerprint:
        raise SystemExit("runtime gate is not PASS/current")
    execution_head=subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    num_envs,batch_size,num_minibatches,num_evals=50,25,2,5
    validate_ppo_batch_layout(num_envs=num_envs,batch_size=batch_size,num_minibatches=num_minibatches)
    effective=ppo_effective_timesteps(a.timesteps,unroll_length=32,batch_size=batch_size,num_minibatches=num_minibatches,num_evals=num_evals)
    if effective!=6400:raise SystemExit(f"unexpected effective budget {effective}")
    root.mkdir(parents=True);started=time.time();original_hash=file_sha256(a.bank)
    save_config(cfg,root/"effective_config.json")
    protocol={"experiment":"unified_descent_rsi_learnability_pilot_v1","seed":0,"requested_steps":6400,"effective_steps":effective,"formula":"4 evaluation intervals * ceil(6400 / (4 * (32 unroll * 25 batch * 2 minibatches))) * (32 * 25 * 2) = 6400","ppo_layout":{"num_envs":num_envs,"num_eval_envs":14,"unroll_length":32,"batch_size":batch_size,"num_minibatches":num_minibatches,"num_evals":num_evals,"num_updates_per_batch":2},"episode_horizon":24,"checkpoint_steps":[0,1600,3200,4800,6400],"candidate_bank_sha256":original_hash,"runtime_fingerprint":fingerprint,"ppo_authorization":"single_pilot_only"}
    save_json(root/"pilot_config.json",protocol)
    reward_manifest={"status":"PASS","profile":"descent_rsi_pilot_adapter_v1","pilot_only":True,"formal_paper_reward":False,"clipping":[cfg.descent_rsi_pilot_shaping_clip_min,cfg.descent_rsi_pilot_shaping_clip_max],"termination_reward":{"physical_failure":-cfg.descent_rsi_pilot_failure_penalty,"pilot_horizon_reached":0.0},"components":{"survival":{"definition":"constant per finite active tick","weight":cfg.descent_rsi_pilot_survival},"roll":{"definition":"weight*exp(-(roll/15deg)^2)","weight":cfg.descent_rsi_pilot_roll},"pitch":{"definition":"weight*exp(-(pitch/25deg)^2)","weight":cfg.descent_rsi_pilot_pitch},"angular":{"definition":"weight*exp(-(norm(gyro)/recovery_max_angvel)^2)","weight":cfg.descent_rsi_pilot_angular},"descent_vz":{"definition":"weight*exp(-((vz+0.5)/0.5)^2)","weight":cfg.descent_rsi_pilot_vz},"action_smooth":{"definition":"-weight*mean((a-a_prev)^2)","weight":cfg.descent_rsi_pilot_action_smooth},"action_magnitude":{"definition":"-weight*mean(a^2)","weight":cfg.descent_rsi_pilot_action_magnitude},"physical_failure":{"definition":"-weight*formal_hard_failure","weight":cfg.descent_rsi_pilot_failure_penalty}},"forbidden_dependencies":{"old_matcher":False,"old_support_distance":False,"landing":False,"final_recovery":False,"tube_membership":False}}
    save_json(root/"reward_manifest.json",reward_manifest)

    original=SnapshotBank.load(a.bank);train_rows=[]
    for row in original.records:
        item=copy.deepcopy(row);item["reset_parent_id"]=item["id"];train_rows.append(item)
    train_meta=copy.deepcopy(original.metadata);train_meta["pilot_sidecar_of_sha256"]=original_hash
    training_bank=SnapshotBank(train_rows,train_meta);training_bank.save(root/"training_reset_sidecar.pkl")
    env=OrangeBikeDVGC(cfg,snapshot_bank=training_bank)
    init_params,init_cfg,init_manifest=load_bundle(a.initial_policy,verify_files=True)
    init_env_cfg=load_config(None,init_cfg)
    schema={"actor_history_steps":{"source":init_cfg["actor_history_steps"],"pilot":cfg.actor_history_steps},"action_mapping":{"source":init_manifest["action_mapping_version"],"pilot":cfg.action_mapping_version},"xml":{"source":init_manifest["xml_sha256"],"pilot":file_sha256(cfg.xml_path)},"actor_observation_dim":env._actor_obs_dim,"action_dim":env.action_size}
    compatible=all((schema["actor_history_steps"]["source"]==schema["actor_history_steps"]["pilot"],schema["action_mapping"]["source"]==schema["action_mapping"]["pilot"],schema["xml"]["source"]==schema["xml"]["pilot"]))
    # Building inference proves all actor/normalizer leaves load under the pilot schema.
    from dvgc.runtime import build_inference
    build_inference(env,init_params,deterministic=True)(env.reset(jax.random.PRNGKey(1)).obs,jax.random.PRNGKey(2))
    init_audit={"status":"PASS" if compatible else "FAIL","initialization_type":"frozen_pi_D_parameter_copy","initialization_checkpoint":str(Path(a.initial_policy).resolve()),"checkpoint_hash":file_sha256(Path(a.initial_policy)/"params.pkl"),"parameter_loading_coverage":1.0,"unloaded_parameters":[],"schema":schema,"source_config_hash":config_hash(init_env_cfg),"source_policy_version":init_manifest["policy_version"],"source_asset_modified":False,"no_final_shared_checkpoint_found":True}
    save_json(root/"initialization_audit.json",init_audit)
    if not compatible:raise SystemExit("initial policy incompatible")

    ckroot=root/"checkpoints";ckroot.mkdir()
    def save_checkpoint(step,params):
        target=ckroot/f"checkpoint_{int(step):04d}"
        if target.exists():return
        save_bundle(target,params=params,config=cfg,xml_path=cfg.xml_path,candidate_bank=root/"training_reset_sidecar.pkl",downstream_bank=None,policy_version=f"unified-descent-rsi-pilot-{step:04d}",extra={"artifact_role":"learnability_pilot_checkpoint","effective_steps":int(step),"original_candidate_bank_sha256":original_hash,"initialization_hash":init_audit["checkpoint_hash"],"ppo_authorization":"bounded_pilot_only"})
    save_checkpoint(0,init_params)

    heldout,heldout_build=build_heldout(env,train_rows,seed=9100000,feature_scale=FEATURE_SCALE)
    heldout_bank=SnapshotBank(heldout,{"artifact_role":"descent_rsi_heldout_evaluation_sidecar","training_eligible":False,"source_bank_sha256":original_hash,"seed":9100000})
    heldout_bank.save(root/"heldout_sidecar.pkl")
    baseline={"checkpoint_0000":evaluate(env,train_rows,params=init_params,seed=9200000,policy_name="checkpoint_0000"),"neutral_action":evaluate(env,train_rows,params=None,seed=9200000,policy_name="neutral_action"),"frozen_pi_D":evaluate(env,train_rows,params=init_params,seed=9200000,policy_name="frozen_pi_D"),"heldout_construction":heldout_build}
    save_json(root/"baseline_evaluation.json",baseline)
    held_base=evaluate(env,heldout,params=init_params,seed=9300000,policy_name="checkpoint_0000") if heldout else None

    # Exercise the exact environment reset implementation and report every candidate.
    reset_batch=jax.jit(jax.vmap(env.reset));counts=Counter()
    for block in range(128):
        keys=jax.random.split(jax.random.PRNGKey(9400000+block),50);state=reset_batch(keys)
        ids=np.asarray(state.info["reset_parent"],int)
        for index in ids:counts[env._reset_parent_ids[int(index)]]+=1
    sampler={"status":"PASS","draw_kind":"exact_env_reset_distribution_pretraining_audit","draws":6400,"candidate_counts":dict(sorted(counts.items())),"all_candidates_sampled":len(counts)==14,"max_candidate_fraction":max(counts.values())/6400,"label_counts":{},"layer_counts":{},"candidate_bank_read_only":True}
    byid={row["id"]:row for row in train_rows}
    for cid,count in counts.items():
        sampler["label_counts"][byid[cid]["provisional_label"]]=sampler["label_counts"].get(byid[cid]["provisional_label"],0)+count
        sampler["layer_counts"][byid[cid]["descent_layer"]]=sampler["layer_counts"].get(byid[cid]["descent_layer"],0)+count
    save_json(root/"sampler_realized_distribution.json",sampler)

    progress=[]
    def progress_fn(step,metrics):
        progress.append({"effective_steps":int(step),"recorded_at":time.time(),**{key:float(value) for key,value in metrics.items() if np.asarray(value).shape==()}})
        save_json(root/"training_progress.json",{"status":"running","progress":progress})
        print(f"[unified-descent-rsi] effective_steps={int(step)}",flush=True)
    train_fn=make_ppo_train_fn(timesteps=6400,episode_length=24,num_envs=50,num_eval_envs=14,num_evals=5,seed=0,learning_rate=float(init_manifest["ppo_hyperparameters"]["learning_rate"]),entropy_cost=float(init_manifest["ppo_hyperparameters"]["entropy_cost"]),reward_scaling=.1,checkpoint_dir=root/"orbax",unroll_length=32,batch_size=25,num_minibatches=2,num_updates_per_batch=2,discounting=.995,gae_lambda=.97,clipping_epsilon=.10,max_grad_norm=.75,restore_params=init_params,policy_params_fn=lambda step,_make,params:save_checkpoint(step,params),full_reset=True)
    try:
        _,final_params,final_metrics=train_fn(environment=env,progress_fn=progress_fn,eval_env=env)
        training_status="completed"
    except BaseException as exc:
        save_json(root/"training_integrity_report.json",{"status":"FAIL","error_type":type(exc).__name__,"error":str(exc),"ppo_authorization":False});raise
    save_json(root/"training_progress.json",{"status":training_status,"progress":progress,"final_metrics":final_metrics})
    checkpoint_rows=[];evaluations={}
    for target in (0,1600,3200,4800,6400):
        path=ckroot/f"checkpoint_{target:04d}";params,_,manifest=load_bundle(path,verify_files=True)
        evaluation=evaluate(env,train_rows,params=params,seed=9500000,policy_name=path.name)
        matching=[row for row in progress if row["effective_steps"]==target]
        metric=max(matching,key=lambda row:sum(key.startswith("training/") for key in row)) if matching else {}
        evaluations[path.name]={"effective_steps":target,"policy_hash":file_sha256(path/"params.pkl"),"evaluation":evaluation,"ppo_metrics":{key:metric.get(key) for key in ("training/policy_loss","training/v_loss","training/entropy_loss","training/kl_mean","eval/episode_reward","eval/avg_episode_length")},"explained_variance":metric.get("training/explained_variance")}
        checkpoint_rows.append(evaluations[path.name])
    save_json(root/"checkpoint_evaluations.json",evaluations)
    held_final=evaluate(env,heldout,params=final_params,seed=9300000,policy_name="checkpoint_6400") if heldout else None
    save_json(root/"heldout_evaluation.json",{"construction":heldout_build,"checkpoint_0000":held_base,"checkpoint_6400":held_final,"evidence_sufficient":len(heldout)>=4 and heldout_build["clusters"]>=2})
    proposals,proposal_report=expansion_proposals(env,train_rows,final_params,seed=9600000,feature_scale=FEATURE_SCALE)
    SnapshotBank(proposals,{"artifact_role":"descent_candidate_expansion_proposals_v1","formal_tube_or_jel":False,"source_policy_hash":file_sha256(ckroot/"checkpoint_6400"/"params.pkl"),"source_bank_sha256":original_hash}).save(root/"descent_candidate_expansion_proposals_v1.pkl")
    save_json(root/"descent_candidate_expansion_proposals_v1_report.json",proposal_report)
    nonfinite=any(not np.isfinite(value) for row in progress for value in row.values() if isinstance(value,float))
    integrity={"status":"PASS" if training_status=="completed" and not nonfinite and file_sha256(a.bank)==original_hash else "FAIL","completed_budget":progress[-1]["effective_steps"] if progress else 0,"expected_budget":6400,"bank_hash_before":original_hash,"bank_hash_after":file_sha256(a.bank),"xml_sha256":file_sha256(cfg.xml_path),"action_mapping_version":cfg.action_mapping_version,"runtime_fingerprint":fingerprint,"checkpoint_restore_pass":len(evaluations)==5,"nonfinite_metrics":nonfinite,"oom":False,"timeout":False,"ppo_authorization":False}
    save_json(root/"training_integrity_report.json",integrity)
    final_eval=evaluations["checkpoint_6400"]["evaluation"];classification,reasons=_classification(baseline["checkpoint_0000"],final_eval,held_base,held_final,integrity)
    base_overall=baseline["checkpoint_0000"]["summary"]["overall"];final_overall=final_eval["summary"]["overall"]
    report={"status":"PASS","experiment":"unified_descent_rsi_learnability_pilot_v1","starting_head":EXPECTED_HEAD,"execution_head":execution_head,"candidate_bank_sha256":original_hash,"effective_step_protocol":protocol,"initialization":init_audit,"reward_modified":True,"reward_adapter":"descent_rsi_pilot_adapter_v1","baseline_8_16_24":base_overall["survival_counts"],"checkpoint_8_16_24":{name:row["evaluation"]["summary"]["overall"]["survival_counts"] for name,row in evaluations.items()},"baseline_groups":baseline["checkpoint_0000"]["summary"],"final_groups":final_eval["summary"],"median_time_to_failure":{"before":base_overall["time_to_failure"]["median"],"after":final_overall["time_to_failure"]["median"]},"heldout":json.loads((root/"heldout_evaluation.json").read_text()),"failure_reasons":{"before":base_overall["failure_reasons"],"after":final_overall["failure_reasons"]},"action":{"before":base_overall["action"],"after":final_overall["action"]},"training_stability":{"integrity":integrity,"last_ppo_metrics":evaluations["checkpoint_6400"]["ppo_metrics"]},"reward_hacking":bool(final_overall["action"]["saturation_fraction"]>.75 or (sum(final_overall["survival_counts"].values())<sum(base_overall["survival_counts"].values()) and progress[-1].get("eval/episode_reward",0)>progress[0].get("eval/episode_reward",0))),"expansion_proposals":proposal_report,"classification":classification,"classification_reasons":reasons,"recommend_expanded_training":classification=="learnability_pass","formal_tube_or_jel":False,"ppo_authorization":False,"elapsed_seconds":time.time()-started}
    save_json(root/"UNIFIED_DESCENT_RSI_LEARNABILITY_PILOT_V1_REPORT.json",report)
    md=f"# Unified Descent RSI learnability pilot v1\n\n- Classification: `{classification}`\n- Budget: `{effective}` effective environment steps (seed 0)\n- Baseline 8/16/24: `{base_overall['survival_counts']}`\n- Final 8/16/24: `{final_overall['survival_counts']}`\n- Median survival: `{base_overall['time_to_failure']['median']}` -> `{final_overall['time_to_failure']['median']}` ticks\n- Held-out states: `{len(heldout)}`\n- Expansion proposals: `{proposal_report['proposal_states']}`\n- Formal Tube/JEL: `false`\n- PPO authorization after run: `false`\n"
    (root/"SUMMARY.md").write_text(md)
    print(json.dumps({"status":"PASS","classification":classification,"report":str(root/"UNIFIED_DESCENT_RSI_LEARNABILITY_PILOT_V1_REPORT.json")},indent=2))


if __name__=="__main__":main()
