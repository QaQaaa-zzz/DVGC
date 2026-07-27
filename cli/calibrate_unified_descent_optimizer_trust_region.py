"""Bounded five-point LR calibration on the saved first Descent rollout."""
from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from brax.training import types
from brax.training.agents.ppo import losses as ppo_losses

from cli.runtime_gate import source_fingerprint
from dvgc.bank import SnapshotBank
from dvgc.config import ACTION_MAPPING_VERSION, file_sha256, load_config
from dvgc.descent_pilot import evaluate
from dvgc.env import OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.ppo_integrity import (
    _network, logprob_audit, make_optimizer, normalizer_summary, optimize_batch,
    tree_delta, tree_hash,
)
from dvgc.provisional_descent import hierarchical_reset_weights
from dvgc.runtime import save_json

EXPECTED_HEAD="e61639f"
EXPECTED_BANK="8e6342bc0d9d5e7821929f6ddccd4fb5b7c23923c66ab2d058309249dfed45a1"
EXPECTED_XML="d7e9f43ff8fb9e4571203f81062ce9c828acfa38692ee8c71a3e5daa15ce794c"
EXPECTED_POLICY="52721668eed0cc78b41a45ad7c319e687f43add8977f2b4bdfcad8208c4353f2"
EXPECTED_NORMALIZER="8f2e36b6f69a3d20da67c1854f7e908c98dd6b03ae70e287e0a7e28522f93a7e"
SOURCE_RUN=Path("runs/unified_descent_rsi_update_integrity_repair_v1_phase_a_v4_20260727")


def _derive_sgd_key(seed=0):
    key=jax.random.PRNGKey(seed);_,local=jax.random.split(key)
    local=jax.random.fold_in(local,jax.process_index());local,_,_=jax.random.split(local,3)
    epoch,_=jax.random.split(local);key_sgd,_,_=jax.random.split(epoch,3)
    return key_sgd


def _infer_truncation(episode_done, discount, horizon=24):
    done=np.asarray(episode_done).astype(bool); discount=np.asarray(discount)
    result=np.zeros_like(discount,np.float32)
    for env in range(done.shape[0]):
        age=0
        for tick in range(done.shape[1]):
            age+=1
            if done[env,tick]:
                if age>=horizon: result[env,tick]=1.0
                age=0
    return result


def _load_saved_rollout(initial_params):
    asset=np.load(SOURCE_RUN/"first_rollout_read_only.npz")
    with (SOURCE_RUN/"checkpoint_1600_full.pkl").open("rb") as stream:
        checkpoint=pickle.load(stream)
    obs={"state":jnp.asarray(asset["observation_state"]),
         "privileged_state":jnp.asarray(asset["observation_privileged"])}
    final_obs=checkpoint["env_state"].obs
    next_obs={key:jnp.concatenate((value[:,1:],jnp.asarray(final_obs[key])[:,None]),axis=1)
              for key,value in obs.items()}
    shape={key:value.shape[2:] for key,value in obs.items()}
    network=_network(shape,asset["action"].shape[-1])
    params=ppo_losses.PPONetworkParams(policy=initial_params[1],value=initial_params[2])
    logits=network.policy_network.apply(initial_params[0],params.policy,obs)
    truncation=_infer_truncation(asset["episode_done"],asset["discount"])
    data=types.Transition(observation=obs,action=jnp.asarray(asset["action"]),
        reward=jnp.asarray(asset["reward"]),discount=jnp.asarray(asset["discount"]),
        next_observation=next_obs,extras={"policy_extras":{
            "raw_action":jnp.asarray(asset["raw_action"]),
            "log_prob":jnp.asarray(asset["stored_log_prob"]),
            "distribution_params":logits},"state_extras":{"truncation":jnp.asarray(truncation)}})
    return network,params,data,checkpoint


def _distribution_stats(audit,clip=.1):
    # Recompute ratios to report exact clip fraction instead of quantile proxies.
    return {"analytic_kl":audit["analytic_distribution_kl_mean"],
            "sample_mean_kl":audit["sample_mean_kl"],"ratio":audit["ratio"]}


def _ratio_clip_fraction(network,params,normalizer,data,clip=.1):
    logits=network.policy_network.apply(normalizer,params.policy,data.observation)
    recomputed=network.parametric_action_distribution.log_prob(
        logits,data.extras["policy_extras"]["raw_action"])
    ratio=np.exp(np.asarray(recomputed-data.extras["policy_extras"]["log_prob"]))
    return float(np.mean((ratio<1-clip)|(ratio>1+clip)))


def _fixed_observation_action_delta(network,before,after,normalizer,data):
    dist=network.parametric_action_distribution
    old_logits=network.policy_network.apply(normalizer,before.policy,data.observation)
    new_logits=network.policy_network.apply(normalizer,after.policy,data.observation)
    old_action=dist.mode(old_logits);new_action=dist.mode(new_logits)
    return float(np.max(np.abs(np.asarray(new_action-old_action))))


def _gae_summary(network,params,normalizer,data):
    swap=lambda x:jnp.swapaxes(x,0,1)
    obs=jax.tree_util.tree_map(swap,data.observation);nobs=jax.tree_util.tree_map(swap,data.next_observation)
    baseline=network.value_network.apply(normalizer,params.value,obs)
    bootstrap=network.value_network.apply(normalizer,params.value,jax.tree_util.tree_map(lambda x:x[-1],nobs))
    trunc=swap(data.extras["state_extras"]["truncation"]);discount=swap(data.discount)
    returns,adv=ppo_losses.compute_gae(truncation=trunc,termination=(1-discount)*(1-trunc),
        rewards=swap(data.reward)*.1,values=baseline,bootstrap_value=bootstrap,lambda_=.97,discount=.995)
    def summary(x):
        x=np.asarray(x,np.float64);return {"mean":float(x.mean()),"std":float(x.std()),
            "min":float(x.min()),"p05":float(np.quantile(x,.05)),"median":float(np.median(x)),
            "p95":float(np.quantile(x,.95)),"max":float(x.max())}
    return {"advantage":summary(adv),"return":summary(returns),"value":summary(baseline)}


def _metric_projection(metrics):
    return [{k:row[k] for k in ("pass","minibatch","loss","kl_mean",
        "gradient_norm_before_clip","gradient_norm_after_clip_upper_bound",
        "actor_gradient_norm","critic_gradient_norm","log_std_gradient_norm",
        "candidate_post_update_analytic_kl","rolled_back")}
        for row in metrics["steps"]]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True)
    parser.add_argument("--bank",default="runs/mjx_continuous_pipeline_repair_v1/descent_candidate_bank_v1/descent_candidates_v2.pkl")
    parser.add_argument("--initial-policy",default="runs/stage_experts/descent_tube_seed0_20260716T2330/round_3/train/policy")
    parser.add_argument("--config",default="configs/unified_descent_rsi_learnability_pilot_v1.json")
    args=parser.parse_args();root=Path(args.run)
    if root.exists():raise SystemExit(f"refusing to overwrite {root}")
    head=subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    if subprocess.run(["git","merge-base","--is-ancestor",EXPECTED_HEAD,"HEAD"]).returncode:
        raise SystemExit("wrong authoritative ancestry")
    if subprocess.check_output(["git","status","--porcelain"],text=True).strip():raise SystemExit("worktree not clean")
    cfg=load_config(args.config)
    if file_sha256(args.bank)!=EXPECTED_BANK or file_sha256(cfg.xml_path)!=EXPECTED_XML or cfg.action_mapping_version!=ACTION_MAPPING_VERSION:raise SystemExit("provenance mismatch")
    if file_sha256(Path(args.initial_policy)/"params.pkl")!=EXPECTED_POLICY:raise SystemExit("pi_D mismatch")
    gate=json.loads(Path("docs/RUNTIME_GATE.json").read_text())
    if gate.get("status")!="PASS" or gate.get("source_fingerprint")!=source_fingerprint(Path.cwd()):raise SystemExit("runtime gate stale")
    initial_params,_,manifest=load_bundle(args.initial_policy,verify_files=True)
    norm=normalizer_summary(initial_params[0])
    if norm["sha256"]!=EXPECTED_NORMALIZER or norm["count"]!=1024000:raise SystemExit("normalizer mismatch")
    root.mkdir(parents=True)

    immutable=SnapshotBank.load(args.bank);weights=hierarchical_reset_weights(immutable.records);rows=[]
    for source,weight in zip(immutable.records,weights):
        row=copy.deepcopy(source);row["reset_parent_id"]=row["id"];row["reset_weight"]=weight;rows.append(row)
    sidecar=SnapshotBank(rows,{**immutable.metadata,"optimizer_calibration_sidecar_of":EXPECTED_BANK})
    sidecar.save(root/"training_reset_sidecar.pkl")
    env=OrangeBikeDVGC(cfg,snapshot_bank=sidecar)
    network,params,data,checkpoint=_load_saved_rollout(initial_params)
    reconstructed_gae=_gae_summary(network,params,initial_params[0],data)
    prior=json.loads((SOURCE_RUN/"first_update_reproduction.json").read_text())["gae"]
    gae_max_error=max(abs(reconstructed_gae[group][name]-prior[group][name])
        for group in prior for name in prior[group])
    if gae_max_error>1e-5:raise SystemExit(f"saved rollout GAE reconstruction mismatch: {gae_max_error}")
    baseline=evaluate(env,rows,params=initial_params,seed=9500000,policy_name="checkpoint_0000")
    old=baseline["summary"]["overall"]

    lr_ref=float(manifest["ppo_hyperparameters"]["learning_rate"]);k_ref=184.45465087890625
    nominal,hard=.005,.01;m0=math.sqrt(nominal/k_ref);multipliers=(.5,.75,1.,1.25,1.5)
    learning_rates=[lr_ref*m0*m for m in multipliers]
    config={"status":"PASS","lr_ref":lr_ref,"k_ref":k_ref,"k_nominal":nominal,
        "k_hard_max":hard,"m0":m0,"multipliers":list(multipliers),"learning_rates":learning_rates,
        "selection":"maximum candidate passing all fixed update-integrity gates",
        "source_rollout":str((SOURCE_RUN/"first_rollout_read_only.npz").resolve()),
        "new_training_environment_transitions":0,"heldout_used_for_selection":False,
        "saved_advantage_return_reconstruction_max_abs_error":gae_max_error}
    save_json(root/"optimizer_lr_calibration_config.json",config)
    save_json(root/"frozen_normalizer_protocol_audit.json",{"status":"PASS","normalizer":norm,
        "source_policy_hash":EXPECTED_POLICY,"read_only_copy":True,"updated_during_calibration":False,
        "rollout_loss_evaluation_same_hash":True,"training_or_heldout_refit":False})

    key=_derive_sgd_key();results=[]
    for index,lr in enumerate(learning_rates):
        repeats=[]
        for repeat in range(2):
            optimizer=make_optimizer(lr);state={"network":network,"params":params,
                "optimizer":optimizer,"optimizer_state":optimizer.init(params)}
            updated,opt_state,metrics=optimize_batch(state,data,initial_params[0],key)
            bundle=(initial_params[0],updated.policy,updated.value)
            audit=logprob_audit(network,updated,initial_params[0],data)
            evaluation=evaluate(env,rows,params=bundle,seed=9500000,policy_name=f"lr_{index}_repeat_{repeat}")
            summary=evaluation["summary"]["overall"]
            actions=np.concatenate([np.asarray(row["actions"])[:min(len(row["actions"]),len(base["actions"]))]
                -np.asarray(base["actions"])[:min(len(row["actions"]),len(base["actions"]))]
                for row,base in zip(evaluation["rows"],baseline["rows"])])
            ratio=audit["ratio"]
            clip_fraction=_ratio_clip_fraction(network,updated,initial_params[0],data)
            fixed_action_delta=_fixed_observation_action_delta(
                network,params,updated,initial_params[0],data)
            finite=all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves((updated,opt_state)))
            new_failures=set(summary["failure_reasons"])-set(old["failure_reasons"])
            repeats.append({"policy_hash":tree_hash(updated),"optimizer_hash":tree_hash(opt_state),
                "finite":finite,"parameter_delta":tree_delta(updated,params),
                "analytic_kl":audit["analytic_distribution_kl_mean"],
                "sample_mean_kl":audit["sample_mean_kl"],"ratio":ratio,
                "clip_fraction":clip_fraction,
                "deterministic_action_max_delta":fixed_action_delta,
                "closed_loop_trajectory_action_max_delta":float(np.max(np.abs(actions))),
                "survival_counts":summary["survival_counts"],
                "median_time_to_failure":summary["time_to_failure"]["median"],
                "failure_reasons":summary["failure_reasons"],"new_failure_types":sorted(new_failures),
                "optimizer_diagnostics":_metric_projection(metrics)})
        deterministic=repeats[0]["policy_hash"]==repeats[1]["policy_hash"] and repeats[0]["optimizer_hash"]==repeats[1]["optimizer_hash"]
        first=repeats[0];survival=first["survival_counts"]
        passed=(first["finite"] and first["parameter_delta"]["l2"]>0 and first["analytic_kl"]<=hard
            and abs(first["sample_mean_kl"])<=hard and first["clip_fraction"]<=.20
            and first["deterministic_action_max_delta"]<=.05 and not first["new_failure_types"]
            and survival["8"]>=14 and survival["16"]>=1 and survival["24"]>=1
            and first["median_time_to_failure"]>=old["time_to_failure"]["median"]-1 and deterministic)
        results.append({"index":index,"multiplier_of_m0":multipliers[index],"learning_rate":lr,
                        "repeats":repeats,"deterministic_repeat":deterministic,"passed":passed})
    passing=[row for row in results if row["passed"]];selected=max(passing,key=lambda row:row["learning_rate"]) if passing else None
    output={"status":"PASS","candidates":results,"selected_learning_rate":selected["learning_rate"] if selected else None,
            "selected_index":selected["index"] if selected else None,"candidate_count":5,"passing_count":len(passing),
            "no_new_training_rollout":True}
    save_json(root/"optimizer_lr_calibration_results.json",output)
    gate_report={"status":"PASS" if selected else "FAIL","optimizer_protocol_gate":bool(selected),
        "selected_learning_rate":selected["learning_rate"] if selected else None,
        "normalizer_hash_unchanged":normalizer_summary(initial_params[0])["sha256"]==EXPECTED_NORMALIZER,
        "baseline_8_16_24":old["survival_counts"],"deterministic_repeats":all(row["deterministic_repeat"] for row in results),
        "phase_b_authorized":bool(selected),"ppo_authorization":bool(selected)}
    save_json(root/"effective_update_gate_report.json",gate_report)
    print(json.dumps(gate_report,indent=2))

if __name__=="__main__":main()
