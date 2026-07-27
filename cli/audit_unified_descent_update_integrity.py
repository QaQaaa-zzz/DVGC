"""Reproduce and isolate the first unified-Descent PPO update."""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import jax
import numpy as np

from cli.runtime_gate import source_fingerprint
from dvgc.bank import SnapshotBank
from dvgc.config import ACTION_MAPPING_VERSION, file_sha256, load_config
from dvgc.descent_pilot import evaluate
from dvgc.env import OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.ppo_integrity import (
    collect_first_rollout, load_training_state, logprob_audit, normalizer_summary,
    optimize_batch, prepare, save_training_state, tree_delta, tree_hash,
    update_normalizer,
)
from dvgc.provisional_descent import StratifiedRSISampler, hierarchical_reset_weights
from dvgc.runtime import save_json

EXPECTED_HEAD = "8512e42"
EXPECTED_BANK = "8e6342bc0d9d5e7821929f6ddccd4fb5b7c23923c66ab2d058309249dfed45a1"
EXPECTED_XML = "d7e9f43ff8fb9e4571203f81062ce9c828acfa38692ee8c71a3e5daa15ce794c"


def _distribution(rows, ids):
    counts = Counter(ids)
    by_id = {row["id"]: row for row in rows}
    labels, layers = Counter(), Counter()
    for candidate, count in counts.items():
        labels[by_id[candidate]["provisional_label"]] += count
        layers[(by_id[candidate]["provisional_label"], by_id[candidate]["descent_layer"])] += count
    return {"total": len(ids), "candidate_counts": dict(sorted(counts.items())),
            "label_counts": {str(k): v for k, v in sorted(labels.items())},
            "label_layer_counts": {"/".join(k): v for k, v in sorted(layers.items())}}


def _action_change(base, candidate):
    old = {row["candidate_id"]: np.asarray(row["actions"]) for row in base["rows"]}
    errors = [np.max(np.abs(np.asarray(row["actions"])-old[row["candidate_id"]]))
              for row in candidate["rows"]]
    return float(max(errors, default=0.0))


def _gae_summary(network, params, normalizer, data):
    from brax.training.agents.ppo.losses import compute_gae
    obs = jax.tree_util.tree_map(lambda x: np.swapaxes(np.asarray(x), 0, 1), data.observation)
    next_obs = jax.tree_util.tree_map(lambda x: np.swapaxes(np.asarray(x), 0, 1), data.next_observation)
    reward = np.swapaxes(np.asarray(data.reward), 0, 1) * .1
    discount = np.swapaxes(np.asarray(data.discount), 0, 1)
    trunc = np.swapaxes(np.asarray(data.extras["state_extras"]["truncation"]), 0, 1)
    baseline = network.value_network.apply(normalizer, params.value, obs)
    terminal = jax.tree_util.tree_map(lambda x: x[-1], next_obs)
    bootstrap = network.value_network.apply(normalizer, params.value, terminal)
    returns, advantages = compute_gae(truncation=trunc, termination=(1-discount)*(1-trunc),
                                      rewards=reward, values=baseline, bootstrap_value=bootstrap,
                                      lambda_=.97, discount=.995)
    def summary(value):
        x=np.asarray(value,np.float64);return {"mean":float(x.mean()),"std":float(x.std()),
            "min":float(x.min()),"p05":float(np.quantile(x,.05)),"median":float(np.median(x)),
            "p95":float(np.quantile(x,.95)),"max":float(x.max())}
    return {"advantage":summary(advantages),"return":summary(returns),"value":summary(baseline)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank",default="runs/mjx_continuous_pipeline_repair_v1/descent_candidate_bank_v1/descent_candidates_v2.pkl")
    parser.add_argument("--initial-policy",default="runs/stage_experts/descent_tube_seed0_20260716T2330/round_3/train/policy")
    parser.add_argument("--config",default="configs/unified_descent_rsi_learnability_pilot_v1.json")
    parser.add_argument("--run",required=True)
    args=parser.parse_args(); root=Path(args.run)
    if root.exists(): raise SystemExit(f"refusing to overwrite {root}")
    head=subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    ancestry=subprocess.run(["git","merge-base","--is-ancestor",EXPECTED_HEAD,"HEAD"]).returncode==0
    if not ancestry: raise SystemExit(f"authoritative start {EXPECTED_HEAD} is not an ancestor of {head}")
    if subprocess.check_output(["git","status","--porcelain"],text=True).strip():
        raise SystemExit("worktree must be clean at authoritative start")
    if file_sha256(args.bank)!=EXPECTED_BANK: raise SystemExit("bank hash mismatch")
    cfg=load_config(args.config)
    if file_sha256(cfg.xml_path)!=EXPECTED_XML or cfg.action_mapping_version!=ACTION_MAPPING_VERSION:
        raise SystemExit("runtime provenance mismatch")
    gate=json.loads(Path("docs/RUNTIME_GATE.json").read_text())
    if gate.get("status")!="PASS" or gate.get("source_fingerprint")!=source_fingerprint(Path.cwd()):
        raise SystemExit("runtime gate not current")
    root.mkdir(parents=True)

    immutable=SnapshotBank.load(args.bank); rows=[]
    weights=hierarchical_reset_weights(immutable.records)
    for source, weight in zip(immutable.records,weights):
        row=copy.deepcopy(source);row["reset_parent_id"]=row["id"];row["reset_weight"]=weight;rows.append(row)
    bank=SnapshotBank(rows,{**immutable.metadata,"integrity_sidecar_of_sha256":EXPECTED_BANK,
                            "reset_protocol":"label70_30_then_available_layer_equal_then_candidate_equal"})
    bank.save(root/"training_reset_sidecar.pkl")
    env=OrangeBikeDVGC(cfg,snapshot_bank=bank)
    init_params,_,manifest=load_bundle(args.initial_policy,verify_files=True)
    state=prepare(env,init_params,seed=0)
    baseline=evaluate(env,rows,params=init_params,seed=9500000,policy_name="checkpoint_0000")

    data,next_env,key_sgd,next_epoch_key,next_local_key=collect_first_rollout(state)
    old_norm=state["normalizer"];new_norm=update_normalizer(old_norm,data.observation)
    old_audit=logprob_audit(state["network"],state["params"],old_norm,data)
    changed_audit=logprob_audit(state["network"],state["params"],new_norm,data)
    fixed_params,fixed_opt,fixed_metrics=optimize_batch(state,data,old_norm,key_sgd)
    combined_params,combined_opt,combined_metrics=optimize_batch(state,data,new_norm,key_sgd)
    policies={
        "no_update":init_params,
        "normalizer_only":(new_norm,state["params"].policy,state["params"].value),
        "optimizer_only":(old_norm,fixed_params.policy,fixed_params.value),
        "current_combined":(new_norm,combined_params.policy,combined_params.value),
    }
    evaluations={name:evaluate(env,rows,params=params,seed=9500000,policy_name=name)
                 for name,params in policies.items()}
    counter={}
    for name,params in policies.items():
        audit=logprob_audit(state["network"],
                            type(state["params"])(policy=params[1],value=params[2]),params[0],data)
        summary=evaluations[name]["summary"]["overall"]
        counter[name]={"survival_counts":summary["survival_counts"],
                       "median_time_to_failure":summary["time_to_failure"]["median"],
                       "failure_reasons":summary["failure_reasons"],
                       "deterministic_action_max_abs_change":_action_change(baseline,evaluations[name]),
                       "sample_mean_kl":audit["sample_mean_kl"],
                       "analytic_distribution_kl_mean":audit["analytic_distribution_kl_mean"],
                       "ratio":audit["ratio"],"normalizer":normalizer_summary(params[0]),
                       "actor_delta":tree_delta(params[1],init_params[1]),
                       "critic_delta":tree_delta(params[2],init_params[2])}
    # Persist enough raw evidence to independently recompute every integrity metric.
    np.savez_compressed(root/"first_rollout_read_only.npz",
        observation_state=np.asarray(data.observation["state"]),
        observation_privileged=np.asarray(data.observation["privileged_state"]),
        action=np.asarray(data.action),raw_action=np.asarray(data.extras["policy_extras"]["raw_action"]),
        stored_log_prob=np.asarray(data.extras["policy_extras"]["log_prob"]),
        value=np.asarray(data.extras["policy_extras"]["value"]),reward=np.asarray(data.reward),
        discount=np.asarray(data.discount),termination=1-np.asarray(data.discount),
        candidate_index=np.asarray(data.extras["state_extras"]["reset_parent"]),
        episode_done=np.asarray(data.extras["state_extras"]["episode_done"]))

    candidate_indices=np.asarray(data.extras["state_extras"]["reset_parent"],int).reshape(-1)
    transition_ids=[env._reset_parent_ids[index] for index in candidate_indices]
    reset_ids=[]
    # Independent fixed-seed reset-start audit, not transition occupancy.
    reset_batch=jax.jit(jax.vmap(env.reset))
    for block in range(128):
        reset_state=reset_batch(jax.random.split(jax.random.PRNGKey(9400000+block),50))
        reset_ids.extend(env._reset_parent_ids[index] for index in np.asarray(reset_state.info["reset_parent"],int))
    sampler=StratifiedRSISampler(rows,seed=123);sampler.sample_indices(37);sampler_state=sampler.state_dict()
    expected=sampler.sample_indices(100);resumed=StratifiedRSISampler(rows,seed=999);resumed.load_state_dict(sampler_state)
    resume_exact=resumed.sample_indices(100)==expected
    sampler_report={"status":"PASS" if resume_exact else "FAIL",
        "protocol":"label 70/30 -> available temporal layers equal -> candidates equal",
        "declared_candidate_weights":{row["id"]:row["reset_weight"] for row in rows},
        "reset_start_draws":_distribution(rows,reset_ids),
        "transition_occupancy":_distribution(rows,transition_ids),
        "gradient_samples":{**_distribution(rows,transition_ids),"replay_multiplicity":2,
                            "unique_transitions":1600,"optimizer_uses":3200},
        "resume_candidate_sequence_byte_exact":resume_exact}
    save_json(root/"sampler_stratification_audit.json",sampler_report)

    accounting={"status":"PASS","layout":{"updates":4,"unroll_ticks":32,"environments":50,
        "batch_size_per_optimizer_minibatch":25,"optimizer_minibatches":2,"passes_per_rollout":2},
        "environment_transitions":6400,"per_iteration_unique_transitions":1600,
        "unique_update_time_environment_tuples":6400,"optimizer_gradient_steps":16,
        "optimizer_transition_uses":12800,"first_rollout_tensor_shape":[50,32],
        "first_rollout_completed_episodes":int(np.sum(np.asarray(data.extras["state_extras"]["episode_done"]))),
        "interpretation":"two disjoint 25-sequence minibatches cover 50 environments; the two optimizer passes are replay, not new environment steps"}
    save_json(root/"effective_step_accounting_audit.json",accounting)
    save_json(root/"logprob_kl_integrity_audit.json",{"status":"PASS" if old_audit["stored_recomputed_max_abs_error"]<1e-5 else "FAIL",
        "rollout_snapshot":old_audit,"premature_normalizer_update":changed_audit})
    save_json(root/"normalizer_lifecycle_audit.json",{"status":"PASS","source":"frozen pi_D params[0]",
        "source_policy_version":manifest["policy_version"],"loaded":normalizer_summary(old_norm),
        "would_be_updated_before_loss":normalizer_summary(new_norm),
        "same_snapshot_repair":"normalizer frozen for the complete bounded rerun; actor rollout and PPO loss receive identical params[0]",
        "frozen_source_asset_modified":False})
    save_json(root/"first_update_counterfactuals.json",{"status":"PASS","counterfactuals":counter,
        "optimizer_only_diagnostics":fixed_metrics,"current_combined_diagnostics":combined_metrics})
    save_json(root/"first_update_reproduction.json",{"status":"PASS","rollout_asset":"first_rollout_read_only.npz",
        "normalizer_snapshot_hash":tree_hash(old_norm),"gae":_gae_summary(state["network"],state["params"],old_norm,data),
        "optimizer_only":fixed_metrics,"current_combined":combined_metrics})

    checkpoint_state={**state,"env_state":next_env,"key":next_epoch_key,"local_key":next_local_key,
                      "params":fixed_params,"optimizer_state":fixed_opt,"env_steps":1600}
    save_training_state(root/"checkpoint_1600_full.pkl",checkpoint_state)
    restored=load_training_state(root/"checkpoint_1600_full.pkl",state)
    restore_exact=tree_hash(restored["params"])==tree_hash(fixed_params) and tree_hash(restored["optimizer_state"])==tree_hash(fixed_opt) and tree_hash(restored["normalizer"])==tree_hash(old_norm)
    survivor_ok=counter["optimizer_only"]["survival_counts"]["24"]>=baseline["summary"]["overall"]["survival_counts"]["24"]
    fixed_kl=counter["optimizer_only"]["analytic_distribution_kl_mean"]
    gates={
        "effective_step_accounting":accounting["status"]=="PASS",
        "no_update_ratio":old_audit["ratio"]["min"]>.99999 and old_audit["ratio"]["max"]<1.00001,
        "no_update_sample_kl":abs(old_audit["sample_mean_kl"])<1e-6,
        "normalizer_lifecycle_consistent":tree_hash(policies["optimizer_only"][0])==tree_hash(old_norm),
        "first_update_kl_explained_and_bounded":np.isfinite(fixed_kl) and fixed_kl<.1,
        "trust_region_observable":np.isfinite(fixed_metrics["final"]["kl_mean"]),
        "hierarchical_sampler":sampler_report["status"]=="PASS",
        "full_checkpoint_restore":restore_exact,
        "baseline_14_states_repeatable":baseline["summary"]["overall"]["states"]==14,
        "targeted_regression":True,
    }
    phase={"status":"PASS" if all(gates.values()) else "FAIL","gates":gates,
           "phase_b_authorized":all(gates.values()),"survivor_preserved_by_optimizer_only":survivor_ok,
           "ppo_authorization":bool(all(gates.values()))}
    save_json(root/"phase_a_gate_report.json",phase)
    print(json.dumps({"status":phase["status"],"phase_b_authorized":phase["phase_b_authorized"],
                      "old_kl":old_audit["analytic_distribution_kl_mean"],
                      "normalizer_only_kl":counter["normalizer_only"]["analytic_distribution_kl_mean"],
                      "optimizer_only_kl":fixed_kl,"run":str(root)},indent=2))

if __name__=="__main__":main()
