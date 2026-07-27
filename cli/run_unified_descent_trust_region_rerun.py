"""Run the single authorized 6,400-transition trust-region Descent rerun."""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax.training import acting
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks

from cli.calibrate_unified_descent_optimizer_trust_region import (
    EXPECTED_BANK, EXPECTED_NORMALIZER, EXPECTED_POLICY, EXPECTED_XML,
    _fixed_observation_action_delta, _ratio_clip_fraction,
)
from cli.runtime_gate import source_fingerprint
from dvgc.bank import SnapshotBank
from dvgc.config import ACTION_MAPPING_VERSION, file_sha256, load_config
from dvgc.descent_pilot import evaluate
from dvgc.env import OrangeBikeDVGC
from dvgc.policy import load_bundle, save_bundle
from dvgc.ppo_integrity import (
    logprob_audit, make_optimizer, normalizer_summary, prepare,
    save_training_state, tree_hash,
)
from dvgc.provisional_descent import hierarchical_reset_weights
from dvgc.runtime import save_json

EXPECTED_START="e61639f"
HELDOUT=Path("runs/unified_descent_rsi_learnability_pilot_v1_seed0_20260727/heldout_sidecar.pkl")


def _collect(state,key_unroll):
    inference=(state["normalizer"],state["params"].policy,state["params"].value)
    policy=ppo_networks.make_inference_fn(state["network"])(inference)
    final,data=acting.generate_unroll(state["env"],state["env_state"],policy,key_unroll,32,
        extra_fields=("truncation","episode_metrics","episode_done","reset_parent"))
    return final,jax.tree_util.tree_map(lambda x:jnp.swapaxes(x,0,1),data)


def _optimize_online(state,data,key,current_lr,halvings,total_halving_limit=3):
    loss_fn=partial(ppo_losses.compute_ppo_loss,ppo_network=state["network"],entropy_cost=.001,
        discounting=.995,reward_scaling=.1,gae_lambda=.97,clipping_epsilon=.10)
    value_grad=jax.jit(jax.value_and_grad(loss_fn,has_aux=True))
    params,opt_state=state["params"],state["optimizer_state"]
    attempts=[];accepted=rolled=0;paused=False
    for pass_index in range(2):
        key,perm_key=jax.random.split(key);order=np.asarray(jax.random.permutation(perm_key,data.reward.shape[0]))
        for minibatch_index,indices in enumerate(np.array_split(order,2)):
            minibatch=jax.tree_util.tree_map(lambda x:x[indices],data)
            retry=0
            while True:
                key,loss_key=jax.random.split(key)
                (loss,metrics),grads=value_grad(params,state["normalizer"],minibatch,loss_key)
                optimizer=make_optimizer(current_lr);updates,candidate_opt=optimizer.update(grads,opt_state,params)
                candidate=optax.apply_updates(params,updates)
                audit=logprob_audit(state["network"],candidate,state["normalizer"],data)
                finite=all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves((candidate,candidate_opt)))
                action_delta=_fixed_observation_action_delta(state["network"],params,candidate,state["normalizer"],data)
                ok=finite and audit["analytic_distribution_kl_mean"]<=.01 and abs(audit["sample_mean_kl"])<=.01 and action_delta<=.05
                attempts.append({"pass":pass_index,"minibatch":minibatch_index,"retry":retry,
                    "learning_rate":current_lr,"accepted":ok,"finite":finite,"loss":float(loss),
                    "analytic_kl":audit["analytic_distribution_kl_mean"],
                    "sample_mean_kl":audit["sample_mean_kl"],
                    "clip_fraction":_ratio_clip_fraction(state["network"],candidate,state["normalizer"],data),
                    "deterministic_action_max_delta":action_delta,
                    "policy_hash_before":tree_hash(params),"policy_hash_candidate":tree_hash(candidate),
                    **{k:float(v) for k,v in metrics.items()}})
                if ok:
                    params,opt_state=candidate,candidate_opt;accepted+=1;break
                rolled+=1
                if retry==0 and halvings<total_halving_limit:
                    current_lr*=.5;halvings+=1;retry=1;continue
                paused=True;break
            if paused:break
        if paused:break
    return params,opt_state,current_lr,halvings,{"attempts":attempts,"accepted":accepted,
        "rolled_back":rolled,"paused":paused}


def _checkpoint(root,step,cfg,sidecar,original_hash,state,current_lr,counters):
    target=root/"checkpoints"/f"checkpoint_{step:04d}"
    params=(state["normalizer"],state["params"].policy,state["params"].value)
    save_bundle(target,params=params,config=cfg,xml_path=cfg.xml_path,candidate_bank=sidecar,
        downstream_bank=None,policy_version=f"unified-descent-trust-region-{step:04d}",extra={
            "artifact_role":"bounded_trust_region_rerun_checkpoint","effective_steps":step,
            "original_candidate_bank_sha256":original_hash,"learning_rate":current_lr,
            "accepted_optimizer_updates":counters["accepted"],"rolled_back_updates":counters["rolled_back"],
            "learning_rate_halvings":counters["halvings"],"normalizer_sha256":tree_hash(state["normalizer"]),
            "ppo_authorization":"single_integrity_rerun_only"})
    save_training_state(target/"full_training_state.pkl",{**state,"current_learning_rate":current_lr,
        "counters":counters,"env_steps":step})
    return target


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run",required=True)
    p.add_argument("--calibration",default="runs/unified_descent_rsi_optimizer_trust_region_repair_v1_calibration_v2_20260727")
    p.add_argument("--bank",default="runs/mjx_continuous_pipeline_repair_v1/descent_candidate_bank_v1/descent_candidates_v2.pkl")
    p.add_argument("--initial-policy",default="runs/stage_experts/descent_tube_seed0_20260716T2330/round_3/train/policy")
    p.add_argument("--config",default="configs/unified_descent_rsi_learnability_pilot_v1.json")
    a=p.parse_args();root=Path(a.run)
    if root.exists():raise SystemExit(f"refusing to overwrite {root}")
    if subprocess.run(["git","merge-base","--is-ancestor",EXPECTED_START,"HEAD"]).returncode or subprocess.check_output(["git","status","--porcelain"],text=True).strip():raise SystemExit("invalid git state")
    calibration=json.loads((Path(a.calibration)/"effective_update_gate_report.json").read_text())
    if not calibration.get("phase_b_authorized"):raise SystemExit("calibration gate did not pass")
    selected=float(calibration["selected_learning_rate"])
    cfg=load_config(a.config);gate=json.loads(Path("docs/RUNTIME_GATE.json").read_text())
    if file_sha256(a.bank)!=EXPECTED_BANK or file_sha256(cfg.xml_path)!=EXPECTED_XML or cfg.action_mapping_version!=ACTION_MAPPING_VERSION or file_sha256(Path(a.initial_policy)/"params.pkl")!=EXPECTED_POLICY:raise SystemExit("provenance mismatch")
    if gate.get("status")!="PASS" or gate.get("source_fingerprint")!=source_fingerprint(Path.cwd()):raise SystemExit("runtime gate stale")
    initial,_,_=load_bundle(a.initial_policy,verify_files=True)
    if normalizer_summary(initial[0])["sha256"]!=EXPECTED_NORMALIZER:raise SystemExit("normalizer mismatch")
    root.mkdir(parents=True);(root/"checkpoints").mkdir()
    immutable=SnapshotBank.load(a.bank);rows=[]
    for source,weight in zip(immutable.records,hierarchical_reset_weights(immutable.records)):
        row=copy.deepcopy(source);row["reset_parent_id"]=row["id"];row["reset_weight"]=weight;rows.append(row)
    sidecar=root/"training_reset_sidecar.pkl";SnapshotBank(rows,{**immutable.metadata,
        "trust_region_rerun_sidecar_of":EXPECTED_BANK}).save(sidecar)
    heldout=SnapshotBank.load(HELDOUT).records
    env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank.load(sidecar))
    state=prepare(env,initial,seed=0,learning_rate=selected)
    counters={"attempted":0,"accepted":0,"rolled_back":0,"halvings":0}
    checkpoints={};heldouts={};online=[]

    def assess(step,path):
        bundle=(state["normalizer"],state["params"].policy,state["params"].value)
        checkpoints[path.name]={"effective_steps":step,"policy_hash":file_sha256(path/"params.pkl"),
            "normalizer_hash":tree_hash(state["normalizer"]),"learning_rate":current_lr,
            "accepted_optimizer_updates":counters["accepted"],"rolled_back_updates":counters["rolled_back"],
            "learning_rate_halvings":counters["halvings"],
            "evaluation":evaluate(env,rows,params=bundle,seed=9500000,policy_name=path.name)}
        heldouts[path.name]=evaluate(env,heldout,params=bundle,seed=9300000,policy_name=path.name)

    current_lr=selected
    path=_checkpoint(root,0,cfg,sidecar,EXPECTED_BANK,state,current_lr,counters);assess(0,path)
    terminal="completed"
    for iteration in range(4):
        epoch_key,state["key"]=jax.random.split(state["key"]);key_sgd,key_unroll,_=jax.random.split(epoch_key,3)
        state["env_state"],data=_collect(state,key_unroll)
        params,opt_state,current_lr,halvings,audit=_optimize_online(
            state,data,key_sgd,current_lr,counters["halvings"])
        state["params"],state["optimizer_state"]=params,opt_state;counters["halvings"]=halvings
        counters["attempted"]+=len(audit["attempts"]);counters["accepted"]+=audit["accepted"];counters["rolled_back"]+=audit["rolled_back"]
        step=(iteration+1)*1600;online.append({"effective_steps":step,**audit,"learning_rate_after":current_lr,
            "cumulative":dict(counters),"normalizer_hash":tree_hash(state["normalizer"])})
        path=_checkpoint(root,step,cfg,sidecar,EXPECTED_BANK,state,current_lr,counters);assess(step,path)
        if audit["paused"]:terminal="trust_region_safe_pause";break
    normalizer_unchanged=all(row["normalizer_hash"]==EXPECTED_NORMALIZER for row in checkpoints.values())
    save_json(root/"online_trust_region_audit.json",{"status":"PASS" if terminal=="completed" else "FAIL",
        "terminal_state":terminal,"iterations":online,"counters":counters,
        "initial_learning_rate":selected,"final_learning_rate":current_lr,
        "normalizer_hash_unchanged":normalizer_unchanged,"ppo_authorization":False})
    save_json(root/"trust_region_rerun_checkpoint_evaluations.json",checkpoints)
    save_json(root/"trust_region_rerun_heldout_evaluation.json",heldouts)
    print(json.dumps({"status":terminal,"counters":counters,"checkpoint_count":len(checkpoints),
        "normalizer_unchanged":normalizer_unchanged,"run":str(root)},indent=2))

if __name__=="__main__":main()
