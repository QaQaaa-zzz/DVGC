"""Run the single 6,400-step P0-seeded Descent-to-C_L RSI pilot."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import time
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cli.run_backward_descent_nominal_pilot import (
    C_L, EXPECTED, PI_D, PI_L, PERTURBATIONS, _batched, _load_record,
    _micro_states, _outcome,
)
from cli.runtime_gate import source_fingerprint
from dvgc.backward_search import active_prefix_exact, make_descent_landing_rollout
from dvgc.backward_tube import BackwardTubeNode, canonical_hash, p0_decision, p1_decision, summarize_tube_nodes, tube_gate, validate_parent_lineage
from dvgc.bank import SnapshotBank
from dvgc.config import ACTION_MAPPING_VERSION, file_sha256, load_config, save_config
from dvgc.env import OrangeBikeDVGC
from dvgc.policy import load_bundle, save_bundle
from dvgc.runtime import make_ppo_train_fn, ppo_effective_timesteps, save_json, validate_ppo_batch_layout


def build_p0_training_records(nodes, loader):
    """Build a balanced proposal-only reset bank without promoting Tube labels."""
    unique={}
    for node in nodes:
        if not node["p0"]:continue
        current=unique.get(node["source_state_hash"])
        if current is None or (node["p1"] and not current["p1"]):unique[node["source_state_hash"]]=node
    buckets=Counter((node["candidate_id"],node["layer"],node["region"]) for node in unique.values())
    records=[]
    for node in unique.values():
        record=copy.deepcopy(loader(node))
        bucket=(node["candidate_id"],node["layer"],node["region"])
        record.update({"reset_source":"flight_curriculum","reset_weight":1.0/(len(buckets)*buckets[bucket]),"reset_parent_id":node["node_id"],"backward_tube_label":"P1" if node["p1"] else "P0","artifact_role":"proposal_support_bank","training_eligible":True})
        for forbidden in ("final","chain","tube_version","policy_version","estimator_version"):
            record.pop(forbidden,None)
        records.append(record)
    return records


def _params_hash(params):
    digest=hashlib.sha256()
    for leaf in jax.tree.leaves(params):
        array=np.asarray(leaf);digest.update(str(array.shape).encode());digest.update(str(array.dtype).encode());digest.update(array.tobytes())
    return digest.hexdigest()


def certify_policy(env,params,landing_params,nodes,seed,record_loader=None):
    rollout=make_descent_landing_rollout(env,params,landing_params,horizon=200,residual_ticks=8)
    rows=[];certified=[];safe_ids={row["id"] for row in env._cert_bank.records if row["final"]["label"]=="safe"}
    policy_hash=_params_hash(params)
    for position,source in enumerate(nodes):
        record=(record_loader(source) if record_loader is not None else _load_record(source["physical_state"]));item_seed=seed+position
        state=_batched(env,record,1,item_seed);zero=jnp.zeros((1,2,4),jnp.float32)
        first=jax.device_get(rollout(state,zero,jax.random.PRNGKey(item_seed)));second=jax.device_get(rollout(state,zero,jax.random.PRNGKey(item_seed)))
        exact,mismatch=active_prefix_exact(first,second);repeats=[_outcome(first,0,exact,mismatch),_outcome(second,0,exact,mismatch)];p0=p0_decision(repeats)
        branches=[];p1={"pass":False,"reasons":["p0_not_passed"],"successes":0,"branches":0}
        if p0["pass"]:
            micro=_micro_states(env,record,item_seed+1000);branch=jax.device_get(rollout(micro,jnp.zeros((4,2,4),jnp.float32),jax.random.PRNGKey(item_seed+2000)))
            branches=[_outcome(branch,i)|{"perturbation_vx_vz":PERTURBATIONS[i].tolist()} for i in range(4)];p1=p1_decision(p0,branches,repeats[0]["failure_type"])
            node=BackwardTubeNode(node_id=canonical_hash({"source":source["source_state_hash"],"policy":policy_hash})[:32],phase="descent",layer=source["layer"],region=source["region"],candidate_id=source["candidate_id"],source_state_hash=source["source_state_hash"],physical_state=source["physical_state"],actor_observation=np.asarray(state.obs["state"])[0].tolist(),parent_node_id=source["parent_node_id"],parent_tube="canonical_C_L",controller_type="phase_local_rsi_policy",controller_artifact_sha256=policy_hash,entry_tick=repeats[0]["downstream_entry_tick"],downstream_entry_state={"qpos":np.asarray(first["entry_qpos"])[0].tolist(),"qvel":np.asarray(first["entry_qvel"])[0].tolist(),"nearest_C_L_node_id":source["parent_node_id"]},final_recovery=True,p0=True,p1=bool(p1["pass"]),branch_results=tuple(branches),nearest_neighbor_radius=0.0,provenance_hashes={"xml":EXPECTED["xml"],"C_L":EXPECTED["C_L"],"policy":policy_hash,"pi_L":EXPECTED["pi_L"]});node.validate();certified.append(node.to_dict())
        rows.append({"node_id":source["node_id"],"candidate_id":source["candidate_id"],"layer":source["layer"],"region":source["region"],"repeats":repeats,"P0":p0,"micro_branches":branches,"P1":p1})
    typed=[BackwardTubeNode(**node) for node in certified]
    return {"rows":rows,"nodes":certified,"P0":sum(row["P0"]["pass"] for row in rows),"P1":sum(row["P1"]["pass"] for row in rows),**summarize_tube_nodes(typed),"lineage":validate_parent_lineage(typed,safe_ids),"RSI_start_gate":tube_gate(typed),"policy_hash":policy_hash}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run",required=True);parser.add_argument("--prior-report",required=True);parser.add_argument("--seed",type=int,default=0);parser.add_argument("--timesteps",type=int,default=6400);args=parser.parse_args()
    root=Path(args.run)
    if root.exists():raise SystemExit(f"refusing to overwrite run: {root}")
    if args.seed!=0 or args.timesteps!=6400:raise SystemExit("only the preregistered seed=0, 6,400-step pilot is authorized")
    if file_sha256(C_L)!=EXPECTED["C_L"] or file_sha256(PI_D/"params.pkl")!=EXPECTED["pi_D"] or file_sha256(PI_L/"params.pkl")!=EXPECTED["pi_L"]:raise SystemExit("frozen asset mismatch")
    cfg=load_config("configs/backward_descent_rsi_pilot_v1.json")
    if file_sha256(cfg.xml_path)!=EXPECTED["xml"] or cfg.action_mapping_version!=ACTION_MAPPING_VERSION:raise SystemExit("runtime model mismatch")
    gate=json.loads(Path("runs/backward_recovery_tube_fast_track_v1/RUNTIME_GATE.json").read_text());fingerprint=source_fingerprint(Path.cwd())
    if gate.get("status")!="PASS" or gate.get("source_fingerprint")!=fingerprint:raise SystemExit("runtime gate stale")
    prior=json.loads(Path(args.prior_report).read_text());source_nodes=prior["nodes"]
    records=build_p0_training_records(source_nodes,lambda node:_load_record(node["physical_state"]))
    if len(records)<16:raise SystemExit("insufficient P0 seeds for the one-shot pilot")
    root.mkdir(parents=True);save_config(cfg,root/"effective_config.json")
    bank=SnapshotBank(records,{"artifact_role":"proposal_support_bank","reset_source_protocol":"backward_P0_balanced_v1","formal_tube_or_jel":False,"source_report":args.prior_report,"source_report_sha256":file_sha256(args.prior_report)})
    bank.save(root/"p0_training_bank.pkl")
    dparams,_,dmanifest=load_bundle(PI_D,verify_files=True);lparams,_,_=load_bundle(PI_L,verify_files=True);entry=SnapshotBank.load(C_L)
    train_env=OrangeBikeDVGC(cfg,snapshot_bank=bank,cert_bank=entry)
    eval_cfg=load_config("configs/backward_descent_rsi_pilot_v1.json",{"expert_chain_termination":False})
    eval_env=OrangeBikeDVGC(eval_cfg,snapshot_bank=SnapshotBank(),cert_bank=entry)
    baseline=certify_policy(eval_env,dparams,lparams,source_nodes,43_000_000);save_json(root/"baseline_construction_certification.json",baseline)
    num_envs,batch_size,num_minibatches,num_evals=50,25,2,5
    validate_ppo_batch_layout(num_envs=num_envs,batch_size=batch_size,num_minibatches=num_minibatches)
    effective=ppo_effective_timesteps(6400,unroll_length=32,batch_size=batch_size,num_minibatches=num_minibatches,num_evals=num_evals)
    if effective!=6400:raise SystemExit(f"unexpected effective budget {effective}")
    protocol={"experiment":"backward_descent_p0_rsi_pilot_v1","seed":0,"effective_steps":effective,"initial_policy_hash":EXPECTED["pi_D"],"C_L_hash":EXPECTED["C_L"],"P0_reset_states":len(records),"formal_RSI_authorized":False,"single_below_gate_pilot":True,"training_terminal":"canonical_C_L_entry","post_training_evaluation":"uninterrupted trained_pi_D_to_C_L_then_frozen_pi_L_to_Final","heldout_used":False}
    save_json(root/"pilot_config.json",protocol)
    save_bundle(root/"checkpoint_0000",params=dparams,config=cfg,xml_path=cfg.xml_path,candidate_bank=root/"p0_training_bank.pkl",downstream_bank=C_L,policy_version="backward-descent-rsi-pilot-0000",extra={"artifact_role":"bounded_phase_local_rsi_pilot","effective_steps":0,"ppo_authorization":"single_pilot_only"})
    progress=[]
    def progress_fn(step,metrics):
        progress.append({"effective_steps":int(step),**{key:float(value) for key,value in metrics.items() if np.asarray(value).shape==()}});save_json(root/"training_progress.json",{"status":"running","progress":progress})
    train_fn=make_ppo_train_fn(timesteps=6400,episode_length=64,num_envs=num_envs,num_eval_envs=16,num_evals=num_evals,seed=0,learning_rate=float(dmanifest["ppo_hyperparameters"]["learning_rate"]),entropy_cost=float(dmanifest["ppo_hyperparameters"]["entropy_cost"]),reward_scaling=.1,checkpoint_dir=root/"orbax",unroll_length=32,batch_size=batch_size,num_minibatches=num_minibatches,num_updates_per_batch=2,discounting=.995,gae_lambda=.97,clipping_epsilon=.10,max_grad_norm=.75,restore_params=dparams,full_reset=True)
    started=time.time()
    try:_,final_params,final_metrics=train_fn(environment=train_env,progress_fn=progress_fn,eval_env=train_env)
    except BaseException as exc:save_json(root/"training_integrity_report.json",{"status":"FAIL","error_type":type(exc).__name__,"error":str(exc),"PPO_authorization":False});raise
    save_bundle(root/"checkpoint_6400",params=final_params,config=cfg,xml_path=cfg.xml_path,candidate_bank=root/"p0_training_bank.pkl",downstream_bank=C_L,policy_version="backward-descent-rsi-pilot-6400",extra={"artifact_role":"bounded_phase_local_rsi_pilot","effective_steps":6400,"initial_policy_hash":EXPECTED["pi_D"],"ppo_authorization":"pilot_complete_no_further_authorization"})
    final=certify_policy(eval_env,final_params,lparams,source_nodes,44_000_000);save_json(root/"final_construction_certification.json",final)
    nonfinite=any(not np.isfinite(value) for row in progress for value in row.values() if isinstance(value,float))
    integrity={"status":"PASS" if not nonfinite else "FAIL","effective_steps":6400,"nonfinite":nonfinite,"oom":False,"timeout":False,"bank_hash":file_sha256(root/"p0_training_bank.pkl"),"runtime_fingerprint":fingerprint,"frozen_assets_unchanged":file_sha256(C_L)==EXPECTED["C_L"] and file_sha256(PI_D/"params.pkl")==EXPECTED["pi_D"] and file_sha256(PI_L/"params.pkl")==EXPECTED["pi_L"],"elapsed_seconds":time.time()-started,"PPO_authorization":False};save_json(root/"training_integrity_report.json",integrity)
    report={"status":"PASS" if integrity["status"]=="PASS" else "FAIL","experiment":"backward_descent_p0_rsi_pilot_v1","head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"protocol":protocol,"baseline":{k:baseline[k] for k in ("P0","P1","candidate_coverage","layer_coverage","region_coverage","RSI_start_gate")},"final":{k:final[k] for k in ("P0","P1","candidate_coverage","layer_coverage","region_coverage","RSI_start_gate")},"new_P0":final["P0"]-baseline["P0"],"new_P1":final["P1"]-baseline["P1"],"integrity":integrity,"final_metrics":final_metrics,"PPO_authorization":False,"heldout_used":False,"formal_tube_or_jel":False}
    save_json(root/"BACKWARD_DESCENT_P0_RSI_PILOT_V1_REPORT.json",report);save_json(root/"completed.json",{"status":report["status"],"gate":final["RSI_start_gate"]["status"],"new_P0":report["new_P0"],"new_P1":report["new_P1"]})
    print(json.dumps({"status":report["status"],"baseline_P0":baseline["P0"],"baseline_P1":baseline["P1"],"final_P0":final["P0"],"final_P1":final["P1"],"new_P0":report["new_P0"],"new_P1":report["new_P1"],"gate":final["RSI_start_gate"]["status"]},indent=2))


if __name__=="__main__":main()
