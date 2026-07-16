"""No-training decomposition of the current descent reward on the local pool."""
from __future__ import annotations

import argparse,json
from collections import Counter,defaultdict
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from dvgc.bank import SnapshotBank
from dvgc.composite import CanonicalEntryMatcher
from dvgc.config import file_sha256,load_config
from dvgc.env import END_REASON,OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.rollout import restore_snapshot
from dvgc.runtime import build_inference,save_json


def scalar(value): return float(np.asarray(jax.device_get(value)))


def rollout(env,step,inference,matcher,row,key,horizon,neutral):
    state=restore_snapshot(env,row,key); previous=np.asarray(state.info["last_action"],np.float32); trace=[]; chain=False
    for t in range(int(horizon)):
        key,ak=jax.random.split(key); action=jp.zeros(env.action_size,jp.float32) if neutral else inference(state.obs,ak)[0]; action_np=np.asarray(jax.device_get(action),np.float32); before=matcher.match(state)[1]; state=step(state,action); hit,distance=matcher.match(state); chain=chain or hit
        feature=np.asarray(jax.device_get(env._physical_feature(state.data)),np.float64); rewards={k:scalar(v) for k,v in state.metrics.items() if k.startswith("reward/")}
        trace.append({"step":t+1,"pitch":float(feature[4]),"pitch_rate":float(feature[10]),"roll":float(feature[3]),"roll_rate":float(feature[9]),"vx":float(feature[6]),"vz":float(feature[8]),"hip":float(feature[13]),"knee":float(feature[14]),"distance_before":before,"distance_to_c_l":distance,"action_magnitude":float(np.linalg.norm(action_np)),"action_difference":float(np.linalg.norm(action_np-previous)),"reward":rewards}); previous=action_np
        if scalar(state.done)>.5: break
    end=int(scalar(state.info["end_code"])); reason=END_REASON.get(end,f"unknown_{end}"); terminated=bool(scalar(state.info["terminated"])); return {"candidate_id":row["id"],"bootstrap_group":row["bootstrap_group"],"descent_layer":row["descent_layer"],"candidate_kind":row["candidate_kind"],"steps":len(trace),"chain":chain,"terminated":terminated,"physical_failure":terminated and reason not in ("chain_entry","recovery"),"truncated":bool(scalar(state.info["truncated"])),"termination_reason":reason,"minimum_distance_to_c_l":min((x["distance_to_c_l"] for x in trace),default=float("inf")),"initial_distance_to_c_l":trace[0]["distance_before"] if trace else float("inf"),"final_distance_to_c_l":trace[-1]["distance_to_c_l"] if trace else float("inf"),"reward_total":sum(x["reward"].get("reward/total",0.0) for x in trace),"positive_reward_total":sum(max(0.0,x["reward"].get("reward/total",0.0)) for x in trace),"trace":trace}


def summarize(rows):
    terms=defaultdict(list)
    for row in rows:
        episode=defaultdict(float)
        for step in row["trace"]:
            for key,value in step["reward"].items(): episode[key]+=value
        for key,value in episode.items(): terms[key].append(value)
    return {"episodes":len(rows),"chain_rate":float(np.mean([r["chain"] for r in rows])) if rows else 0.0,"physical_failure_rate":float(np.mean([r["physical_failure"] for r in rows])) if rows else 0.0,"timeout_rate":float(np.mean([r["truncated"] for r in rows])) if rows else 0.0,"steps":{"mean":float(np.mean([r["steps"] for r in rows])) if rows else 0.0,"p50":float(np.median([r["steps"] for r in rows])) if rows else 0.0,"p95":float(np.quantile([r["steps"] for r in rows],.95)) if rows else 0.0},"minimum_distance":{"min":float(np.min([r["minimum_distance_to_c_l"] for r in rows])) if rows else None,"p50":float(np.median([r["minimum_distance_to_c_l"] for r in rows])) if rows else None,"p95":float(np.quantile([r["minimum_distance_to_c_l"] for r in rows],.95)) if rows else None},"reward_episode_means":{key:float(np.mean(values)) for key,values in sorted(terms.items())},"positive_return_p95":float(np.quantile([r["positive_reward_total"] for r in rows],.95)) if rows else 0.0,"termination_reasons":dict(Counter(r["termination_reason"] for r in rows))}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--policy",required=True); p.add_argument("--pool",required=True); p.add_argument("--landing-entry-set",required=True); p.add_argument("--output",required=True); p.add_argument("--config",default="configs/default.json"); p.add_argument("--seed",type=int,default=7500000); p.add_argument("--horizon",type=int,default=64); p.add_argument("--descent-local-reward",action="store_true"); a=p.parse_args(); out=Path(a.output)
    if out.exists(): raise SystemExit(f"Output exists: {out}")
    params,cfg_dict,manifest=load_bundle(a.policy,verify_files=True); cfg=load_config(a.config,{**cfg_dict,"training_stage":"flight","expert_chain_termination":True,"domain_randomization":False,"obs_noise_enable":False,"use_bank_resets":False,"descent_local_reward_enable":bool(a.descent_local_reward)}); entry=SnapshotBank.load(a.landing_entry_set); env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank(),cert_bank=entry); matcher=CanonicalEntryMatcher(env,"flight",a.landing_entry_set); inference=build_inference(env,params,deterministic=True); step=jax.jit(env.step)
    bank=SnapshotBank.load(a.pool); candidates=[r for r in bank.records_for_phase("flight",include_training_only=False) if r.get("local_bootstrap_eligible")]; rows=[]
    for policy_name,neutral in (("pi_f_descent_local",False),("neutral",True)):
        for i,row in enumerate(candidates):
            result=rollout(env,step,inference,matcher,row,jax.random.PRNGKey(a.seed+i),a.horizon,neutral); result["probe_policy"]=policy_name; rows.append(result)
    summary={policy:{"overall":summarize([r for r in rows if r["probe_policy"]==policy]),"groups":{group:summarize([r for r in rows if r["probe_policy"]==policy and r["bootstrap_group"]==group]) for group in sorted({r["bootstrap_group"] for r in rows})},"layers":{layer:summarize([r for r in rows if r["probe_policy"]==policy and r["descent_layer"]==layer]) for layer in ("late","middle","early")}} for policy in ("pi_f_descent_local","neutral")}
    by={(r["candidate_id"],r["probe_policy"]):r for r in rows}; diagnosis=Counter()
    for row in candidates:
        policy=by[(row["id"],"pi_f_descent_local")]; neutral=by[(row["id"],"neutral")]
        if neutral["terminated"] and neutral["steps"]<=5: diagnosis["candidate_immediately_unstable"]+=1
        if policy["physical_failure"] and neutral["steps"]>=policy["steps"]+5: diagnosis["policy_induced_instability"]+=1
        if not policy["chain"] and policy["steps"]>=a.horizon and policy["minimum_distance_to_c_l"]>=policy["initial_distance_to_c_l"]: diagnosis["survives_without_c_l_progress"]+=1
    report={"status":"PASS","reward_profile":"descent_local" if a.descent_local_reward else "current_unified","seed":a.seed,"horizon":a.horizon,"policy_version":manifest["policy_version"],"policy_hash":file_sha256(Path(a.policy)/"params.pkl"),"pool_sha256":file_sha256(a.pool),"c_l_sha256":file_sha256(a.landing_entry_set),"candidate_count":len(candidates),"diagnosis":dict(diagnosis),"summary":summary,"rows":rows}
    save_json(out,report); print(json.dumps({k:v for k,v in report.items() if k!="rows"},indent=2))


if __name__=="__main__": main()
