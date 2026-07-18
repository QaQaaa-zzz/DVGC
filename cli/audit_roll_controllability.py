"""Audit deployable action authority over roll on fixed descent support."""
from __future__ import annotations

import argparse,json
from collections import Counter,defaultdict
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from dvgc.bank import SnapshotBank
from dvgc.certification import branch_seed
from dvgc.config import file_sha256,load_config
from dvgc.env import END_REASON,OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.roll_controllability import CHANNELS,audit_decision
from dvgc.rollout import restore_snapshot
from dvgc.runtime import build_inference,save_json


def parent(row):return str(row.get("trajectory_parent_id",row.get("entry_source_id",row.get("parent_candidate_id",row["id"]))))


def selected_records(rows):
    chosen=[]
    def take(group,count):
        ordered=sorted(group,key=lambda row:(-abs(float(row["physical_feature"][3])),-abs(float(row["physical_feature"][9])),str(row["id"])))
        seen=Counter();added=0
        for row in ordered:
            if seen[parent(row)]>=2:continue
            chosen.append(row);seen[parent(row)]+=1;added+=1
            if added>=min(count,len(ordered)):break
    take([r for r in rows if r.get("stable_safe")],3)
    take([r for r in rows if r["final"]["label"]=="boundary"],12)
    roll_dead=[r for r in rows if r["final"]["label"]=="dead" and any(e.get("end_reason")=="roll_limit" for e in r.get("certification_branches",[]))]
    take(roll_dead,12)
    take([r for r in rows if r.get("descent_layer")=="early"],4)
    unique={r["id"]:r for r in chosen};return list(unique.values())


def rollout(env,step,inference,state,key,channel,sign,cfg):
    delta=float(cfg.roll_controllability_action_delta);initial=np.asarray(jax.device_get(env._physical_feature(state.data)),np.float64)
    contact_step=None;done=False;reason="horizon"
    for tick in range(int(cfg.roll_controllability_horizon)):
        key,ak=jax.random.split(key);action,_=inference(state.obs,ak)
        if channel is not None and tick<int(cfg.roll_controllability_pulse_steps):action=action.at[channel].set(jp.clip(action[channel]+sign*delta,-1,1))
        state=step(state,action)
        if contact_step is None and int(np.asarray(jax.device_get(state.info.get("had_valid_landing",0)))):contact_step=tick+1
        if bool(np.asarray(jax.device_get(state.done))):done=True;reason=END_REASON.get(int(np.asarray(jax.device_get(state.info["end_code"]))),"unknown");break
    final=np.asarray(jax.device_get(env._physical_feature(state.data)),np.float64)
    return {"initial":initial,"final":final,"done":done,"reason":reason,"contact_step":contact_step,"steps":tick+1}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--candidate-bank",required=True);p.add_argument("--policy",required=True)
    p.add_argument("--output",required=True);p.add_argument("--seed",type=int,required=True);p.add_argument("--config",default="configs/default.json");a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise SystemExit("Roll controllability output exists")
    params,policy_cfg,manifest=load_bundle(a.policy,verify_files=True);cfg=load_config(a.config,{**policy_cfg,"training_stage":"flight",
        "expert_chain_termination":False,"domain_randomization":False,"obs_noise_enable":False,"use_bank_resets":False})
    bank=SnapshotBank.load(a.candidate_bank);rows=selected_records(bank.records_for_phase("flight",include_training_only=False))
    if not rows:raise SystemExit("No states selected for controllability audit")
    env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank());step=jax.jit(env.step);inference=build_inference(env,params,deterministic=True)
    details=[];by_channel=defaultdict(list)
    for index,row in enumerate(rows):
        seed=branch_seed(a.seed,index,0);base_key=jax.random.PRNGKey(seed)
        baseline=rollout(env,step,inference,restore_snapshot(env,row,base_key),base_key,None,0,cfg)
        for channel,name in enumerate(CHANNELS):
            plus=rollout(env,step,inference,restore_snapshot(env,row,base_key),base_key,channel,1,cfg)
            minus=rollout(env,step,inference,restore_snapshot(env,row,base_key),base_key,channel,-1,cfg)
            delta=float(cfg.roll_controllability_action_delta);roll_s=(plus["final"][3]-minus["final"][3])/(2*delta);rate_s=(plus["final"][9]-minus["final"][9])/(2*delta)
            base_margin=abs(baseline["final"][3])+.25*abs(baseline["final"][9]);best=min(abs(plus["final"][3])+.25*abs(plus["final"][9]),abs(minus["final"][3])+.25*abs(minus["final"][9]))
            item={"id":row["id"],"parent":parent(row),"layer":row.get("descent_layer"),"label":row["final"]["label"],"channel":name,
                "branch_seed":seed,"roll_sensitivity":float(roll_s),"roll_rate_sensitivity":float(rate_s),
                "beneficial":bool(best+1e-5<base_margin),"pitch_side_effect":float(max(abs(plus["final"][4]-baseline["final"][4]),abs(minus["final"][4]-baseline["final"][4]))),
                "forward_side_effect":float(max(abs(plus["final"][0]-baseline["final"][0]),abs(minus["final"][0]-baseline["final"][0]))),
                "baseline_reason":baseline["reason"],"plus_reason":plus["reason"],"minus_reason":minus["reason"],
                "perturbation_immediate_failure":bool((plus["done"] or minus["done"]) and not baseline["done"]),
                "baseline_contact_step":baseline["contact_step"],"plus_contact_step":plus["contact_step"],"minus_contact_step":minus["contact_step"]}
            details.append(item);by_channel[name].append(item)
    decision=audit_decision(by_channel,cfg)
    grouped={}
    for key in ("layer","label","parent"):
        grouped[key]={value:audit_decision({name:[x for x in details if x[key]==value and x["channel"]==name] for name in CHANNELS},cfg)
                      for value in sorted({str(x[key]) for x in details})}
    report={"status":"PASS","artifact_role":"roll_controllability_development_audit","policy_hash":file_sha256(Path(a.policy)/"params.pkl"),
        "policy_version":manifest["policy_version"],"candidate_bank_sha256":file_sha256(a.candidate_bank),"xml_sha256":file_sha256(cfg.xml_path),
        "seed":a.seed,"states":len(rows),"state_ids":[row["id"] for row in rows],**decision,"by_group":grouped,
        "thresholds":{"action_delta":cfg.roll_controllability_action_delta,"horizon":cfg.roll_controllability_horizon,
            "pulse_steps":cfg.roll_controllability_pulse_steps,"min_roll_sensitivity":cfg.roll_controllability_min_roll_sensitivity,
            "min_roll_rate_sensitivity":cfg.roll_controllability_min_roll_rate_sensitivity,
            "min_beneficial_fraction":cfg.roll_controllability_min_beneficial_fraction,"max_pitch_side_effect":cfg.roll_controllability_max_pitch_side_effect},
        "termination_reasons":dict(Counter(x["baseline_reason"] for x in details)),"rows":details}
    save_json(out,report);print(json.dumps({k:v for k,v in report.items() if k!="rows"},indent=2))


if __name__=="__main__":main()
