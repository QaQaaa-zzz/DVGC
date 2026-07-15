"""Deterministic, no-training Flight-to-Landing handoff trace audit."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import jax
import numpy as np

from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config
from dvgc.entry import ENTRY_FEATURE_NAMES
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.rollout import restore_snapshot
from dvgc.runtime import build_inference


PHYSICAL_FEATURE_NAMES=("x","y","z","roll","pitch","yaw","vx","vy","vz","wx","wy","wz","steer","hip","knee","rearwheel_velocity")


def _quantiles(values):
    return {name:float(np.quantile(values,q)) for name,q in (("min",0),("p05",.05),("p50",.5),("p95",.95),("max",1))} if values else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy",required=True); p.add_argument("--candidate-bank",required=True)
    p.add_argument("--downstream-bank",required=True); p.add_argument("--output",required=True)
    p.add_argument("--seed",type=int,default=2500000); p.add_argument("--contact-window",type=int,default=3)
    a=p.parse_args(); output=Path(a.output)
    if output.exists(): raise SystemExit(f"Diagnostic output already exists: {output}")
    params,cfg_dict,manifest=load_bundle(a.policy,verify_files=True)
    cfg=load_config(overrides={**cfg_dict,"training_stage":"flight","domain_randomization":False,"obs_noise_enable":False,"use_bank_resets":False})
    candidates=SnapshotBank.load(a.candidate_bank); downstream=SnapshotBank.load(a.downstream_bank)
    rows=candidates.records_for_phase("flight",include_training_only=False)
    safe=downstream.records_for_phase("landing",final_labels=["safe"],include_training_only=False)
    env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank(),cert_bank=downstream)
    if env._safe_count!=len(safe) or not safe: raise SystemExit("Downstream Final-safe set was not loaded exactly")
    inference=build_inference(env,params,deterministic=True); step_fn=jax.jit(env.step)
    safe_z=np.asarray(jax.device_get(env._safe_features),np.float64); center=np.asarray(jax.device_get(env._safe_center),np.float64); scale=np.asarray(jax.device_get(env._safe_scale),np.float64)
    traces=[]
    for i,row in enumerate(rows):
        key=jax.random.PRNGKey(a.seed+i); state=restore_snapshot(env,row,key); steps=[]
        for t in range(int(cfg.branch_horizon)):
            key,action_key=jax.random.split(key); action,_=inference(state.obs,action_key); state=step_fn(state,action)
            if env._entry_matcher:
                feature=env._landing_entry_feature(
                    state.data,state.info["had_valid_landing"]>0,
                    state.info["contact_age"]>0,state.info["landing_entry_age"],
                )
            else:
                feature=env._physical_feature(state.data)
            feature=np.asarray(jax.device_get(feature),np.float64); z=(feature-center)/scale
            delta=safe_z-z[None,:]; squared=delta*delta; distances=np.sqrt(squared.sum(axis=1)); nearest=int(np.argmin(distances))
            valid=bool(float(np.asarray(jax.device_get(state.metrics["event/landing"])))>.5)
            chain=bool(float(np.asarray(jax.device_get(state.metrics["event/chain"])))>.5)
            steps.append({
                "step":t+1,"distance":float(distances[nearest]),"nearest_entry_id":safe[nearest]["id"],
                "valid_landing":valid,"chain":chain,
                "signed_normalized_delta":delta[nearest].tolist(),"squared_distance_contribution":squared[nearest].tolist(),
            })
            if float(np.asarray(jax.device_get(state.done)))>.5: break
        first=next((x["step"] for x in steps if x["valid_landing"]),None); minimum=min(steps,key=lambda x:x["distance"])
        contact=next((x for x in steps if x["step"]==first),None) if first is not None else None
        window=[{"relative_step":x["step"]-first,"distance":x["distance"],"nearest_entry_id":x["nearest_entry_id"]} for x in steps if first is not None and abs(x["step"]-first)<=int(a.contact_window)]
        chain_ever=bool(int(np.asarray(jax.device_get(state.info["chain_ever"]))))
        final=bool(int(np.asarray(jax.device_get(state.info["recovery_success"]))))
        traces.append({
            "candidate_id":row["id"],"candidate_kind":row.get("candidate_kind","unknown"),"flight_subinterval":row.get("flight_subinterval"),
            "valid_landing_ever":bool(any(x["valid_landing"] for x in steps)),"chain_ever":chain_ever,"final_recovery":final,
            "first_valid_contact_step":first,"minimum_distance":minimum["distance"],"nearest_entry_id":minimum["nearest_entry_id"],
            "minimum_distance_signed_normalized_delta":minimum["signed_normalized_delta"],
            "minimum_distance_squared_contribution":minimum["squared_distance_contribution"],
            "contact_distance":None if contact is None else contact["distance"],
            "contact_nearest_entry_id":None if contact is None else contact["nearest_entry_id"],
            "contact_squared_contribution":None if contact is None else contact["squared_distance_contribution"],
            "contact_window":window,"distance_trace":[x["distance"] for x in steps],
            "chain_trigger_count":sum(x["chain"] for x in steps),"chain_reward":float(cfg.coeff_chain_event*sum(x["chain"] for x in steps)),
            "termination_reason":END_REASON.get(int(np.asarray(jax.device_get(state.info["end_code"]))),"unknown"),"steps":len(steps),
        })
    table=Counter((int(x["chain_ever"]),int(x["final_recovery"])) for x in traces); finals=[x for x in traces if x["final_recovery"]]
    report={
        "status":"COMPLETED","policy_version":manifest["policy_version"],"candidate_bank":str(Path(a.candidate_bank).resolve()),
        "candidate_bank_sha256":file_sha256(a.candidate_bank),"downstream_bank":str(Path(a.downstream_bank).resolve()),
        "downstream_bank_sha256":file_sha256(a.downstream_bank),"downstream_policy_version":downstream.metadata.get("last_policy_version"),
        "downstream_tube_version":downstream.metadata.get("last_tube_version"),"downstream_final_safe_count":len(safe),
        "diagnostic_seed":int(a.seed),"matcher":{"feature_names":ENTRY_FEATURE_NAMES if env._entry_matcher else PHYSICAL_FEATURE_NAMES,"center":center.tolist(),"scale":scale.tolist(),"radius":float(env._safe_radius),"event":"fixed Landing-entry window AND safe_entry" if env._entry_matcher else "same-step valid_landing AND safe_entry"},
        "chain_final_table":{"chain0_final0":table[(0,0)],"chain0_final1":table[(0,1)],"chain1_final0":table[(1,0)],"chain1_final1":table[(1,1)]},
        "false_progress":table[(1,0)],"missed_success":table[(0,1)],
        "final_without_chain_rate":float(table[(0,1)]/len(finals)) if finals else 0.0,
        "final_without_valid_landing_count":sum(not x["valid_landing_ever"] for x in finals),
        "all_final_successes_have_valid_landing":all(x["valid_landing_ever"] for x in finals),
        "minimum_distance_distribution":_quantiles([x["minimum_distance"] for x in traces]),
        "contact_distance_distribution":_quantiles([x["contact_distance"] for x in traces if x["contact_distance"] is not None]),
        "chain_trigger_count":sum(x["chain_trigger_count"] for x in traces),"chain_reward_total":sum(x["chain_reward"] for x in traces),
        "termination_reasons":dict(Counter(x["termination_reason"] for x in traces)),"rows":traces,
    }
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("rows","matcher")},indent=2))


if __name__=="__main__": main()
