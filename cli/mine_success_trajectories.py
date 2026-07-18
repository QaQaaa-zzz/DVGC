"""Mine validated Flight snapshots from frozen-policy successful Final trajectories."""
from __future__ import annotations

import argparse,copy,hashlib,json
from collections import Counter
from pathlib import Path

import jax
import numpy as np

from cli.build_descent_entries import snapshot_identity
from cli.build_descent_support_repair import joint_ranges_ok
from dvgc.bank import SnapshotBank
from dvgc.candidate_geometry import TerrainClearanceSolver
from dvgc.certification import DYNAMICS_VARIANTS,branch_seed
from dvgc.composite import CanonicalEntryMatcher,CompositeSession
from dvgc.config import STAGE_ID,config_hash,file_sha256,load_config
from dvgc.descent_local import robust_scale
from dvgc.env import END_REASON,OrangeBikeDVGC
from dvgc.flight_augmentation import normalized_distance
from dvgc.policy import load_bundle
from dvgc.rollout import restore_snapshot
from dvgc.runtime import build_inference,save_json
from dvgc.snapshot_provenance import validate_snapshot_source_records
from dvgc.trajectory_mining import policy_state_complete,select_parent_balanced,select_trace_records,trajectory_parent_id
from dvgc.viability import ViabilityEnsemble

FEATURE_NAMES=("x","y","z","roll","pitch","yaw","vx","vy","vz","wx","wy","wz","steer","hip","knee","rearwheel_velocity")


def finite_record(row):
    arrays=[row[k] for k in ("qpos","qvel","ctrl","qacc_warmstart","physical_feature")]
    arrays.extend(v for v in row.get("policy_state",{}).values() if isinstance(v,(np.ndarray,list,tuple)))
    return all(np.isfinite(np.asarray(value)).all() for value in arrays)


def extraction_layer(origin,role):
    order={"early":0,"middle":1,"late":2};progress={"earliest":0,"middle":1,"late":2,"best_roll_margin":1}
    return ("early","middle","late")[max(order.get(str(origin),1),progress.get(str(role),1))]


def validate_candidate(row,env,step_fn,inference,solver,cfg,key):
    if not finite_record(row):return False,"nonfinite",None
    if not policy_state_complete(row):return False,"policy_state_incomplete",None
    if int(row.get("oracle_phase",-1))!=STAGE_ID["flight"] or not int(row.get("had_airborne",0)):
        return False,"flight_semantic",None
    if not joint_ranges_ok(solver.model,row["qpos"]):return False,"joint_range",None
    placement=solver.solve(row["qpos"],row["qvel"],row["ctrl"])
    if not placement.accepted or placement.root_z_shift>1e-7 or placement.robot_terrain_contacts:
        return False,"contact_or_penetration",placement
    state=restore_snapshot(env,row,key)
    for _ in range(int(cfg.descent_local_validation_steps)):
        key,action_key=jax.random.split(key);action,_=inference(state.obs,action_key);state=step_fn(state,action)
        values=(state.data.qpos,state.data.qvel,state.obs["state"],state.obs["privileged_state"])
        if not all(np.isfinite(np.asarray(jax.device_get(value))).all() for value in values):return False,"short_nonfinite",placement
        if bool(np.asarray(jax.device_get(state.done))):
            reason=END_REASON.get(int(np.asarray(jax.device_get(state.info["end_code"]))),"termination")
            return False,"short_"+reason,placement
    return True,"accepted",placement


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-bank",required=True);p.add_argument("--descent-policy",required=True)
    p.add_argument("--landing-policy",required=True);p.add_argument("--landing-entry-set",required=True)
    p.add_argument("--viability-model",required=True);p.add_argument("--output-bank",required=True)
    p.add_argument("--output-report",required=True);p.add_argument("--seed",type=int,required=True)
    p.add_argument("--branches-per-state",type=int,default=2);p.add_argument("--target",type=int,default=64)
    p.add_argument("--namespace",required=True);p.add_argument("--config",default="configs/default.json")
    a=p.parse_args();out=Path(a.output_bank);report_path=Path(a.output_report)
    if out.exists() or report_path.exists():raise SystemExit("Trajectory-mining output already exists")
    if not 1<=a.branches_per_state<=4 or not 1<=a.target<=64:raise SystemExit("Mining budget exceeds bounded protocol")
    dp,dc,dm=load_bundle(a.descent_policy,verify_files=True);lp,_,lm=load_bundle(a.landing_policy,verify_files=True)
    policy_hash=file_sha256(Path(a.descent_policy)/"params.pkl");base=SnapshotBank.load(a.base_bank)
    rows=base.records_for_phase("flight",include_training_only=False);old_sources=validate_snapshot_source_records(rows,base.metadata)
    if base.metadata.get("policy_hash")!=policy_hash:raise SystemExit("Mining base/current-policy mismatch")
    if base.metadata.get("landing_entry_set_sha256")!=file_sha256(a.landing_entry_set):raise SystemExit("Mining C_L mismatch")
    cfg0=load_config(a.config,{**dc,"training_stage":"flight","expert_chain_termination":False,
        "domain_randomization":False,"obs_noise_enable":False,"use_bank_resets":False})
    variants=[]
    for spec in DYNAMICS_VARIANTS:
        cfg=load_config(a.config,{**cfg0.to_dict(),**{k:v for k,v in spec.items() if k!="id"}})
        env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank(),cert_bank=SnapshotBank.load(a.landing_entry_set))
        inference={"flight":build_inference(env,dp,deterministic=True),"landing":build_inference(env,lp,deterministic=True)}
        variants.append((spec["id"],env,jax.jit(env.step),inference,CanonicalEntryMatcher(env,"flight",a.landing_entry_set)))
    gate_env=variants[0][1];gate_step=variants[0][2];gate_inference=variants[0][3]["flight"]
    solver=TerrainClearanceSolver(cfg0.xml_path,margin=0.0,max_correction=cfg0.flight_candidate_max_root_z_correction)
    existing=[np.asarray(row["physical_feature"],np.float64) for row in rows];_,scale=robust_scale(existing,1e-4)
    identities={snapshot_identity(row) for row in rows};rejected=Counter();raw_candidates=[];successful=[];end_reasons=Counter();empty_successful_traces=0
    for index,parent in enumerate(rows):
        for branch in range(int(a.branches_per_state)):
            variant,env,step,inference,matcher=variants[branch%len(variants)];seed=branch_seed(a.seed,index,branch);key=jax.random.PRNGKey(seed)
            initial=restore_snapshot(env,parent,key);session=CompositeSession(env,("flight","landing"),inference,{"flight":matcher},initial,key)
            trace=[{"step":0,"record":env.snapshot_record(initial,"flight")}] if int(np.asarray(jax.device_get(initial.info["phase"])))==STAGE_ID["flight"] else []
            for rollout_step in range(int(cfg0.branch_horizon)):
                state=session.step(step_fn=step,action_noise_std=float(cfg0.action_noise_std))
                if session.active_stage=="flight" and int(np.asarray(jax.device_get(state.info["phase"])))==STAGE_ID["flight"] and rollout_step%2==1:
                    trace.append({"step":rollout_step+1,"record":env.snapshot_record(state,"flight")})
                if bool(np.asarray(jax.device_get(state.done))):break
            final=bool(np.asarray(jax.device_get(session.state.info.get("recovery_success",0))));chain=bool(session.handoffs)
            end_reasons[END_REASON.get(int(np.asarray(jax.device_get(session.state.info.get("end_code",0)))),"unknown")]+=1
            if not (chain and final):continue
            trajectory=trajectory_parent_id(policy_hash,parent["id"],seed);successful.append({"trajectory_parent_id":trajectory,
                "candidate_id":parent["id"],"original_parent":parent.get("entry_source_id",parent.get("parent_candidate_id",parent["id"])),
                "branch_seed":seed,"dynamics_variant":variant,"trace_snapshots":len(trace)})
            if not trace:empty_successful_traces+=1
            for role,item in select_trace_records(trace,max_children=4,min_step_gap=2):
                row=item["record"];identifier=hashlib.sha256(f"mined:{trajectory}:{item['step']}".encode()).hexdigest()[:32]
                row.update({"id":identifier,"candidate_kind":"successful_trajectory_snapshot","trajectory_parent_id":trajectory,
                    "original_candidate_id":parent["id"],"original_candidate_parent":successful[-1]["original_parent"],
                    "parent_candidate_id":parent["id"],"entry_source_id":trajectory,"mining_branch_seed":seed,
                    "mining_seed_namespace":f"{a.namespace}:trajectory_mining","mining_dynamics_variant":variant,
                    "extraction_step":int(item["step"]),"extraction_role":role,
                    "descent_layer":extraction_layer(parent.get("descent_layer"),role),
                    "snapshot_source_policy_hash":policy_hash,"training_only":False,"bootstrap_eligible":True})
                ok,reason,placement=validate_candidate(row,gate_env,gate_step,gate_inference,solver,cfg0,jax.random.PRNGKey(seed^item["step"]))
                if not ok:rejected[reason]+=1;continue
                identity=snapshot_identity(row)
                if identity in identities:rejected["byte_duplicate"]+=1;continue
                feature=np.asarray(row["physical_feature"],np.float64);distance=normalized_distance(feature,existing,scale)
                if distance<float(cfg0.descent_local_normalized_dedup_distance):
                    nearest=int(np.argmin(np.linalg.norm((np.asarray(existing)-feature)/scale,axis=1)))
                    for dim in np.argsort(np.abs((existing[nearest]-feature)/scale))[:3]:rejected[f"duplicate_near_{FEATURE_NAMES[int(dim)]}"]+=1
                    rejected["normalized_duplicate"]+=1;continue
                row.update({"snapshot_identity_sha256":identity,"normalized_nearest_neighbor_distance":distance,
                    "root_z_shift_m":0.0,"terrain_clearance_m":placement.clearance,"wheel_clearance_m":placement.wheel_clearance,
                    "nonwheel_clearance_m":placement.nonwheel_clearance,"robot_terrain_contacts":placement.robot_terrain_contacts})
                identities.add(identity);existing.append(feature);raw_candidates.append(row)
    if raw_candidates:
        model=ViabilityEnsemble.load(a.viability_model);mean,std,support=model.predict_records(raw_candidates)
        safe=[row for row in rows if row.get("stable_safe")];boundary=[row for row in rows if row["final"]["label"]=="boundary"]
        safe_center=np.median([row["physical_feature"] for row in safe],axis=0) if safe else np.zeros(16)
        for row,pred,unc,sup in zip(raw_candidates,mean,std,support):
            f=np.asarray(row["physical_feature"],np.float64);roll_margin=np.exp(-abs(f[3]-safe_center[3])/max(scale[3],1e-6)-.25*abs(f[9]-safe_center[9])/max(scale[9],1e-6))
            bd=normalized_distance(f,[x["physical_feature"] for x in boundary],scale) if boundary else 1.0
            row.update({"viability_probability":float(pred),"viability_disagreement":float(unc),"viability_support":float(sup),
                "boundary_distance":float(bd),"roll_margin_score":float(roll_margin),
                "mining_rank_score":float(pred+.75*unc+.25*(1-sup)+.20*roll_margin+.10/(1+bd))})
    masses={"middle":.40,"late":.40,"early":.20};selected=select_parent_balanced(raw_candidates,target=a.target,masses=masses,parent_cap=4)
    base_records=copy.deepcopy(rows)
    fallback=old_sources[0] if len(old_sources)==1 else None
    for row in base_records:
        row["snapshot_source_policy_hash"]=row.get("snapshot_source_policy_hash",fallback)
    observed_sources=set(old_sources)|({policy_hash} if selected else set())
    metadata=copy.deepcopy(base.metadata);metadata.update({"bank_role":"successful_trajectory_mined_candidates",
        "policy_hash":policy_hash,"snapshot_source_policy_hashes":sorted(observed_sources),
        "trajectory_mining_seed":a.seed,"trajectory_mining_seed_namespace":f"{a.namespace}:trajectory_mining",
        "trajectory_mining_branches_per_state":a.branches_per_state,"trajectory_mining_source_bank_sha256":file_sha256(a.base_bank),
        "viability_model_sha256":file_sha256(a.viability_model),"candidate_config_hash":config_hash(cfg0)})
    result=SnapshotBank(base_records+selected,metadata);validate_snapshot_source_records(result.records_for_phase("flight",include_training_only=False),metadata);result.save(out)
    selected_parents={row["trajectory_parent_id"] for row in selected}
    report={"status":"PASS" if selected else "FAIL","artifact_role":"development_success_trajectory_mining",
        "seed":a.seed,"seed_namespace":f"{a.namespace}:trajectory_mining","branches_per_state":a.branches_per_state,
        "source_states":len(rows),"rollouts":len(rows)*a.branches_per_state,"successful_trajectories":len(successful),
        "successful_trajectory_parents":len({x["trajectory_parent_id"] for x in successful}),"raw_valid_snapshots":len(raw_candidates),
        "successful_trajectories_with_flight_snapshots":len(successful)-empty_successful_traces,"empty_successful_flight_traces":empty_successful_traces,
        "selected_snapshots":len(selected),"selected_trajectory_parents":len(selected_parents),
        "maximum_children_per_trajectory_parent":max(Counter(row["trajectory_parent_id"] for row in selected).values(),default=0),
        "selected_layers":dict(Counter(row["descent_layer"] for row in selected)),"proposal_target_masses":masses,
        "original_parent_count":len({row["original_candidate_parent"] for row in selected}),"rejections":dict(rejected),
        "rollout_end_reasons":dict(end_reasons),"all_state_unique":len(identities)==len(rows)+len(raw_candidates),
        "prediction_can_promote_empirical_safe":False,"output_states":len(result.records),"output_bank_sha256":file_sha256(out),
        "provenance":{"policy_hash":policy_hash,"source_bank_sha256":file_sha256(a.base_bank),"xml_sha256":file_sha256(cfg0.xml_path),
            "landing_entry_set_sha256":file_sha256(a.landing_entry_set),"landing_policy_hash":file_sha256(Path(a.landing_policy)/"params.pkl"),
            "viability_model_sha256":file_sha256(a.viability_model),"descent_policy_version":dm["policy_version"],"landing_policy_version":lm["policy_version"]}}
    save_json(report_path,report);print(json.dumps(report,indent=2))
    if report["status"]!="PASS":raise SystemExit(2)


if __name__=="__main__":main()
