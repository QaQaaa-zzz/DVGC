"""Build a layered, physically validated candidate-guided descent reset pool."""
from __future__ import annotations

import argparse,copy,hashlib,json
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from cli.build_descent_entries import snapshot_identity
from dvgc.bank import SnapshotBank
from dvgc.candidate_geometry import TerrainClearanceSolver
from dvgc.composite import CanonicalEntryMatcher
from dvgc.config import STAGE_ID,config_hash,file_sha256,load_config
from dvgc.descent_local import BOOTSTRAP_GROUPS,balanced_parent,difficulty_layers,robust_scale,tangent_factor
from dvgc.env import END_REASON,OrangeBikeDVGC
from dvgc.flight_augmentation import apply_tangent,normalized_distance
from dvgc.rollout import restore_snapshot
from dvgc.runtime import save_json


def joint_ranges_ok(model,qpos):
    for joint in range(model.njnt):
        if int(model.jnt_type[joint])==mujoco.mjtJoint.mjJNT_FREE or not bool(model.jnt_limited[joint]): continue
        address=int(model.jnt_qposadr[joint]); low,high=model.jnt_range[joint]
        if not float(low)<=float(qpos[address])<=float(high): return False
    return True


def finite_state(state):
    return all(np.isfinite(np.asarray(jax.device_get(value))).all() for value in (state.data.qpos,state.data.qvel,state.obs["state"],state.obs["privileged_state"]))


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--certified-candidates",required=True); p.add_argument("--flight-bank",required=True); p.add_argument("--successful-rollouts",required=True); p.add_argument("--landing-entry-set",required=True); p.add_argument("--output-bank",required=True); p.add_argument("--output-report",required=True); p.add_argument("--config",default="configs/default.json"); p.add_argument("--seed",type=int,default=7400000); a=p.parse_args(); out=Path(a.output_bank)
    if out.exists() or Path(a.output_report).exists(): raise SystemExit("Local descent pool output already exists")
    cfg=load_config(a.config,{"training_stage":"flight","domain_randomization":False,"obs_noise_enable":False,"use_bank_resets":False,"expert_chain_termination":False})
    if a.seed!=int(cfg.descent_local_candidate_seed): raise SystemExit("Local candidate seed does not match config")
    source=SnapshotBank.load(a.certified_candidates); rows=source.records_for_phase("flight",include_training_only=False)
    if Counter(row["final"]["label"] for row in rows)!={"safe":3,"boundary":22,"dead":33,"unknown":12}: raise SystemExit("Unexpected 70-state C_D diagnostic baseline")
    env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank()); step=jax.jit(env.step); matcher=CanonicalEntryMatcher(env,"flight",a.landing_entry_set); solver=TerrainClearanceSolver(cfg.xml_path,margin=0.0,max_correction=cfg.flight_candidate_max_root_z_correction)
    distances={}
    for i,row in enumerate(rows): distances[row["id"]]=matcher.match(restore_snapshot(env,row,jax.random.PRNGKey(a.seed+i)))[1]
    layers=difficulty_layers(rows,distances); records=[]; parents={group:[] for group in BOOTSTRAP_GROUPS}; training_identities=set()
    for original in rows:
        row=copy.deepcopy(original); label=row["final"]["label"]; group="provisional_safe" if label=="safe" else "boundary" if label=="boundary" else "diagnostic_only"
        identity=snapshot_identity(row); eligible=group in BOOTSTRAP_GROUPS and identity not in training_identities
        if eligible: training_identities.add(identity)
        row.update({"candidate_kind":"descent_diagnostic_anchor","old_policy_label":label,"old_policy_hash":source.metadata["descent_policy_hash"],"bootstrap_group":group,"local_bootstrap_eligible":eligible,"duplicate_diagnostic_only":group in BOOTSTRAP_GROUPS and not eligible,"descent_layer":layers[row["id"]],"distance_to_c_l":distances[row["id"]],"reset_source":"descent_local_candidate","snapshot_identity_sha256":identity}); records.append(row)
        if eligible: parents[group].append(row)
    build=json.loads(Path(a.successful_rollouts).read_text()); successful_ids=sorted({r["candidate_id"] for r in build["rollouts"] if r["chain"] and r["final"]}); flight=SnapshotBank.load(a.flight_bank); flight_by={r["id"]:r for r in flight.records_for_phase("flight",include_training_only=False)}
    for identifier in successful_ids:
        if identifier not in flight_by: continue
        row=copy.deepcopy(flight_by[identifier]); state=restore_snapshot(env,row,jax.random.PRNGKey(a.seed+len(records))); distance=matcher.match(state)[1]; identity=snapshot_identity(row); eligible=identity not in training_identities
        if eligible: training_identities.add(identity)
        row.update({"candidate_kind":"descent_successful_anchor","old_policy_label":"successful_rollout_anchor","old_policy_hash":source.metadata["descent_policy_hash"],"bootstrap_group":"successful_anchor","local_bootstrap_eligible":eligible,"duplicate_diagnostic_only":not eligible,"descent_layer":"late","distance_to_c_l":distance,"reset_source":"descent_local_candidate","entry_source_id":identifier,"proposal_step":0,"snapshot_identity_sha256":identity}); records.append(row)
        if eligible: parents["successful_anchor"].append(row)
    support=parents["provisional_safe"]+parents["boundary"]+parents["successful_anchor"]
    qdim=len(support[0]["qpos"])-7; vdim=len(support[0]["qvel"]); allowed=np.zeros(6+qdim+vdim,bool); allowed[3:5]=True
    model=solver.model
    for name in ("hip_joint","knee_joint"):
        try: joint=model.joint(name).id
        except KeyError: continue
        allowed[6+int(model.jnt_qposadr[joint])-7]=True; allowed[6+qdim+int(model.jnt_dofadr[joint])-6]=True
    allowed[6+qdim+np.asarray([0,2,3,4])]=True
    for name in ("frontwheel_joint","rearwheel_joint"):
        try: allowed[6+qdim+int(model.jnt_dofadr[model.joint(name).id])-6]=True
        except KeyError: pass
    factor=tangent_factor(support,allowed); feature_center,feature_scale=robust_scale([row["physical_feature"] for row in support],1e-4); existing=[np.asarray(row["physical_feature"],np.float64) for row in records]
    identities={snapshot_identity(row) for row in records}; rng=np.random.default_rng(a.seed); children=Counter(); accepted=Counter(); rejected=Counter(); attempts=0
    total_children=int(cfg.descent_local_target_children); cap=int(cfg.descent_local_max_children_per_parent)
    safe_target=min(round(total_children*cfg.descent_local_reset_safe_mass),len(parents["provisional_safe"])*cap); anchor_target=min(round(total_children*cfg.descent_local_reset_anchor_mass),len(parents["successful_anchor"])*cap)
    targets={"provisional_safe":safe_target,"successful_anchor":anchor_target,"boundary":total_children-safe_target-anchor_target}
    scales={"provisional_safe":cfg.descent_local_safe_covariance_scale,"boundary":cfg.descent_local_boundary_covariance_scale,"successful_anchor":cfg.descent_local_anchor_covariance_scale}
    while sum(accepted.values())<int(cfg.descent_local_target_children) and attempts<int(cfg.descent_local_proposal_budget):
        attempts+=1; deficits={group:max(0,targets[group]-accepted[group]) for group in BOOTSTRAP_GROUPS}; available=[g for g in BOOTSTRAP_GROUPS if deficits[g] and parents[g]]
        if not available: break
        weights=np.asarray([deficits[g] for g in available],np.float64); group=str(rng.choice(available,p=weights/weights.sum())); parent=balanced_parent(parents[group],children,rng,cfg.descent_local_max_children_per_parent)
        if parent is None: rejected["parent_cap"]+=1; continue
        latent=rng.normal(size=factor.shape[1]); norm=float(np.linalg.norm(latent)); latent*=min(1.0,2.0/max(norm,1e-9)); delta=float(scales[group])*(factor@latent); qpos,qvel=apply_tangent(parent["qpos"],parent["qvel"],delta)
        if not joint_ranges_ok(model,qpos): rejected["joint_range"]+=1; continue
        placement=solver.solve(qpos,qvel,parent["ctrl"])
        if not placement.accepted or placement.root_z_shift>1e-7 or placement.robot_terrain_contacts: rejected["contact_or_penetration"]+=1; continue
        key=jax.random.PRNGKey(a.seed+attempts); state=env.reset_from_snapshot(jp.asarray(qpos,jp.float32),jp.asarray(qvel,jp.float32),jp.asarray(parent["ctrl"],jp.float32),key,jp.asarray(STAGE_ID["flight"],jp.int32),jp.asarray(1,jp.int32),jp.asarray(0,jp.int32),jp.asarray(0,jp.int32))
        if not finite_state(state) or int(np.asarray(state.info["phase"]))!=STAGE_ID["flight"]: rejected["nonfinite_or_phase"]+=1; continue
        probe=state; failure=None
        for _ in range(int(cfg.descent_local_validation_steps)):
            probe=step(probe,jp.zeros(env.action_size,jp.float32))
            if not finite_state(probe): failure="nonfinite"; break
            if float(np.asarray(probe.done))>.5: failure=END_REASON.get(int(np.asarray(probe.info["end_code"])),"termination"); break
        if failure: rejected["short_"+failure]+=1; continue
        feature=np.asarray(jax.device_get(env._physical_feature(state.data)),np.float64); nn=normalized_distance(feature,existing,feature_scale)
        if nn<float(cfg.descent_local_normalized_dedup_distance): rejected["normalized_duplicate"]+=1; continue
        record=env.snapshot_record(state,"flight"); identity=snapshot_identity(record)
        if identity in identities: rejected["byte_duplicate"]+=1; continue
        identifier=hashlib.sha256(f"descent-local:{a.seed}:{attempts}:{parent['id']}".encode()).hexdigest()[:32]; distance=matcher.match(state)[1]
        record.update({"id":identifier,"candidate_kind":"descent_local_proposal","old_policy_label":"unlabeled_candidate","old_policy_hash":source.metadata["descent_policy_hash"],"bootstrap_group":group,"local_bootstrap_eligible":True,"descent_layer":parent["descent_layer"],"distance_to_c_l":distance,"reset_source":"descent_local_candidate","parent_candidate_id":parent["id"],"entry_source_id":parent.get("entry_source_id",parent["id"]),"perturbation_tangent":delta.astype(np.float32),"sampling_latent_norm":norm,"candidate_generation_seed":a.seed,"candidate_proposal_index":attempts,"snapshot_identity_sha256":identity,"root_z_shift_m":0.0,"terrain_clearance_m":placement.clearance,"wheel_clearance_m":placement.wheel_clearance,"nonwheel_clearance_m":placement.nonwheel_clearance,"normalized_nearest_neighbor_distance":nn,"bootstrap_eligible":True,"training_only":False})
        records.append(record); existing.append(feature); identities.add(identity); children[parent["id"]]+=1; accepted[group]+=1
    metadata=copy.deepcopy(source.metadata); metadata.update({"bank_role":"candidate_guided_local_bootstrap","source_certified_bank":str(Path(a.certified_candidates).resolve()),"source_certified_bank_sha256":file_sha256(a.certified_candidates),"flight_bank_sha256":file_sha256(a.flight_bank),"landing_entry_set_sha256":file_sha256(a.landing_entry_set),"local_candidate_seed":a.seed,"candidate_config_hash":config_hash(cfg),"reset_group_masses":{"provisional_safe":cfg.descent_local_reset_safe_mass,"boundary":cfg.descent_local_reset_boundary_mass,"successful_anchor":cfg.descent_local_reset_anchor_mass},"diagnostic_audit_seed_consumed":7200000})
    SnapshotBank(records,metadata).save(out); parent_counts=Counter(row.get("parent_candidate_id") for row in records if row["candidate_kind"]=="descent_local_proposal"); source_counts=Counter(row.get("entry_source_id") for row in records if row.get("local_bootstrap_eligible")); base_parent_count=sum(len(parents[g]) for g in BOOTSTRAP_GROUPS); support_complete=sum(accepted.values())>=base_parent_count and all(accepted[g]>0 for g in BOOTSTRAP_GROUPS)
    report={"status":"PASS" if support_complete else "FAIL","target_reached":sum(accepted.values())==int(cfg.descent_local_target_children),"minimum_support_children":base_parent_count,"records":len(records),"diagnostic_records":sum(not row.get("local_bootstrap_eligible") for row in records),"eligible_records":sum(bool(row.get("local_bootstrap_eligible")) for row in records),"base_groups":{g:sum(row["bootstrap_group"]==g and row["candidate_kind"]!="descent_local_proposal" and row.get("local_bootstrap_eligible") for row in records) for g in BOOTSTRAP_GROUPS},"accepted_children":dict(accepted),"targets":targets,"attempts":attempts,"rejections":dict(rejected),"layers":dict(Counter(row["descent_layer"] for row in records if row.get("local_bootstrap_eligible"))),"unique_parent_candidates":len(parent_counts),"maximum_children_per_parent":max(parent_counts.values(),default=0),"unique_source_parents":len(source_counts),"maximum_records_per_source":max(source_counts.values(),default=0),"bank_sha256":file_sha256(out),"provenance":{"source_bank_sha256":file_sha256(a.certified_candidates),"flight_bank_sha256":file_sha256(a.flight_bank),"c_l_sha256":file_sha256(a.landing_entry_set),"xml_sha256":file_sha256(cfg.xml_path),"seed":a.seed}}
    save_json(a.output_report,report); print(json.dumps(report,indent=2));
    if report["status"]!="PASS": raise SystemExit(2)


if __name__=="__main__": main()
