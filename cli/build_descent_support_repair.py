"""Build one bounded, parent-diverse descent support-repair candidate pool."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from cli.build_descent_entries import snapshot_identity
from dvgc.bank import SnapshotBank
from dvgc.candidate_geometry import TerrainClearanceSolver
from dvgc.certification import branch_seed
from dvgc.config import STAGE_ID, config_hash, file_sha256, load_config
from dvgc.descent_local import robust_scale, tangent_factor
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.flight_augmentation import apply_tangent, normalized_distance
from dvgc.runtime import save_json


def parent_key(row):
    return str(row.get("entry_source_id", row.get("parent_candidate_id", row["id"])))


def proposal_group(row, safe_parents):
    label=str(row["final"]["label"]);parent=parent_key(row)
    if parent not in safe_parents and label in {"boundary","unknown"}:return "new_parent"
    if label=="boundary":return "boundary"
    return "safe_continuity"


def choose_parent(rows, source_children, row_children, cap, rng):
    available=[row for row in rows if source_children[parent_key(row)]<cap]
    if not available:return None
    minimum=min(source_children[parent_key(row)] for row in available)
    sources=sorted({parent_key(row) for row in available if source_children[parent_key(row)]==minimum})
    source=sources[int(rng.integers(len(sources)))]
    candidates=[row for row in available if parent_key(row)==source]
    minimum=min(row_children[row["id"]] for row in candidates)
    candidates=[row for row in candidates if row_children[row["id"]]==minimum]
    return candidates[int(rng.integers(len(candidates)))]


def joint_ranges_ok(model,qpos):
    for joint in range(model.njnt):
        if int(model.jnt_type[joint])==mujoco.mjtJoint.mjJNT_FREE or not bool(model.jnt_limited[joint]):continue
        address=int(model.jnt_qposadr[joint]);low,high=model.jnt_range[joint]
        if not float(low)<=float(qpos[address])<=float(high):return False
    return True


def finite_state(state):
    return all(np.isfinite(np.asarray(jax.device_get(value))).all()
               for value in (state.data.qpos,state.data.qvel,state.obs["state"],state.obs["privileged_state"]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-bank",required=True);parser.add_argument("--policy",required=True)
    parser.add_argument("--landing-entry-set",required=True);parser.add_argument("--output-bank",required=True)
    parser.add_argument("--output-report",required=True);parser.add_argument("--config",default="configs/default.json")
    parser.add_argument("--seed",type=int,required=True);parser.add_argument("--target",type=int,default=64)
    parser.add_argument("--proposal-budget",type=int,default=3000);parser.add_argument("--parent-cap",type=int,default=4)
    args=parser.parse_args();out=Path(args.output_bank)
    if out.exists() or Path(args.output_report).exists():raise SystemExit("Support-repair output already exists")
    if not 1<=args.target<=64:raise SystemExit("Support-repair target must be in [1,64]")
    cfg=load_config(args.config,{"training_stage":"flight","domain_randomization":False,"obs_noise_enable":False,
                                "use_bank_resets":False,"expert_chain_termination":False})
    base=SnapshotBank.load(args.base_bank);rows=base.records_for_phase("flight",include_training_only=False)
    policy_hash=file_sha256(Path(args.policy)/"params.pkl")
    if base.metadata.get("policy_hash")!=policy_hash:raise SystemExit("Support-repair policy/base-bank mismatch")
    if base.metadata.get("landing_entry_set_sha256")!=file_sha256(args.landing_entry_set):raise SystemExit("Support-repair C_L mismatch")
    if any(int(row.get("oracle_phase",-1))!=STAGE_ID["flight"] for row in rows):raise SystemExit("Non-Flight base state")
    safe_parents={parent_key(row) for row in rows if row["final"]["label"]=="safe"}
    support=[row for row in rows if row["final"]["label"] in {"safe","boundary","unknown"}]
    grouped=defaultdict(list)
    for row in support:grouped[proposal_group(row,safe_parents)].append(row)
    if any(not grouped[name] for name in ("new_parent","boundary","safe_continuity")):
        raise SystemExit("Support-repair source groups are incomplete")

    env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank());step=jax.jit(env.step)
    solver=TerrainClearanceSolver(cfg.xml_path,margin=0.0,max_correction=cfg.flight_candidate_max_root_z_correction)
    qdim=len(support[0]["qpos"])-7;vdim=len(support[0]["qvel"]);allowed=np.zeros(6+qdim+vdim,bool);allowed[3:5]=True
    for name in ("hip_joint","knee_joint"):
        try:joint=solver.model.joint(name).id
        except KeyError:continue
        allowed[6+int(solver.model.jnt_qposadr[joint])-7]=True
        allowed[6+qdim+int(solver.model.jnt_dofadr[joint])-6]=True
    allowed[6+qdim+np.asarray([0,2,3,4])]=True
    for name in ("frontwheel_joint","rearwheel_joint"):
        try:allowed[6+qdim+int(solver.model.jnt_dofadr[solver.model.joint(name).id])-6]=True
        except KeyError:pass
    factor=tangent_factor(support,allowed);_,feature_scale=robust_scale([row["physical_feature"] for row in support],1e-4)
    existing=[np.asarray(row["physical_feature"],np.float64) for row in rows]
    identities={snapshot_identity(row) for row in rows};records=[copy.deepcopy(row) for row in rows]
    rng=np.random.default_rng(args.seed);source_children=Counter();row_children=Counter();accepted=Counter();rejected=Counter()
    group_weights={"new_parent":.50,"boundary":.30,"safe_continuity":.20}
    scales={"new_parent":float(cfg.descent_local_boundary_covariance_scale),
            "boundary":float(cfg.descent_local_boundary_covariance_scale),
            "safe_continuity":float(cfg.descent_local_safe_covariance_scale)}
    attempts=0
    while sum(accepted.values())<args.target and attempts<args.proposal_budget:
        attempts+=1
        available=[name for name in group_weights if choose_parent(grouped[name],source_children,row_children,args.parent_cap,np.random.default_rng(args.seed)) is not None]
        if not available:break
        weights=np.asarray([group_weights[name]/(1+accepted[name]) for name in available],np.float64)
        group=str(rng.choice(available,p=weights/weights.sum()))
        parent=choose_parent(grouped[group],source_children,row_children,args.parent_cap,rng)
        if parent is None:rejected["parent_cap"]+=1;continue
        latent=rng.normal(size=factor.shape[1]);raw_norm=float(np.linalg.norm(latent));latent*=min(1.0,2.0/max(raw_norm,1e-9))
        delta=scales[group]*(factor@latent);qpos,qvel=apply_tangent(parent["qpos"],parent["qvel"],delta)
        if not joint_ranges_ok(solver.model,qpos):rejected["joint_range"]+=1;continue
        placement=solver.solve(qpos,qvel,parent["ctrl"])
        if not placement.accepted or placement.root_z_shift>1e-7 or placement.robot_terrain_contacts:
            rejected["contact_or_penetration"]+=1;continue
        key=jax.random.PRNGKey(branch_seed(args.seed,attempts,0))
        state=env.reset_from_snapshot(jp.asarray(qpos,jp.float32),jp.asarray(qvel,jp.float32),jp.asarray(parent["ctrl"],jp.float32),key,
                                      jp.asarray(STAGE_ID["flight"],jp.int32),jp.asarray(parent.get("had_airborne",1),jp.int32),
                                      jp.asarray(parent.get("had_valid_landing",0),jp.int32),jp.asarray(parent.get("contact_age",0),jp.int32))
        if not finite_state(state) or int(np.asarray(state.info["phase"]))!=STAGE_ID["flight"]:
            rejected["nonfinite_or_phase"]+=1;continue
        probe=state;failure=None
        for _ in range(int(cfg.descent_local_validation_steps)):
            probe=step(probe,jp.zeros(env.action_size,jp.float32))
            if not finite_state(probe):failure="nonfinite";break
            if float(np.asarray(probe.done))>.5:failure=END_REASON.get(int(np.asarray(probe.info["end_code"])),"termination");break
        if failure:rejected["short_"+failure]+=1;continue
        feature=np.asarray(jax.device_get(env._physical_feature(state.data)),np.float64)
        nn=normalized_distance(feature,existing,feature_scale)
        if nn<float(cfg.descent_local_normalized_dedup_distance):rejected["normalized_duplicate"]+=1;continue
        record=env.snapshot_record(state,"flight");record["policy_state"]=copy.deepcopy(parent.get("policy_state",{}))
        identity=snapshot_identity(record)
        if identity in identities:rejected["byte_duplicate"]+=1;continue
        source=parent_key(parent);identifier=hashlib.sha256(f"descent-support:{args.seed}:{attempts}:{parent['id']}".encode()).hexdigest()[:32]
        bootstrap="provisional_safe" if parent["final"]["label"]=="safe" else "boundary"
        record.update({"id":identifier,"candidate_kind":"descent_support_repair_proposal","parent_candidate_id":parent["id"],
                       "entry_source_id":source,"descent_layer":parent.get("descent_layer"),"bootstrap_group":bootstrap,
                       "local_bootstrap_eligible":True,"bootstrap_eligible":True,"training_only":False,
                       "support_repair_group":group,"candidate_generation_seed":args.seed,"candidate_proposal_index":attempts,
                       "perturbation_tangent":delta.astype(np.float32),"sampling_latent_norm":raw_norm,
                       "snapshot_identity_sha256":identity,"normalized_nearest_neighbor_distance":nn,
                       "root_z_shift_m":0.0,"terrain_clearance_m":placement.clearance,
                       "wheel_clearance_m":placement.wheel_clearance,"nonwheel_clearance_m":placement.nonwheel_clearance})
        records.append(record);existing.append(feature);identities.add(identity);source_children[source]+=1;row_children[parent["id"]]+=1;accepted[group]+=1
    metadata=copy.deepcopy(base.metadata);metadata.update({"bank_role":"descent_candidate_support_repair","policy_hash":policy_hash,
        "descent_policy_hash":policy_hash,"support_repair_seed":args.seed,"support_repair_target":args.target,
        "support_repair_parent_cap":args.parent_cap,"source_bank_sha256":file_sha256(args.base_bank),
        "candidate_config_hash":config_hash(cfg),"xml_sha256":file_sha256(cfg.xml_path)})
    SnapshotBank(records,metadata).save(out)
    children=[row for row in records if row.get("candidate_kind")=="descent_support_repair_proposal"]
    minimum_children=min(args.target,12);minimum_parents=min(6,len({parent_key(row) for row in support}))
    quality=(len(children)>=minimum_children and len(source_children)>=minimum_parents
             and max(source_children.values(),default=0)<=args.parent_cap and len(identities)==len(records))
    report={"status":"PASS" if quality else "FAIL","seed":args.seed,"target":args.target,"proposal_budget":args.proposal_budget,
            "attempts":attempts,"base_states":len(rows),"children":len(children),"total_states":len(records),
            "minimum_children":minimum_children,"minimum_parents":minimum_parents,
            "accepted":dict(accepted),"rejections":dict(rejected),"unique_child_parents":len(source_children),
            "maximum_children_per_parent":max(source_children.values(),default=0),"layers":dict(Counter(row.get("descent_layer") for row in children)),
            "labels_of_parents":dict(Counter(next(x for x in rows if x["id"]==row["parent_candidate_id"])["final"]["label"] for row in children)),
            "all_state_unique":len(identities)==len(records),"bank_sha256":file_sha256(out),
            "provenance":{"base_bank_sha256":file_sha256(args.base_bank),"policy_hash":policy_hash,
                          "c_l_sha256":file_sha256(args.landing_entry_set),"xml_sha256":file_sha256(cfg.xml_path)}}
    save_json(args.output_report,report);print(json.dumps(report,indent=2))
    if report["status"]!="PASS":raise SystemExit(2)


if __name__=="__main__":main()
