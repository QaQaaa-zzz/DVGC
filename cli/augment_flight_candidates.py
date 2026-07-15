"""Build a Flight bank from fixed reference anchors plus local diversification."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from dvgc.bank import SnapshotBank
from dvgc.candidate_geometry import TerrainClearanceSolver
from dvgc.config import STAGE_ID, config_hash, file_sha256, load_config
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.flight_augmentation import (
    REGIONS, apply_tangent, clone_as_anchors, covariance_factor, feature_scale,
    interpolate_state, normalized_distance, proportional_quotas,
)
from dvgc.reference import ReferenceTrajectory


def _finite(state):
    values=(state.data.qpos,state.data.qvel,state.reward,state.obs["state"],state.obs["privileged_state"])
    return all(np.isfinite(np.asarray(jax.device_get(value))).all() for value in values)


def _joint_ranges_ok(model, qpos):
    for joint in range(model.njnt):
        if int(model.jnt_type[joint])==0 or not bool(model.jnt_limited[joint]):
            continue
        address=int(model.jnt_qposadr[joint]); low,high=model.jnt_range[joint]
        if not float(low)<=float(qpos[address])<=float(high): return False
    return True


def _region_reference_counts(reference, anchors, window):
    indices=range(anchors.takeoff_end,anchors.landing_start+1)
    return {
        "ascent":sum(i<anchors.apex-window for i in indices),
        "apex":sum(anchors.apex-window<=i<=anchors.apex+window for i in indices),
        "descent":sum(i>anchors.apex+window for i in indices),
    }


def _quantiles(values):
    if not values: return None
    return {name:float(np.quantile(values,q)) for name,q in (("min",0),("p50",.5),("p95",.95),("max",1))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anchor-bank",required=True); p.add_argument("--output-bank",required=True)
    p.add_argument("--config",default="configs/default.json"); p.add_argument("--reference",default="data/reference_jump.csv")
    p.add_argument("--target",type=int,default=160); p.add_argument("--seed",type=int,default=17001)
    p.add_argument("--attempt-budget",type=int,default=300); p.add_argument("--allow-partial",action="store_true")
    a=p.parse_args(); output=Path(a.output_bank); anchor_path=Path(a.anchor_bank)
    if anchor_path.resolve()==output.resolve(): raise SystemExit("Augmented output must not overwrite the anchor bank")
    cfg=load_config(a.config,{"training_stage":"flight","use_bank_resets":False,"domain_randomization":False,"obs_noise_enable":False})
    if int(a.seed)!=int(cfg.flight_augmentation_seed): raise SystemExit("Generation seed must match the configured independent seed")
    anchors_bank=SnapshotBank.load(anchor_path); source_rows=anchors_bank.records_for_phase("flight",include_training_only=False)
    if len(source_rows)<80: raise SystemExit(f"At least 80 valid reference anchors required, got {len(source_rows)}")
    source_sha=file_sha256(anchor_path)
    if output.exists():
        bank=SnapshotBank.load(output)
        if bank.metadata.get("augmentation_anchor_bank_sha256")!=source_sha or int(bank.metadata.get("augmentation_seed",-1))!=a.seed:
            raise SystemExit("Existing augmented bank has incompatible anchor provenance")
    else:
        bank=SnapshotBank(clone_as_anchors(source_rows),copy.deepcopy(anchors_bank.metadata))
        bank.metadata.update({
            "augmentation_anchor_bank":str(anchor_path),"augmentation_anchor_bank_sha256":source_sha,
            "augmentation_seed":int(a.seed),"augmentation_rng_state":None,"augmentation_history":[],
        })
    rows=bank.records_for_phase("flight",include_training_only=False)
    anchor_rows=[row for row in rows if row["candidate_kind"]=="reference_anchor"]
    if len(anchor_rows)!=len(source_rows): raise SystemExit("Fixed anchor subset changed")
    for original,fixed in zip(source_rows,anchor_rows):
        for key in ("qpos","qvel","ctrl","qacc_warmstart","physical_feature"):
            if not np.array_equal(original[key],fixed[key]): raise SystemExit(f"Anchor state changed: {fixed['id']}:{key}")

    reference=ReferenceTrajectory.load(a.reference); ref_anchors=reference.anchors(); window=int(cfg.flight_candidate_apex_window_steps)
    reference_counts=_region_reference_counts(reference,ref_anchors,window); quotas=proportional_quotas(reference_counts,int(a.target))
    anchor_counts=Counter(row["flight_subinterval"] for row in anchor_rows)
    if any(anchor_counts[name]>quotas[name] for name in REGIONS): raise SystemExit("Anchor count exceeds an automatic region quota")
    factors={name:covariance_factor([row for row in anchor_rows if row["flight_subinterval"]==name]) for name in REGIONS}
    ordered={name:sorted([row for row in anchor_rows if row["flight_subinterval"]==name],key=lambda row:int(row["source_index"])) for name in REGIONS}
    segment_features={name:np.asarray([row["physical_feature"] for row in ordered[name]],np.float64) for name in REGIONS}
    scale=feature_scale(anchor_rows); existing_features=[np.asarray(row["physical_feature"],np.float64) for row in rows]
    children=Counter(row.get("parent_anchor_id") for row in rows if row["candidate_kind"]=="local_augmented")

    rng=np.random.default_rng(int(a.seed))
    if bank.metadata.get("augmentation_rng_state") is not None: rng.bit_generator.state=bank.metadata["augmentation_rng_state"]
    aggregate_before=sum(int(h["attempts"]) for h in bank.metadata.get("augmentation_history",[]))
    remaining=max(0,int(cfg.flight_augmentation_proposal_budget)-aggregate_before)
    budget=min(int(a.attempt_budget),remaining); rejected=Counter(); region_attempts=Counter(); region_accepts=Counter(); accepted_nn=[]
    env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank()); step=jax.jit(env.step); zero=jp.zeros(env.action_size,jp.float32)
    solver=TerrainClearanceSolver(cfg.xml_path,margin=cfg.flight_candidate_clearance_margin,max_correction=cfg.flight_candidate_max_root_z_correction,tolerance=cfg.flight_candidate_clearance_tolerance,max_iterations=cfg.flight_candidate_clearance_iterations)

    attempts=0
    while len(rows)<a.target and attempts<budget:
        attempts+=1; proposal_index=aggregate_before+attempts
        counts=Counter(row["flight_subinterval"] for row in rows)
        deficits={name:max(0,quotas[name]-counts[name]) for name in REGIONS}
        available=[name for name in REGIONS if deficits[name]>0]
        if not available: break
        weights=np.asarray([deficits[name] for name in available],np.float64); region=str(rng.choice(available,p=weights/weights.sum())); region_attempts[region]+=1
        eligible=[row for row in ordered[region] if children[row["id"]]<int(cfg.flight_augmentation_max_children_per_parent)]
        if not eligible: rejected["parent_cap_exhausted"]+=1; continue
        minimum=min(children[row["id"]] for row in eligible); balanced=[row for row in eligible if children[row["id"]]==minimum]
        parent=balanced[int(rng.integers(0,len(balanced)))]; position=next(i for i,row in enumerate(ordered[region]) if row["id"]==parent["id"])
        neighbors=[]
        if position: neighbors.append(ordered[region][position-1])
        if position+1<len(ordered[region]): neighbors.append(ordered[region][position+1])
        if not neighbors: rejected["no_adjacent_anchor"]+=1; continue
        neighbor=neighbors[int(rng.integers(0,len(neighbors)))]
        alpha=float(rng.uniform(cfg.flight_augmentation_interpolation_min,cfg.flight_augmentation_interpolation_max))
        qpos,qvel,ctrl=interpolate_state(parent,neighbor,alpha)
        factor=factors[region]; latent=rng.normal(size=factor.shape[1]); latent_norm=float(np.linalg.norm(latent))
        if latent_norm>3.0: latent*=3.0/latent_norm; latent_norm=3.0
        perturb=float(cfg.flight_augmentation_covariance_scale)*(factor@latent)
        qpos,qvel=apply_tangent(qpos,qvel,perturb)
        if not _joint_ranges_ok(solver.model,qpos): rejected["joint_range"]+=1; continue
        placement=solver.solve(qpos,qvel,ctrl,com_z_tolerance=cfg.flight_candidate_com_z_envelope_tolerance)
        if not placement.accepted: rejected[placement.reason]+=1; continue
        state=env.reset_from_snapshot(jp.asarray(placement.qpos,jp.float32),jp.asarray(qvel,jp.float32),jp.asarray(ctrl,jp.float32),jax.random.PRNGKey(a.seed+proposal_index),jp.asarray(STAGE_ID["flight"],jp.int32),jp.asarray(1,jp.int32),jp.asarray(0,jp.int32),jp.asarray(0,jp.int32))
        if not _finite(state): rejected["nonfinite"]+=1; continue
        if int(np.asarray(jax.device_get(state.info["phase"])))!=STAGE_ID["flight"]: rejected["non_flight_phase"]+=1; continue
        feature=np.asarray(jax.device_get(env._physical_feature(state.data)),np.float64)
        low=segment_features[region].min(axis=0)-float(cfg.flight_augmentation_envelope_sigma)*segment_features[region].std(axis=0)
        high=segment_features[region].max(axis=0)+float(cfg.flight_augmentation_envelope_sigma)*segment_features[region].std(axis=0)
        if np.any(feature<low-1e-7) or np.any(feature>high+1e-7): rejected["segment_envelope"]+=1; continue
        probe=state; failure=None
        for _ in range(int(cfg.flight_candidate_validation_steps)):
            probe=step(probe,zero)
            if not _finite(probe): failure="short_rollout_nonfinite"; break
            if float(np.asarray(jax.device_get(probe.done)))>.5:
                code=int(np.asarray(jax.device_get(probe.info["end_code"]))); failure="short_rollout_"+END_REASON.get(code,f"unknown_{code}"); break
        if failure: rejected[failure]+=1; continue
        nn=normalized_distance(feature,np.asarray(existing_features),scale)
        if nn<float(cfg.flight_augmentation_normalized_dedup_distance): rejected["normalized_duplicate"]+=1; continue
        record=env.snapshot_record(state,"flight")
        stable_id=hashlib.sha256(f"flight-augmentation:{a.seed}:{proposal_index}:{parent['id']}".encode()).hexdigest()[:32]
        log_density=float(-.5*latent_norm**2-.5*factor.shape[1]*np.log(2*np.pi))
        record.update({
            "id":stable_id,"training_only":False,"bootstrap_eligible":True,"candidate_kind":"local_augmented",
            "flight_subinterval":region,"source_index":int(parent["source_index"]),"reference_index":int(parent["source_index"]),
            "parent_anchor_id":parent["id"],"neighbor_anchor_id":neighbor["id"],"parent_source_index":int(parent["source_index"]),
            "interpolation_fraction":alpha,"perturbation_tangent":perturb.astype(np.float32),"sampling_latent_norm":latent_norm,
            "sampling_log_density":log_density,"sampling_density":float(np.exp(log_density)),"candidate_generation_seed":int(a.seed),
            "candidate_proposal_index":proposal_index,"original_system_com":placement.original_com.astype(np.float32),
            "corrected_system_com":placement.corrected_com.astype(np.float32),"root_z_shift_m":float(placement.root_z_shift),
            "terrain_clearance_m":float(placement.clearance),"wheel_clearance_m":float(placement.wheel_clearance),
            "nonwheel_clearance_m":float(placement.nonwheel_clearance),"normalized_nearest_neighbor_distance":nn,
        })
        bank.add(record,deduplicate=False); rows=bank.records_for_phase("flight",include_training_only=False)
        existing_features.append(feature); children[parent["id"]]+=1; region_accepts[region]+=1; accepted_nn.append(nn)

    history=list(bank.metadata.get("augmentation_history",[])); history.append({
        "attempts":attempts,"accepted":sum(region_accepts.values()),"region_attempts":dict(region_attempts),
        "region_accepted":dict(region_accepts),"rejection_counts":dict(rejected),
    })
    bank.metadata.update({
        "augmentation_rng_state":rng.bit_generator.state,"augmentation_history":history,
        "augmentation_reference_counts":reference_counts,"augmentation_region_quotas":quotas,
        "augmentation_feature_scale":scale.tolist(),"candidate_config_hash":config_hash(cfg),
    }); bank.save(output)
    rows=bank.records_for_phase("flight",include_training_only=False); augmented=[row for row in rows if row["candidate_kind"]=="local_augmented"]
    counts=Counter(row["flight_subinterval"] for row in rows); aggregate_rejections=Counter(); aggregate_attempts=0; aggregate_region_attempts=Counter()
    for item in history:
        aggregate_attempts+=int(item["attempts"]); aggregate_rejections.update(item["rejection_counts"]); aggregate_region_attempts.update(item["region_attempts"])
    all_nn=[float(row["normalized_nearest_neighbor_distance"]) for row in augmented]
    parent_counts=Counter(row["parent_anchor_id"] for row in augmented)
    coverage={name:counts[name]>=quotas[name] for name in REGIONS}; reached=len(rows)==a.target and all(coverage.values())
    report={
        "status":"PASS" if reached else "PARTIAL" if a.allow_partial and aggregate_attempts<int(cfg.flight_augmentation_proposal_budget) else "FAIL",
        "target":int(a.target),"anchor_count":len(anchor_rows),"augmented_count":len(augmented),"candidate_count":len(rows),
        "reference_region_counts":reference_counts,"region_quotas":quotas,"region_counts":dict(counts),"coverage_flags":coverage,
        "unique_parent_count":len(parent_counts),"maximum_children_per_parent":max(parent_counts.values(),default=0),
        "configured_parent_cap":int(cfg.flight_augmentation_max_children_per_parent),"normalized_nearest_neighbor_distance":_quantiles(all_nn),
        "normalized_dedup_threshold":float(cfg.flight_augmentation_normalized_dedup_distance),"aggregate_attempts":aggregate_attempts,
        "proposal_budget":int(cfg.flight_augmentation_proposal_budget),"region_attempts":dict(aggregate_region_attempts),
        "region_acceptance_rates":{name:float(sum(int(h["region_accepted"].get(name,0)) for h in history)/aggregate_region_attempts[name]) if aggregate_region_attempts[name] else 0.0 for name in REGIONS},
        "rejection_counts":dict(aggregate_rejections),"generation_seed":int(a.seed),"ppo_seed":0,
        "certification_seed":3100000,"independent_audit_seed":1000000,"anchor_bank_sha256":source_sha,
        "bank_sha256":file_sha256(output),"contract":{key:bank.metadata.get(key) for key in ("xml_sha256","action_mapping_version","actor_history_steps","candidate_config_hash")},
        "history":history,
    }
    output.with_suffix(".build.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({key:report[key] for key in report if key not in ("history",)},indent=2))
    if not reached and not a.allow_partial: raise SystemExit(2)
    if report["status"]=="FAIL": raise SystemExit(2)


if __name__=="__main__": main()
