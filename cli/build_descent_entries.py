"""Build early-descent handoff proposals from successful frozen composite rollouts."""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path

import jax
import numpy as np

from dvgc.bank import SnapshotBank
from dvgc.candidate_geometry import TerrainClearanceSolver
from dvgc.certification import DYNAMICS_VARIANTS
from dvgc.composite import CanonicalEntryMatcher, CompositeSession
from dvgc.config import file_sha256, load_config
from dvgc.descent_entry import DESCENT_ENTRY_FEATURE_NAMES, descent_entry_feature
from dvgc.entry import robust_normalization
from dvgc.env import OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.rollout import restore_snapshot
from dvgc.runtime import build_inference, save_json


def snapshot_identity(row):
    digest=hashlib.sha256()
    for key in ("qpos","qvel","ctrl","qacc_warmstart"):
        value=np.ascontiguousarray(row[key]); digest.update(key.encode()); digest.update(value.dtype.str.encode()); digest.update(value.shape.__repr__().encode()); digest.update(value.tobytes())
    policy=row.get("policy_state",{})
    for key in sorted(policy):
        value=np.ascontiguousarray(policy[key]); digest.update(key.encode()); digest.update(value.dtype.str.encode()); digest.update(value.shape.__repr__().encode()); digest.update(value.tobytes())
    for key in ("oracle_phase","had_airborne","had_valid_landing","contact_age","airborne_count","recovery_count"):
        digest.update(f"{key}:{row.get(key)}".encode())
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--descent-policy",required=True); p.add_argument("--landing-policy",required=True)
    p.add_argument("--flight-bank",required=True); p.add_argument("--landing-entry-set",required=True); p.add_argument("--output-bank",required=True)
    p.add_argument("--config",default="configs/default.json"); p.add_argument("--seed",type=int,default=7000000); p.add_argument("--stride",type=int,default=1)
    p.add_argument("--rollouts-per-candidate",type=int,default=len(DYNAMICS_VARIANTS)); p.add_argument("--action-noise-std",type=float,default=0.0); a=p.parse_args(); out=Path(a.output_bank)
    if out.exists(): raise SystemExit(f"Output exists: {out}")
    dp,dc,dm=load_bundle(a.descent_policy,verify_files=True); lp,_,lm=load_bundle(a.landing_policy,verify_files=True)
    if file_sha256(a.flight_bank)!=dm["candidate_bank_sha256"] or file_sha256(a.landing_entry_set)!=dm["downstream_bank_sha256"]: raise SystemExit("Frozen descent provenance mismatch")
    cfg=load_config(a.config,{**dc,"training_stage":"flight","expert_chain_termination":False,"domain_randomization":False,"obs_noise_enable":False,"use_bank_resets":False})
    if a.rollouts_per_candidate < 1: raise SystemExit("--rollouts-per-candidate must be positive")
    variants=[]
    for spec in DYNAMICS_VARIANTS:
        variant_cfg=load_config(a.config,{**cfg.to_dict(),**{k:v for k,v in spec.items() if k!="id"}})
        env=OrangeBikeDVGC(variant_cfg,snapshot_bank=SnapshotBank(),cert_bank=SnapshotBank.load(a.landing_entry_set))
        variants.append((spec["id"],env,jax.jit(env.step),{"flight":build_inference(env,dp,deterministic=True),"landing":build_inference(env,lp,deterministic=True)},CanonicalEntryMatcher(env,"flight",a.landing_entry_set)))
    source=SnapshotBank.load(a.flight_bank); rows=[r for r in source.records_for_phase("flight",include_training_only=False) if r.get("flight_subinterval")=="descent"]
    # C_D snapshots are real pre-contact rollout states, not newly placed Flight
    # candidates.  Preserve them exactly and require the authoritative model to
    # report zero contact and non-negative geom distance; do not impose the
    # separate 2 mm candidate-placement margin or correct root z.
    solver=TerrainClearanceSolver(cfg.xml_path,margin=0.0,max_correction=float(cfg.flight_candidate_max_root_z_correction))
    proposals=[]; rollout_summary=[]; rejected={"contact_or_penetration":0,"nonfinite":0}
    for i,row in enumerate(rows):
        for rollout_index in range(a.rollouts_per_candidate):
            variant_id,env,step,inference,matcher=variants[rollout_index%len(variants)]
            seed=a.seed+i*a.rollouts_per_candidate+rollout_index; key=jax.random.PRNGKey(seed); state=restore_snapshot(env,row,key); session=CompositeSession(env,("flight","landing"),inference,{"flight":matcher},state,key); captured=[]
            initial=env.snapshot_record(state,"flight"); initial["proposal_step"]=0; captured.append(initial)
            for t in range(int(cfg.branch_horizon)):
                before=session.active_stage; session.step(step_fn=step,action_noise_std=a.action_noise_std)
                if before=="flight" and session.active_stage=="flight" and (t+1)%max(1,a.stride)==0:
                    phase=int(np.asarray(jax.device_get(session.state.info["phase"]))); landed=int(np.asarray(jax.device_get(session.state.info["had_valid_landing"])))
                    if phase==2 and not landed:
                        snap=env.snapshot_record(session.state,"flight"); snap["proposal_step"]=t+1; captured.append(snap)
                if bool(np.asarray(jax.device_get(session.state.done))): break
            final=bool(np.asarray(jax.device_get(session.state.info["recovery_success"]))); chain=bool(session.handoffs)
            rollout_summary.append({"candidate_id":row["id"],"rollout_index":rollout_index,"dynamics_variant":variant_id,"seed":seed,"chain":chain,"final":final,"captured":len(captured),"steps":int(np.asarray(jax.device_get(session.state.info["episode_step"])))})
            if not (chain and final): continue
            for snap in captured:
                placement=solver.solve(snap["qpos"],snap["qvel"],snap["ctrl"])
                if not placement.accepted or placement.root_z_shift>1e-7 or placement.robot_terrain_contacts:
                    rejected["nonfinite" if placement.reason=="nonfinite" else "contact_or_penetration"]+=1; continue
                snap.update({"id":hashlib.sha256(f"descent-entry:{a.seed}:{row['id']}:{rollout_index}:{snap['proposal_step']}".encode()).hexdigest()[:32],"candidate_kind":"descent_entry_proposal","entry_feature":descent_entry_feature(snap["physical_feature"],cfg).astype(np.float32),"entry_source_id":row["id"],"entry_source_reference_index":row.get("reference_index"),"entry_source_policy":dm["policy_version"],"entry_source_rollout_index":rollout_index,"entry_source_dynamics_variant":variant_id,"entry_source_rollout_seed":seed,"entry_construction_seed":a.seed,"flight_subinterval":"descent","bootstrap_eligible":True,"terrain_clearance_m":placement.clearance,"wheel_clearance_m":placement.wheel_clearance,"nonwheel_clearance_m":placement.nonwheel_clearance,"root_z_shift_m":0.0})
                proposals.append(snap)
    if not proposals: raise SystemExit("No successful composite descent proposals")
    center,scale=robust_normalization([r["entry_feature"] for r in proposals],cfg.descent_entry_scale_floors)
    # Reliability is established only by branch certification.  Retain nearby
    # temporal proposals, but never count byte-identical full snapshots twice.
    accepted=[]; seen=set(); duplicates=0
    for row in sorted(proposals,key=lambda r:(r["proposal_step"],r["entry_source_id"],r["entry_source_rollout_index"])):
        identity=snapshot_identity(row)
        if identity in seen: duplicates+=1; continue
        seen.add(identity); row["snapshot_identity_sha256"]=identity; accepted.append(row)
    metadata={"entry_bank_role":"descent_handoff_proposals","feature_names":DESCENT_ENTRY_FEATURE_NAMES,"construction_seed":a.seed,"stride":a.stride,"rollouts_per_candidate":a.rollouts_per_candidate,"construction_action_noise_std":a.action_noise_std,"construction_dynamics_variants":[dict(x) for x in DYNAMICS_VARIANTS],"descent_policy_version":dm["policy_version"],"descent_policy_hash":file_sha256(Path(a.descent_policy)/"params.pkl"),"landing_policy_version":lm["policy_version"],"landing_policy_hash":file_sha256(Path(a.landing_policy)/"params.pkl"),"landing_entry_set":str(Path(a.landing_entry_set).resolve()),"landing_entry_set_sha256":file_sha256(a.landing_entry_set),"flight_bank_sha256":file_sha256(a.flight_bank),"xml_sha256":file_sha256(cfg.xml_path),"action_mapping_version":cfg.action_mapping_version,"actor_history_steps":cfg.actor_history_steps,"dedup_distance":float(cfg.flight_augmentation_normalized_dedup_distance),"proposal_center":center.tolist(),"proposal_scale":scale.tolist()}
    SnapshotBank(accepted,metadata).save(out); report={"status":"PASS","descent_states":len(rows),"successful_composite_rollouts":sum(r["final"] and r["chain"] for r in rollout_summary),"proposals_before_dedup":len(proposals),"proposals":len(accepted),"duplicates":duplicates,"rejections":rejected,"earliest_step":min(r["proposal_step"] for r in accepted),"latest_step":max(r["proposal_step"] for r in accepted),"bank_sha256":file_sha256(out),"rollouts":rollout_summary}
    save_json(out.with_suffix(".build.json"),report); print(json.dumps({k:v for k,v in report.items() if k!="rollouts"},indent=2))


if __name__=="__main__": main()
