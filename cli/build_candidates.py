"""Build a clean event-anchored candidate bank from the supplied trajectory envelopes."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import jax
import jax.numpy as jp
import numpy as np
from dvgc.bank import SnapshotBank
from dvgc.candidate_geometry import TerrainClearanceSolver
from dvgc.config import STAGE_ID, config_hash, file_sha256, load_config
from dvgc.env import END_REASON
from dvgc.env import OrangeBikeDVGC
from dvgc.reference import ReferenceTrajectory


def _seed(row, phase, cfg, training_only, rng, angular_velocity=None):
    euler=np.deg2rad([row.roll_angle,row.pitch_angle,row.yaw_angle]).astype(np.float32)
    common={"euler":euler,"steer":0.0,"hip":float(np.clip(row.hip_position,cfg.hip_min,cfg.hip_max)),"knee":float(np.clip(row.knee_position,cfg.knee_min,cfg.knee_max)),"linear_velocity":np.asarray([row.vel_x,row.vel_y,row.vel_z],np.float32),"angular_velocity":np.zeros(3,np.float32) if angular_velocity is None else np.asarray(angular_velocity,np.float32)}
    if phase in ("flight","landing"):
        common.update(seed_type="system_com",desired_com=np.asarray([row.pos_x,row.pos_y,row.pos_z],np.float32))
    else:
        vz=float(rng.uniform(0.35,0.85)) if (phase=="takeoff" and training_only) else float(np.clip(row.vel_z,-.08,.08))
        common["linear_velocity"]=np.asarray([row.vel_x,row.vel_y,vz],np.float32)
        common.update(seed_type="ground",base_pos=np.asarray([row.pos_x,row.pos_y,cfg.nominal_base_z_ground+0.03],np.float32))
    return common


def _landing_seed(row, cfg, rng):
    """Perturb a reference envelope into a low-height descending proposal."""
    seed=_seed(row,"landing",cfg,False,rng)
    seed["desired_com"][0]=np.clip(
        seed["desired_com"][0]+rng.uniform(-cfg.landing_candidate_x_jitter,cfg.landing_candidate_x_jitter),
        cfg.step_front_x+cfg.valid_landing_min_past_edge,
        cfg.step_back_x-cfg.valid_landing_back_margin,
    )
    seed["desired_com"][1]=np.clip(
        seed["desired_com"][1]+rng.uniform(-cfg.landing_candidate_y_jitter,cfg.landing_candidate_y_jitter),
        -cfg.step_half_width+cfg.landing_side_margin,
        cfg.step_half_width-cfg.landing_side_margin,
    )
    angle_jitter=np.deg2rad([
        cfg.landing_candidate_roll_jitter_deg,
        cfg.landing_candidate_pitch_jitter_deg,
        cfg.landing_candidate_yaw_jitter_deg,
    ])
    seed["euler"]=(seed["euler"]+rng.uniform(-angle_jitter,angle_jitter)).astype(np.float32)
    seed["linear_velocity"][0]+=rng.uniform(-cfg.landing_candidate_vx_jitter,cfg.landing_candidate_vx_jitter)
    seed["linear_velocity"][1]+=rng.uniform(-cfg.landing_candidate_vy_jitter,cfg.landing_candidate_vy_jitter)
    seed["linear_velocity"][2]=-rng.uniform(cfg.landing_candidate_descend_vz_min,cfg.landing_candidate_descend_vz_max)
    seed["hip"]=float(np.clip(seed["hip"]+rng.uniform(-cfg.landing_candidate_hip_jitter,cfg.landing_candidate_hip_jitter),cfg.hip_min,cfg.hip_max))
    seed["knee"]=float(np.clip(seed["knee"]+rng.uniform(-cfg.landing_candidate_knee_jitter,cfg.landing_candidate_knee_jitter),cfg.knee_min,cfg.knee_max))
    return seed


def _hip_hold_action(hip, cfg):
    if hip < cfg.hip_initial:
        return (hip-cfg.hip_initial)/(cfg.hip_initial-cfg.hip_min)
    return (hip-cfg.hip_initial)/(cfg.hip_max-cfg.hip_initial)


def _prepare_landing_impact(env, state, row, seed, cfg, rng, key):
    """Place wheels above the platform using XML geometry, not reference z."""
    clearance=float(rng.uniform(cfg.landing_candidate_clearance_min,cfg.landing_candidate_clearance_max))
    tire_bottom=min(
        float(np.asarray(jax.device_get(state.info["prev_front_tire_bottom_z"]))),
        float(np.asarray(jax.device_get(state.info["prev_rear_tire_bottom_z"]))),
    )
    vertical_correction=float(cfg.step_top_z+clearance-tire_bottom)
    qpos=state.data.qpos.at[env._qpos0+2].add(vertical_correction)
    qvel=state.data.qvel
    qvel=qvel.at[env._joint_qvel["hip_joint"]].set(float(np.clip(row.hip_velocity,-4.0,4.0)))
    qvel=qvel.at[env._joint_qvel["knee_joint"]].set(float(np.clip(row.knee_velocity,-4.0,4.0)))
    action=jp.asarray([
        float(np.clip(row.action_steering,-1.0,1.0)),
        float(np.clip(row.action_rearwheel,-1.0,1.0)),
        float(np.clip(_hip_hold_action(seed["hip"],cfg),-1.0,1.0)),
        0.0,
    ],jp.float32)
    ctrl=env._action_to_ctrl(action,qpos[env._joint_qpos["knee_joint"]])
    state=env.reset_from_snapshot(
        qpos,qvel,ctrl,key,
        jp.asarray(STAGE_ID["flight"],jp.int32),jp.asarray(1,jp.int32),
        jp.asarray(0,jp.int32),jp.asarray(0,jp.int32),last_action=action,
    )
    return state,action,{"preimpact_clearance_m":clearance,"vertical_correction_m":vertical_correction}


def _finite_state(state):
    values=(state.data.qpos,state.data.qvel,state.reward,state.obs["state"],state.obs["privileged_state"])
    return all(np.isfinite(np.asarray(jax.device_get(value))).all() for value in values)


def _flight_region(index, anchors, window):
    if index < anchors.apex-window: return "ascent"
    if index <= anchors.apex+window: return "apex"
    return "descent"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase",required=True,choices=["landing","flight","takeoff","approach"])
    p.add_argument("--target",type=int,default=128)
    p.add_argument("--bank",required=True)
    p.add_argument("--reference",default="data/reference_jump.csv")
    p.add_argument("--config",default="configs/default.json")
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--aux-fraction",type=float,default=.20)
    p.add_argument("--attempt-budget",type=int,default=0)
    p.add_argument("--allow-partial",action="store_true")
    p.add_argument("--dedup-distance",type=float,default=.06)
    a=p.parse_args(); rng=np.random.default_rng(a.seed)
    cfg=load_config(a.config,{"training_stage":a.phase,"use_bank_resets":False,"domain_randomization":False,"obs_noise_enable":False})
    env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank()); bank=SnapshotBank.load(a.bank)
    contract={
        "xml_sha256":file_sha256(cfg.xml_path),
        "action_mapping_version":str(cfg.action_mapping_version),
        "actor_history_steps":int(cfg.actor_history_steps),
        "candidate_config_hash":config_hash(cfg),
    }
    if bank.records:
        for key,value in contract.items():
            if bank.metadata.get(key)!=value:
                raise SystemExit(f"Existing candidate bank has incompatible {key}")
    reference=ReferenceTrajectory.load(a.reference); df=reference.df; anchors=reference.anchors()
    angle_values=np.unwrap(np.deg2rad(df[["roll_angle","pitch_angle","yaw_angle"]].to_numpy(float)),axis=0)
    angular_velocities=np.gradient(angle_values,df["time"].to_numpy(float),axis=0)
    bounds={
        "approach":(0,anchors.approach_end),
        "takeoff":(anchors.approach_end,anchors.takeoff_end),
        "flight":(anchors.takeoff_end,anchors.landing_start),
        "landing":(anchors.landing_start,anchors.recovery_start),
    }
    lo,hi=bounds[a.phase]
    attempts=0; accepted_before=len(bank.records_for_phase(a.phase)); duplicates=0
    semantic_rejects=0; relaxation_rejects=0; nonfinite_rejects=0
    physical_failures=0; timeouts=0; end_reasons={}; impact_steps=[]; vertical_corrections=[]; clearances=[]
    rejection_counts={}; region_proposals={name:0 for name in ("ascent","apex","descent")}
    region_accepted={name:0 for name in region_proposals}
    clearance_solver=None
    if a.phase=="flight":
        clearance_solver=TerrainClearanceSolver(
            cfg.xml_path,margin=cfg.flight_candidate_clearance_margin,
            max_correction=cfg.flight_candidate_max_root_z_correction,
            tolerance=cfg.flight_candidate_clearance_tolerance,
            max_iterations=cfg.flight_candidate_clearance_iterations,
        )
    step=jax.jit(env.step); zero=jp.zeros(env.action_size,jp.float32)
    attempt_budget=int(a.attempt_budget) if a.attempt_budget else a.target*30
    if attempt_budget<=0: raise SystemExit("--attempt-budget must be positive")
    if float(a.dedup_distance)<=0: raise SystemExit("--dedup-distance must be positive")
    while len(bank.records_for_phase(a.phase))<a.target and attempts<attempt_budget:
        attempts+=1; idx=int(rng.integers(lo,hi+1)); row=df.iloc[idx]
        training_only=(a.phase=="takeoff" and rng.random()<a.aux_fraction)
        region=_flight_region(idx,anchors,int(cfg.flight_candidate_apex_window_steps)) if a.phase=="flight" else None
        if region: region_proposals[region]+=1
        seed=_landing_seed(row,cfg,rng) if a.phase=="landing" else _seed(row,a.phase,cfg,training_only,rng,angular_velocities[idx])
        state=env.reset_from_com_seed(seed,jax.random.PRNGKey(a.seed+attempts))
        if a.phase=="flight":
            # The reference joint rates and all rigid-body velocity/orientation
            # components are retained; placement may alter free-root z only.
            qvel=state.data.qvel.at[env._joint_qvel["hip_joint"]].set(float(row.hip_velocity))
            qvel=qvel.at[env._joint_qvel["knee_joint"]].set(float(row.knee_velocity))
            qpos0=np.asarray(jax.device_get(state.data.qpos),np.float64)
            qvel0=np.asarray(jax.device_get(qvel),np.float64)
            ctrl0=np.asarray(jax.device_get(state.data.ctrl),np.float64)
            placement=clearance_solver.solve(
                qpos0,qvel0,ctrl0,
                com_z_tolerance=cfg.flight_candidate_com_z_envelope_tolerance,
            )
            if not placement.accepted:
                rejection_counts[placement.reason]=rejection_counts.get(placement.reason,0)+1
                continue
            changed=np.flatnonzero(np.abs(placement.qpos-qpos0)>2e-7)
            if any(int(i)!=clearance_solver.root_z_index for i in changed):
                raise RuntimeError("Flight clearance changed a non-root-z coordinate")
            state=env.reset_from_snapshot(
                jp.asarray(placement.qpos,jp.float32),qvel,state.data.ctrl,
                jax.random.PRNGKey(a.seed+200000+attempts),
                jp.asarray(STAGE_ID["flight"],jp.int32),jp.asarray(1,jp.int32),
                jp.asarray(0,jp.int32),jp.asarray(0,jp.int32),
            )
            if not _finite_state(state):
                nonfinite_rejects+=1; rejection_counts["nonfinite"] = rejection_counts.get("nonfinite",0)+1; continue
            if int(np.asarray(jax.device_get(state.info["phase"])))!=STAGE_ID["flight"]:
                semantic_rejects+=1; rejection_counts["non_flight_phase"] = rejection_counts.get("non_flight_phase",0)+1; continue
            probe=state; failed=False
            for _ in range(int(cfg.flight_candidate_validation_steps)):
                probe=step(probe,zero)
                if not _finite_state(probe):
                    nonfinite_rejects+=1; rejection_counts["short_rollout_nonfinite"] = rejection_counts.get("short_rollout_nonfinite",0)+1; failed=True; break
                if float(np.asarray(jax.device_get(probe.done)))>.5:
                    terminated=bool(np.asarray(jax.device_get(probe.info["terminated"])))
                    truncated=bool(np.asarray(jax.device_get(probe.info["truncated"])))
                    physical_failures+=int(terminated and not truncated); timeouts+=int(truncated)
                    code=int(np.asarray(jax.device_get(probe.info["end_code"])))
                    reason=END_REASON.get(code,f"unknown_{code}"); end_reasons[reason]=end_reasons.get(reason,0)+1
                    key="short_rollout_"+reason; rejection_counts[key]=rejection_counts.get(key,0)+1; failed=True; break
            if failed: continue
        # Convert the grounded proposal to the requested semantic phase.
        if a.phase=="approach":
            state=env.reset_from_snapshot(state.data.qpos,state.data.qvel,state.data.ctrl,jax.random.PRNGKey(a.seed+100000+attempts),jp.asarray(STAGE_ID["approach"],jp.int32),jp.asarray(0,jp.int32),jp.asarray(0,jp.int32),jp.asarray(0,jp.int32))
        if a.phase=="landing":
            state,impact_action,proposal=_prepare_landing_impact(
                env,state,row,seed,cfg,rng,jax.random.PRNGKey(a.seed+100000+attempts)
            )
            impact_step=0
            for impact_step in range(1,int(cfg.landing_candidate_impact_horizon)+1):
                state=step(state,impact_action)
                if int(np.asarray(jax.device_get(state.info["phase"])))==STAGE_ID["landing"]:
                    break
                if float(np.asarray(jax.device_get(state.done)))>.5: break
            phase=int(np.asarray(jax.device_get(state.info["phase"])))
            done=float(np.asarray(jax.device_get(state.done)))>.5
            if phase!=STAGE_ID["landing"] or done:
                semantic_rejects+=1
                if done:
                    terminated=bool(np.asarray(jax.device_get(state.info["terminated"])))
                    truncated=bool(np.asarray(jax.device_get(state.info["truncated"])))
                    physical_failures+=int(terminated and not truncated)
                    timeouts+=int(truncated)
                    code=int(np.asarray(jax.device_get(state.info["end_code"])))
                    reason=END_REASON.get(code,f"unknown_{code}")
                    end_reasons[reason]=end_reasons.get(reason,0)+1
                continue
            probe=state
            relaxation_failed=False
            for _ in range(int(cfg.landing_candidate_relaxation_steps)):
                probe=step(probe,zero)
                if not _finite_state(probe):
                    nonfinite_rejects+=1; relaxation_failed=True; break
                if float(np.asarray(jax.device_get(probe.done)))>.5:
                    relaxation_rejects+=1; relaxation_failed=True
                    terminated=bool(np.asarray(jax.device_get(probe.info["terminated"])))
                    truncated=bool(np.asarray(jax.device_get(probe.info["truncated"])))
                    physical_failures+=int(terminated and not truncated); timeouts+=int(truncated)
                    code=int(np.asarray(jax.device_get(probe.info["end_code"])))
                    reason=END_REASON.get(code,f"unknown_{code}")
                    end_reasons[reason]=end_reasons.get(reason,0)+1
                    break
            if relaxation_failed:
                continue
        rec=env.snapshot_record(state,a.phase)
        rec.update({"training_only":training_only,"bootstrap_eligible":True,"candidate_kind":"reference_envelope_impact" if a.phase=="landing" else "velocity_seed" if training_only else "reference_envelope","reference_index":idx})
        if a.phase=="flight":
            rec.update({
                "flight_subinterval":region,
                "source_index":idx,
                "original_system_com":placement.original_com.astype(np.float32),
                "corrected_system_com":placement.corrected_com.astype(np.float32),
                "root_z_shift_m":float(placement.root_z_shift),
                "terrain_clearance_m":float(placement.clearance),
                "wheel_clearance_m":float(placement.wheel_clearance),
                "nonwheel_clearance_m":float(placement.nonwheel_clearance),
            })
        if a.phase=="landing":
            rec.update(proposal); rec["impact_step"]=impact_step
        # Certification candidates must respect phase semantics.
        if a.phase=="takeoff":
            rec["had_airborne"]=0; rec["airborne_count"]=0; rec["policy_state"]["filter_phase"]=STAGE_ID["takeoff"]
        if not bank.add(rec,deduplicate=True,distance=float(a.dedup_distance)):
            duplicates+=1
        elif a.phase=="landing":
            impact_steps.append(impact_step); vertical_corrections.append(proposal["vertical_correction_m"]); clearances.append(proposal["preimpact_clearance_m"])
        elif a.phase=="flight":
            vertical_corrections.append(float(placement.root_z_shift)); clearances.append(float(placement.clearance)); region_accepted[region]+=1
        if attempts%25==0 or len(bank.records_for_phase(a.phase))==a.target:
            print(f"[candidates] phase={a.phase} accepted={len(bank.records_for_phase(a.phase))}/{a.target} attempts={attempts} duplicates={duplicates} semantic_rejects={semantic_rejects}",flush=True)
    history=list(bank.metadata.get("candidate_build_history",[]))
    history.append({
        "seed":int(a.seed),"attempt_budget":attempt_budget,"dedup_distance":float(a.dedup_distance),"attempts":attempts,
        "accepted_before":accepted_before,"accepted_after":len(bank.records_for_phase(a.phase)),
        "accepted_new":len(bank.records_for_phase(a.phase))-accepted_before,
        "duplicates":duplicates,"semantic_rejects":semantic_rejects,
        "relaxation_rejects":relaxation_rejects,"nonfinite_rejects":nonfinite_rejects,
        "proposal_physical_failures":physical_failures,"proposal_timeouts":timeouts,
        "rejection_counts":rejection_counts,"flight_region_proposals":region_proposals,
        "flight_region_accepted":region_accepted,
    })
    bank.metadata.update({
        "reference":str(Path(a.reference)),
        "reference_usage":"candidate envelopes only; never reward tracking",
        "reference_anchors":anchors.as_dict(),
        "build_seed":int(bank.metadata.get("build_seed",a.seed)),
        "build_seeds":[int(row["seed"]) for row in history],
        "candidate_dedup_distance":float(a.dedup_distance),
        "candidate_build_history":history,
        "candidate_generation":{
            "landing":"reference envelope perturbation -> XML wheel-clearance placement -> real IMU impact snapshot -> zero-action relaxation",
            "landing_clearance_m":[float(cfg.landing_candidate_clearance_min),float(cfg.landing_candidate_clearance_max)],
            "landing_descend_vz_mps":[-float(cfg.landing_candidate_descend_vz_max),-float(cfg.landing_candidate_descend_vz_min)],
            "landing_impact_horizon":int(cfg.landing_candidate_impact_horizon),
            "landing_relaxation_steps":int(cfg.landing_candidate_relaxation_steps),
            "flight":"reference state -> authoritative XML geom-distance/contact minimal root-z clearance -> unchanged-state short rollout",
            "flight_clearance_margin_m":float(cfg.flight_candidate_clearance_margin),
            "flight_max_root_z_correction_m":float(cfg.flight_candidate_max_root_z_correction),
            "flight_com_z_envelope_tolerance_m":float(cfg.flight_candidate_com_z_envelope_tolerance),
            "flight_validation_steps":int(cfg.flight_candidate_validation_steps),
        },
        **contract,
    })
    bank.save(a.bank)
    accepted_new=len(bank.records_for_phase(a.phase))-accepted_before
    valid_proposals=accepted_new+duplicates
    reached=len(bank.records_for_phase(a.phase))>=a.target
    range_or_none=lambda values: None if not values else {"min":float(min(values)),"max":float(max(values)),"mean":float(np.mean(values))}
    aggregate_attempts=sum(int(row["attempts"]) for row in history); aggregate_duplicates=sum(int(row["duplicates"]) for row in history); aggregate_accepted=sum(int(row["accepted_new"]) for row in history)
    flight_rows=bank.records_for_phase("flight") if a.phase=="flight" else []
    accepted_regions={name:sum(row.get("flight_subinterval")==name for row in flight_rows) for name in region_proposals}
    aggregate_region_proposals={name:sum(int(h.get("flight_region_proposals",{}).get(name,0)) for h in history) for name in region_proposals}
    region_acceptance={name:{"accepted":accepted_regions[name],"proposals":aggregate_region_proposals[name],"acceptance_rate":float(accepted_regions[name]/aggregate_region_proposals[name]) if aggregate_region_proposals[name] else 0.0} for name in region_proposals}
    coverage_flags={}
    if a.phase=="flight":
        coverage_flags={
            "ascent":accepted_regions["ascent"]>=int(np.ceil(a.target*cfg.flight_candidate_min_ascent_fraction)),
            "apex":accepted_regions["apex"]>=int(np.ceil(a.target*cfg.flight_candidate_min_apex_fraction)),
            "descent":accepted_regions["descent"]>=int(np.ceil(a.target*cfg.flight_candidate_min_descent_fraction)),
        }
    coverage_ok=all(coverage_flags.values()) if coverage_flags else True
    status="PASS" if reached and coverage_ok else "PARTIAL" if a.allow_partial and not reached else "FAIL"
    corrections=[float(row["root_z_shift_m"]) for row in flight_rows]
    correction_quantiles=None if not corrections else {"min":float(np.min(corrections)),"p50":float(np.quantile(corrections,.50)),"p95":float(np.quantile(corrections,.95)),"max":float(np.max(corrections))}
    aggregate_rejections={}
    for h in history:
        for key,value in h.get("rejection_counts",{}).items(): aggregate_rejections[key]=aggregate_rejections.get(key,0)+int(value)
    report={"status":status,"phase":a.phase,"target":a.target,"attempts":attempts,"aggregate_attempts":aggregate_attempts,"accepted_before":accepted_before,"accepted_new":accepted_new,"aggregate_accepted_new":aggregate_accepted,"duplicates":duplicates,"aggregate_duplicates":aggregate_duplicates,"deduplication_rate":float(aggregate_duplicates/(aggregate_accepted+aggregate_duplicates)) if aggregate_accepted+aggregate_duplicates else 0.0,"semantic_rejects":semantic_rejects,"relaxation_rejects":relaxation_rejects,"nonfinite_rejects":nonfinite_rejects,"proposal_physical_failures":physical_failures,"proposal_timeouts":timeouts,"proposal_physical_failure_rate":float(physical_failures/attempts) if attempts else 0.0,"proposal_timeout_rate":float(timeouts/attempts) if attempts else 0.0,"proposal_end_reasons":end_reasons,"rejection_counts":aggregate_rejections,"accepted_impact_step_range":range_or_none(impact_steps),"accepted_vertical_correction_range_m":range_or_none(vertical_corrections),"accepted_preimpact_clearance_range_m":range_or_none(clearances),"flight_correction_quantiles_m":correction_quantiles,"flight_subintervals":region_acceptance,"flight_coverage_flags":coverage_flags,"contract":contract,"build_history":history,"summary":bank.summary()}
    Path(a.bank).with_suffix(".build.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    if (not reached or not coverage_ok) and not (a.allow_partial and not reached): raise SystemExit(2)
if __name__=="__main__": main()
