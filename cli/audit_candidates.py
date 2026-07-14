"""Audit candidate geometry, semantics, and short-horizon physical failures before PPO."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from dvgc.bank import SnapshotBank
from dvgc.config import ID_STAGE, STAGE_ID, config_hash, file_sha256, load_config
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.rollout import restore_snapshot


FEATURE_NAMES = (
    "x", "y", "z", "roll_rad", "pitch_rad", "yaw_rad",
    "vx", "vy", "vz", "wx", "wy", "wz",
    "steer", "hip", "knee", "rearwheel_velocity",
)


def _ranges(rows):
    features=np.asarray([row["physical_feature"] for row in rows],np.float64)
    return {
        name:{"min":float(features[:,i].min()),"max":float(features[:,i].max()),"mean":float(features[:,i].mean())}
        for i,name in enumerate(FEATURE_NAMES)
    }


def _contact_audit(rows, xml_path, deep_penetration_threshold):
    model=mujoco.MjModel.from_xml_path(str(xml_path)); data=mujoco.MjData(model)
    enabled=lambda g: bool(int(model.geom_contype[g]) or int(model.geom_conaffinity[g]))
    terrain={g for g in range(model.ngeom) if int(model.geom_bodyid[g])==0 and enabled(g)}
    wheel_bodies={model.body(name).id for name in ("frontwheel","rearwheel")}
    wheel={g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in wheel_bodies and enabled(g)}
    robot={g for g in range(model.ngeom) if int(model.geom_bodyid[g])!=0 and enabled(g)}
    body=robot-wheel
    contact_records=deep_records=body_records=0; min_dist=float("inf"); pairs=Counter()
    for row in rows:
        data.qpos[:]=row["qpos"]; data.qvel[:]=row["qvel"]; data.ctrl[:]=row["ctrl"]
        mujoco.mj_forward(model,data)
        has_contact=has_deep=has_body=False
        for i in range(data.ncon):
            con=data.contact[i]; g1,g2=int(con.geom1),int(con.geom2); distance=float(con.dist)
            is_robot_terrain=((g1 in robot and g2 in terrain) or (g2 in robot and g1 in terrain))
            if not is_robot_terrain: continue
            has_contact=True; has_deep|=distance<float(deep_penetration_threshold)
            has_body|=((g1 in body and g2 in terrain) or (g2 in body and g1 in terrain))
            min_dist=min(min_dist,distance)
            n1=model.geom(g1).name or str(g1); n2=model.geom(g2).name or str(g2)
            pairs["|".join(sorted((n1,n2)))]+=1
        contact_records+=int(has_contact); deep_records+=int(has_deep); body_records+=int(has_body)
    return {
        "records_with_any_contact":contact_records,
        "records_with_robot_terrain_contact":contact_records,
        "records_with_deep_penetration":deep_records,
        "records_with_body_terrain_contact":body_records,
        "deep_penetration_threshold_m":float(deep_penetration_threshold),
        "minimum_contact_distance_m":None if min_dist==float("inf") else min_dist,
        "contact_pair_counts":dict(sorted(pairs.items())),
        "allowed_wheel_geom_ids":sorted(wheel),
    }


def _rollout_audit(rows, cfg, horizon):
    env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank()); step=jax.jit(env.step)
    zero=jp.zeros(env.action_size,jp.float32); causes=Counter(); reasons=Counter(); phases=Counter()
    one_step_failures=0; nonfinite=0; physical_failure_records=[]
    for i,row in enumerate(rows):
        state=restore_snapshot(env,row,jax.random.PRNGKey(500000+i))
        for t in range(int(horizon)):
            state=step(state,zero)
            phase=int(np.asarray(jax.device_get(state.info["phase"])))
            phases[ID_STAGE.get(phase,str(phase))]+=1
            finite=all(np.isfinite(np.asarray(jax.device_get(value))).all() for value in (state.data.qpos,state.data.qvel,state.reward,state.obs["state"],state.obs["privileged_state"]))
            if not finite:
                nonfinite+=1; break
            if float(np.asarray(jax.device_get(state.done)))>.5:
                final=bool(np.asarray(jax.device_get(state.info["recovery_success"])))
                terminated=bool(np.asarray(jax.device_get(state.info["terminated"])))
                truncated=bool(np.asarray(jax.device_get(state.info["truncated"])))
                cause="final_recovery" if final else "physical_failure" if terminated else "timeout" if truncated else "invalid_done"
                causes[cause]+=1; one_step_failures+=int(t==0 and cause=="physical_failure")
                code=int(np.asarray(jax.device_get(state.info["end_code"])))
                reason=END_REASON.get(code,f"unknown_{code}"); reasons[reason]+=1
                if cause=="physical_failure":
                    physical_failure_records.append({
                        "bank_index":i,"record_id":row.get("id"),
                        "reference_index":row.get("reference_index"),
                        "terminal_step":t+1,"reason":reason,
                    })
                break
        else:
            causes["active_at_horizon"]+=1
    total=len(rows)
    return {
        "horizon_steps":int(horizon),
        "terminal_counts":dict(causes),
        "termination_reasons":dict(reasons),
        "one_step_physical_failure_rate":float(one_step_failures/total) if total else 0.0,
        "short_horizon_physical_failure_rate":float(causes["physical_failure"]/total) if total else 0.0,
        "short_horizon_timeout_rate":float(causes["timeout"]/total) if total else 0.0,
        "nonfinite_records":nonfinite,
        "phase_visitation_steps":dict(phases),
        "physical_failure_records":physical_failure_records,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank",required=True); parser.add_argument("--phase",required=True,choices=list(STAGE_ID))
    parser.add_argument("--config",default="configs/default.json"); parser.add_argument("--output",required=True)
    parser.add_argument("--expected-count",type=int,default=0); parser.add_argument("--horizon",type=int,default=25)
    parser.add_argument("--deep-penetration-threshold",type=float,default=-0.005)
    args=parser.parse_args(); bank=SnapshotBank.load(args.bank)
    rows=bank.records_for_phase(args.phase,include_training_only=True)
    cfg=load_config(args.config,{"training_stage":args.phase,"use_bank_resets":False,"domain_randomization":False,"obs_noise_enable":False})
    oracle=Counter(ID_STAGE.get(int(row.get("oracle_phase",-1)),str(row.get("oracle_phase"))) for row in rows)
    filtered=Counter(ID_STAGE.get(int(row.get("policy_state",{}).get("filter_phase",-1)),str(row.get("policy_state",{}).get("filter_phase"))) for row in rows)
    finite_records=sum(all(np.isfinite(np.asarray(row[key])).all() for key in ("qpos","qvel","ctrl","qacc_warmstart","physical_feature")) for row in rows)
    contact=_contact_audit(rows,cfg.xml_path,args.deep_penetration_threshold)
    rollout=_rollout_audit(rows,cfg,args.horizon)
    eligible=sum(bool(row.get("bootstrap_eligible",False)) for row in rows)
    training_only=sum(bool(row.get("training_only",False)) for row in rows)
    expected_ok=(not args.expected_count or len(rows)==args.expected_count)
    contract_flags={
        "xml_sha256":bank.metadata.get("xml_sha256")==file_sha256(cfg.xml_path),
        "action_mapping_version":bank.metadata.get("action_mapping_version")==str(cfg.action_mapping_version),
        "actor_history_steps":bank.metadata.get("actor_history_steps")==int(cfg.actor_history_steps),
        "candidate_config_hash":bank.metadata.get("candidate_config_hash")==config_hash(cfg),
    }
    quality_flags={
        "expected_count":expected_ok,
        "all_finite":finite_records==len(rows),
        "all_oracle_phase":oracle==Counter({args.phase:len(rows)}),
        "all_filter_phase":filtered==Counter({args.phase:len(rows)}),
        "all_bootstrap_eligible":eligible==len(rows),
        "no_training_only":training_only==0,
        "no_deep_penetration":contact["records_with_deep_penetration"]==0,
        "no_robot_terrain_contact":contact["records_with_robot_terrain_contact"]==0,
        "no_body_terrain_contact":contact["records_with_body_terrain_contact"]==0,
        "no_one_step_physical_failure":rollout["one_step_physical_failure_rate"]==0.0,
        "short_horizon_physical_failure_within_limit":rollout["short_horizon_physical_failure_rate"]<=float(cfg.landing_candidate_max_short_horizon_failure_rate),
        "no_nonfinite_rollout":rollout["nonfinite_records"]==0,
        "contract_consistent":all(contract_flags.values()),
    }
    if args.phase=="flight":
        regions=Counter(row.get("flight_subinterval","missing") for row in rows)
        quality_flags.update({
            "flight_ascent_coverage":regions["ascent"]>=int(np.ceil(len(rows)*cfg.flight_candidate_min_ascent_fraction)),
            "flight_apex_coverage":regions["apex"]>=int(np.ceil(len(rows)*cfg.flight_candidate_min_apex_fraction)),
            "flight_descent_coverage":regions["descent"]>=int(np.ceil(len(rows)*cfg.flight_candidate_min_descent_fraction)),
            "flight_clearance_metadata":all(
                all(key in row for key in ("source_index","original_system_com","corrected_system_com","root_z_shift_m","terrain_clearance_m"))
                for row in rows
            ),
        })
    report={
        "status":"PASS" if all(quality_flags.values()) else "FAIL",
        "bank":str(Path(args.bank).resolve()),"bank_sha256":file_sha256(args.bank),"phase":args.phase,
        "candidate_count":len(rows),"finite_records":finite_records,"bootstrap_eligible":eligible,
        "training_only":training_only,"oracle_phase_counts":dict(oracle),"filter_phase_counts":dict(filtered),
        "feature_ranges":_ranges(rows) if rows else {},"contact_audit":contact,"rollout_audit":rollout,
        "flight_subinterval_counts":dict(Counter(row.get("flight_subinterval","missing") for row in rows)) if args.phase=="flight" else {},
        "quality_flags":quality_flags,"contract_flags":contract_flags,
        "quality_thresholds":{"max_short_horizon_physical_failure_rate":float(cfg.landing_candidate_max_short_horizon_failure_rate)},
        "bank_metadata":bank.metadata,
    }
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps({key:report[key] for key in ("status","candidate_count","bootstrap_eligible","training_only","oracle_phase_counts","filter_phase_counts","contact_audit","rollout_audit","contract_flags","quality_flags")},indent=2))
    if report["status"]!="PASS": raise SystemExit(2)


if __name__=="__main__": main()
