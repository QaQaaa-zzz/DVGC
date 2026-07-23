"""Prepare immutable, reference-aligned six-state stage-controller banks."""
from __future__ import annotations
import argparse,copy,hashlib,json
from pathlib import Path
import mujoco
import numpy as np
import pandas as pd
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256,load_config,save_config
from dvgc.runtime import save_json
from cli.stage_label_pilot import evenly

OBJECTIVES={"takeoff":"takeoff_to_ascent","ascent":"ascent_to_apex","apex":"apex_to_descent"}

def aligned_reference_anchors(rows, frame, model, stage):
 hip_id=int(model.joint("hip_joint").id);knee_id=int(model.joint("knee_joint").id)
 hq=int(model.jnt_qposadr[hip_id]);kq=int(model.jnt_qposadr[knee_id])
 hv=int(model.jnt_dofadr[hip_id]);kv=int(model.jnt_dofadr[knee_id])
 angle_columns={"roll_angle","pitch_angle","yaw_angle","time"}
 angular_velocity=None
 if angle_columns.issubset(frame.columns):
  angles=np.unwrap(
   np.deg2rad(frame[["roll_angle","pitch_angle","yaw_angle"]].to_numpy(float)),
   axis=0,
  )
  angular_velocity=np.gradient(angles,frame["time"].to_numpy(float),axis=0)
 accepted=[];rejected={}
 for row in rows:
  reason=None;index=row.get("reference_index")
  if row.get("candidate_kind")!="reference_anchor":reason="not_reference_anchor"
  elif index is None or int(index)<0 or int(index)>=len(frame):reason="invalid_reference_index"
  else:
   ref=frame.iloc[int(index)]
   if not (model.jnt_range[hip_id,0]<=ref.hip_position<=model.jnt_range[hip_id,1]
           and model.jnt_range[knee_id,0]<=ref.knee_position<=model.jnt_range[knee_id,1]):
    reason="reference_joint_outside_authoritative_xml_range"
   elif not np.allclose(
       [row["qpos"][hq],row["qpos"][kq]],[ref.hip_position,ref.knee_position],
       atol=2e-5,rtol=0,
   ):reason="joint_qpos_not_time_aligned"
   elif not np.allclose(
       [row["qvel"][hv],row["qvel"][kv]],[ref.hip_velocity,ref.knee_velocity],
       atol=2e-5,rtol=0,
   ):reason="joint_qvel_not_time_aligned"
   elif not np.allclose(
       np.asarray(row["qvel"][:3],float),
       [ref.vel_x,ref.vel_y,ref.vel_z],atol=2e-5,rtol=0,
   ):reason="root_linear_velocity_not_time_aligned"
   elif angular_velocity is not None and not np.allclose(
       np.asarray(row["qvel"][3:6],float),angular_velocity[int(index)],
       atol=2e-4,rtol=0,
   ):reason="root_angular_velocity_not_time_aligned"
   elif angular_velocity is not None and not np.allclose(
       np.asarray(row["physical_feature"][3:6],float),
       np.deg2rad([ref.roll_angle,ref.pitch_angle,ref.yaw_angle]),
       atol=2e-4,rtol=0,
   ):reason="root_orientation_not_time_aligned"
   elif stage=="ascent" and abs(float(ref.vel_z))<=0.25:
    reason="premature_apex_entry_at_reset"
   elif (stage=="apex" and int(index)>=int(frame["pos_z"].idxmax())
         and float(ref.vel_z)<=-0.05):
    reason="premature_descent_entry_at_reset"
   elif row.get("flight_subinterval")!=stage:reason="wrong_subinterval"
   elif int(row.get("oracle_phase",-1))!=2:reason="wrong_oracle_phase"
   elif int(row.get("policy_state",{}).get("filter_phase",-1))!=2:
    reason="policy_state_phase_mismatch"
   elif "obs_history" not in row.get("policy_state",{}):
    reason="missing_observation_history"
   elif "last_action" not in row.get("policy_state",{}):
    reason="missing_previous_action"
  if reason:
   rejected[reason]=rejected.get(reason,0)+1
  else:
   accepted.append(row)
 return accepted,rejected

def main():
 p=argparse.ArgumentParser();p.add_argument('--takeoff-bank',required=True);p.add_argument('--flight-bank',required=True);p.add_argument('--output-root',required=True);p.add_argument('--config',default='configs/default.json');p.add_argument('--reference',default='data/reference_jump.csv');p.add_argument('--only-stage',choices=tuple(OBJECTIVES));a=p.parse_args();root=Path(a.output_root);root.mkdir(parents=True,exist_ok=True)
 takeoff=SnapshotBank.load(a.takeoff_bank);flight=SnapshotBank.load(a.flight_bank)
 sources={"takeoff":takeoff.records_for_phase('takeoff'),"ascent":[r for r in flight.records if r.get('flight_subinterval')=='ascent'],"apex":[r for r in flight.records if r.get('flight_subinterval')=='apex']};report={"status":"PASS","artifact_role":"stage_controller_pilot_inputs","stages":{}}
 if a.only_stage:sources={a.only_stage:sources[a.only_stage]}
 cfg_base=load_config(a.config);model=mujoco.MjModel.from_xml_path(str(cfg_base.xml_path));reference=pd.read_csv(a.reference)
 for stage,rows in sources.items():
  rejected={}
  if stage!="takeoff":
   rows,rejected=aligned_reference_anchors(rows,reference,model,stage)
  if stage=="takeoff":
   canonical=[r for r in rows if r.get("candidate_kind")=="canonical_compressed"]
   aligned=[r for r in rows if r.get("candidate_kind")=="reference_aligned_compressed"]
   chosen=evenly(canonical,3)+evenly(aligned,3)
  else:
   chosen=evenly(rows,6)
  if len(chosen)!=6:raise SystemExit(f'{stage} has {len(chosen)} pilot states')
  stage_root=root/stage;stage_root.mkdir(exist_ok=True);bank_path=stage_root/'reset_bank.pkl';config_path=stage_root/'config.json'
  source_hash=file_sha256(a.takeoff_bank if stage=='takeoff' else a.flight_bank)
  reset_protocol={"version":"stage_reference_aligned_reset_v1","stage":stage,"source_bank_sha256":source_hash,"xml_sha256":file_sha256(cfg_base.xml_path),"reference_sha256":file_sha256(a.reference),"joint_contract":"XML key initial_state" if stage=="takeoff" else "same reference index qpos/qvel/root pose/velocity"}
  reset_protocol["sha256"]=hashlib.sha256(json.dumps(reset_protocol,sort_keys=True,separators=(",",":")).encode()).hexdigest()
  metadata={"artifact_role":"proposal_support_set_stage_controller_pilot","stage":stage,"objective":OBJECTIVES[stage],"source_bank_sha256":source_hash,"certified_tube":False,"reset_protocol":reset_protocol,"reset_protocol_sha256":reset_protocol["sha256"]}
  SnapshotBank([copy.deepcopy(r) for r in chosen],metadata).save(bank_path)
  cfg=load_config(a.config,{"training_stage":"takeoff" if stage=='takeoff' else "flight","stage_reachability_objective":OBJECTIVES[stage],"use_bank_resets":True,"domain_randomization":False,"obs_noise_enable":False,"stage_curriculum_scale":0.0});save_config(cfg,config_path)
  report['stages'][stage]={"states":6,"ids":[r['id'] for r in chosen],"reference_indices":[r.get('reference_index') for r in chosen],"bank":str(bank_path),"bank_sha256":file_sha256(bank_path),"config":str(config_path),"objective":OBJECTIVES[stage],"reset_protocol_sha256":reset_protocol["sha256"],"reference_alignment":{"eligible_anchors":len(rows),"rejected":rejected,"qpos_qvel_time_aligned":stage=="takeoff" or len(chosen)==6,"root_pose_velocity_time_aligned":stage=="takeoff" or len(chosen)==6,"t0_next_stage_false":True}}
 save_json(root/'inputs.json',report)
if __name__=='__main__':main()
