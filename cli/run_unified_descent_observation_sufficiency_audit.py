"""Audit immutable Descent snapshot observation authority before statistics."""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import subprocess
from pathlib import Path

import jax
import numpy as np

from cli.run_unified_descent_feedback_probe import _assets
from dvgc.config import ACTION_MAPPING_VERSION, file_sha256
from dvgc.descent_supervised import build_actor_tools
from dvgc.observation_audit import array_sha256, history_alignment
from dvgc.ppo_integrity import normalizer_summary
from dvgc.rollout import restore_snapshot
from dvgc.runtime import save_json


EXPECTED_HEAD="a781250"
SNAPSHOTS=Path("runs/unified_descent_feedback_teacher_support_and_representation_probe_v1/feedback_probe_snapshots.pkl")
AUTHORITY=Path("runs/unified_descent_feedback_teacher_support_and_representation_probe_v1_replay_corrected/local_cem_authority_results.json")
TRANSFER=Path("runs/unified_descent_feedback_correction_transfer_and_support_geometry_audit_v1/feedback_correction_cross_snapshot_transfer_matrix.json")
EXPECTED_NORMALIZER="8f2e36b6f69a3d20da67c1854f7e908c98dd6b03ae70e287e0a7e28522f93a7e"
POLICY_STATE_KEYS=("last_action","obs_history","actor_observation","filter_phase","phase_probs","contact_probs","phase_progress","phase_confidence","estimator_hidden","delay_buffer","prev_acc_z","prev_vz")
FRAME=(
 ("steering_joint_position","rad","sensor"),("hip_joint_position","rad","sensor"),("knee_joint_position","rad","sensor"),("rear_wheel_angular_velocity","rad/s","sensor"),
 ("steering_joint_velocity","rad/s","sensor"),("hip_joint_velocity","rad/s","sensor"),("knee_joint_velocity","rad/s","sensor"),("front_wheel_angular_velocity","rad/s","sensor"),
 ("gyro_x","rad/s","sensor"),("gyro_y","rad/s","sensor"),("gyro_z","rad/s","sensor"),("gravity_body_x","1","estimated"),("gravity_body_y","1","estimated"),("gravity_body_z","1","estimated"),
 ("last_normalized_action_steer","normalized","command_history"),("last_normalized_action_drive","normalized","command_history"),("last_normalized_action_hip","normalized","command_history"),("last_normalized_action_knee","normalized","command_history"),
 ("task_step_front_relative_x","scaled","command/map"),("task_step_back_relative_x","scaled","command/map"),("task_step_height","scaled","command/map"),("task_target_speed","scaled","command"),("task_terrain_relative_height","scaled","sensor+map"),
 ("accelerometer_x","m/s^2","sensor"),("accelerometer_y","m/s^2","sensor"),("accelerometer_z","m/s^2","sensor"),("accelerometer_z_delta","m/s^2","sensor_history"),("rear_wheel_linear_speed","m/s","sensor"),("imu_support_estimate","boolean","estimated"),
 ("phase_probability_approach","probability","estimated"),("phase_probability_takeoff","probability","estimated"),("phase_probability_flight","probability","estimated"),("phase_probability_landing","probability","estimated"),("contact_age","normalized","estimated"),("recovery_count","normalized","estimated"),
)


def _contract(env,params):
    normalizer=normalizer_summary(params[0]);dims=[]
    for history_index in range(env._actor_history_steps):
        age=env._actor_history_steps-1-history_index
        for frame_index,(name,unit,source) in enumerate(FRAME):
            dims.append({"index":history_index*35+frame_index,"name":f"history_t_minus_{age}/{name}","frame_index":frame_index,"history_age_ticks":age,"unit":unit,"source":source,"sampling_tick":"control state before current policy evaluation"})
    return {"action_order":["steer","drive","hip","knee"],"action_mapping_version":ACTION_MAPPING_VERSION,"actor_input_shape":[140],"frame_dim":35,"history_steps":4,"history_order":"oldest_to_current","dimensions":dims,"normalization":{"location":"Brax policy network apply","type":"frozen running mean/std per flattened dimension","sha256":normalizer["sha256"],"count":normalizer["count"],"clipping":None},"environment_observation_clipping":None,"obs_noise_enable":False,"command_action_distinction":{"policy_output":"deterministic post-tanh normalized action in [-1,1]","commanded_action":"policy output after explicit clip; same four-channel order","last_action":"previous normalized commanded action embedded in every frame","applied_control":"four XML actuator targets in data.ctrl after _action_to_ctrl; units are actuator-specific, not normalized action","actuator_dynamic_state":f"MuJoCo na={env.mj_model.na}","snapshot_delay_buffer":"compatibility field equal to phase_probs[None,:], not an action-delay FIFO"},"history_update_contract":{"actor_observation_at_step":"pre-update obs_history concatenated with current frame","snapshot_saved_obs_history":"next_info obs_history after _advance_actor_history","required_snapshot_reconstruction":"must retain pre-update history or exact actor_observation sidecar"}}


def _independent_record(record):
    result=copy.deepcopy(record);result["policy_state"].pop("actor_observation",None);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run",required=True);args=parser.parse_args();root=Path(args.run)
    if root.exists():raise SystemExit(f"refusing overwrite {root}")
    if subprocess.run(["git","merge-base","--is-ancestor",EXPECTED_HEAD,"HEAD"]).returncode or subprocess.check_output(["git","status","--porcelain"],text=True).strip():raise SystemExit("invalid git state")
    _,_,env,params=_assets();normalizer=normalizer_summary(params[0]);
    if normalizer["sha256"]!=EXPECTED_NORMALIZER:raise SystemExit("normalizer gate")
    with SNAPSHOTS.open("rb") as handle:snapshots=pickle.load(handle)
    authority=json.loads(AUTHORITY.read_text());transfer=json.loads(TRANSFER.read_text())
    if len(snapshots)!=24 or len(authority["rows"])!=24 or len(transfer["pairs"])!=244:raise SystemExit("frozen target count gate")
    _,actor_action,_=build_actor_tools(env,params);rows=[]
    for index,item in enumerate(snapshots):
        record=item["snapshot"];seed=int(item["generation_seed"]);stored=np.asarray(record["policy_state"]["actor_observation"],np.float32)
        override1=restore_snapshot(env,record,jax.random.PRNGKey(seed));override2=restore_snapshot(env,record,jax.random.PRNGKey(seed));independent1=restore_snapshot(env,_independent_record(record),jax.random.PRNGKey(seed));independent2=restore_snapshot(env,_independent_record(record),jax.random.PRNGKey(seed))
        override_obs1=np.asarray(override1.obs["state"],np.float32);override_obs2=np.asarray(override2.obs["state"],np.float32);independent_obs1=np.asarray(independent1.obs["state"],np.float32);independent_obs2=np.asarray(independent2.obs["state"],np.float32)
        stored_action=np.asarray(actor_action(params[1],stored));override_action1=np.asarray(actor_action(params[1],override_obs1));override_action2=np.asarray(actor_action(params[1],override_obs2));independent_action1=np.asarray(actor_action(params[1],independent_obs1));independent_action2=np.asarray(actor_action(params[1],independent_obs2))
        align=history_alignment(stored,record["policy_state"]["obs_history"])
        delay=np.asarray(record["policy_state"]["delay_buffer"]);phase=np.asarray(record["policy_state"]["phase_probs"])
        rows.append({"snapshot_index":index,"snapshot_hash":item["snapshot_hash"],"candidate_id":item["candidate_id"],"tick":item["tick"],"stored_shape":list(stored.shape),"stored_hash":array_sha256(stored),"override_restore":{"max_abs_error":float(np.max(np.abs(override_obs1-stored))),"hash":array_sha256(override_obs1),"repeat_bit_exact":bool(np.array_equal(override_obs1,override_obs2)),"action_max_abs_error":float(np.max(np.abs(override_action1-stored_action))),"action_repeat_bit_exact":bool(np.array_equal(override_action1,override_action2))},"independent_reconstruction":{"max_abs_error":float(np.max(np.abs(independent_obs1-stored))),"hash":array_sha256(independent_obs1),"repeat_bit_exact":bool(np.array_equal(independent_obs1,independent_obs2)),"action_max_abs_error":float(np.max(np.abs(independent_action1-stored_action))),"action_repeat_bit_exact":bool(np.array_equal(independent_action1,independent_action2))},"history_alignment":align,"policy_state_complete":all(key in record["policy_state"] for key in POLICY_STATE_KEYS),"phase_estimator_complete":all(key in record["policy_state"] for key in ("filter_phase","phase_probs","phase_confidence","phase_progress")),"contact_estimator_complete":"contact_probs" in record["policy_state"],"delay_compatibility_alias_matches_phase_probs":bool(delay.shape==(1,4) and np.array_equal(delay[0],phase)),"qacc_warmstart_present":"qacc_warmstart" in record})
    gate=all(row["stored_shape"]==[140] and row["override_restore"]["max_abs_error"]==0 and row["override_restore"]["action_max_abs_error"]==0 and row["independent_reconstruction"]["max_abs_error"]==0 and row["independent_reconstruction"]["action_max_abs_error"]==0 and row["history_alignment"]["saved_equals_required_pre_current"] and row["policy_state_complete"] and row["phase_estimator_complete"] and row["contact_estimator_complete"] and row["delay_compatibility_alias_matches_phase_probs"] for row in rows)
    root.mkdir(parents=True);manifest={"status":"PASS" if gate else "OBSERVATION_PIPELINE_AUTHORITY_FAILURE","gate":gate,"contract":_contract(env,params),"frozen_assets":{"snapshots":24,"authority_labels":24,"eligible_transfer_pairs":244,"snapshot_artifact_sha256":file_sha256(SNAPSHOTS),"authority_sha256":file_sha256(AUTHORITY),"transfer_sha256":file_sha256(TRANSFER),"normalizer":normalizer},"summary":{"override_restore_exact":sum(row["override_restore"]["max_abs_error"]==0 for row in rows),"independent_observation_exact":sum(row["independent_reconstruction"]["max_abs_error"]==0 for row in rows),"independent_action_exact":sum(row["independent_reconstruction"]["action_max_abs_error"]==0 for row in rows),"saved_history_is_required_pre_current":sum(row["history_alignment"]["saved_equals_required_pre_current"] for row in rows),"saved_history_is_post_current":sum(row["history_alignment"]["saved_equals_post_current"] for row in rows),"independent_observation_error_min":min(row["independent_reconstruction"]["max_abs_error"] for row in rows),"independent_observation_error_max":max(row["independent_reconstruction"]["max_abs_error"] for row in rows),"independent_action_error_min":min(row["independent_reconstruction"]["action_max_abs_error"] for row in rows),"independent_action_error_max":max(row["independent_reconstruction"]["action_max_abs_error"] for row in rows)},"root_cause":"snapshot stores next_info obs_history (post-current) while state.obs was formed from pre-update history plus current frame; actor_observation sidecar override masks the off-by-one during restore","heldout_used":False,"ppo_authorization":False,"bootstrap_authorization":False,"rows":rows}
    save_json(root/"observation_contract_and_alignment_manifest.json",manifest)
    panels={"status":"PREREGISTERED_NOT_ACTIVATED","reason":manifest["status"],"panels":{"V0":"current 4x35 actor observation","V1":"8-frame causal history","V2":"16-frame causal history","V3":"V2 per-signal current/mean/slope/min/max","V4":"V0 plus deployable command/control trace","V5":"V0 plus estimator confidence/progress history","P0":"complete privileged diagnostic","P1":"true phase/contact block","P2":"body dynamics block","P3":"actuation/delay block","P4":"relative geometry block","P5":"current constraint-margin block"},"selection_after_results":False,"heldout_used":False}
    save_json(root/"observation_sufficiency_panel_manifest.json",panels)
    reason="observation authority gate failed before statistical panels"
    for name in ("snapshot_support_separability_results.json","action_conditioned_transfer_separability_results.json","actor_visible_alias_pair_audit.json","privileged_block_reconstructability_audit.json"):
        save_json(root/name,{"status":"NOT_EXECUTED","reason":reason,"heldout_used":False,"ppo_authorization":False,"bootstrap_authorization":False})
    report={"experiment":"unified_descent_observation_history_sufficiency_audit_v1","status":"OBSERVATION_PIPELINE_AUTHORITY_FAILURE","causal_classification":"OBSERVATION_PIPELINE_AUTHORITY_FAILURE","alignment_summary":manifest["summary"],"v0_p0_reproduction":"NOT_EXECUTED","panel_results":"NOT_EXECUTED","observation_amendment_evidence":False,"required_next_step":"new audit version after snapshot history/schema correction; do not continue this version with repaired data","old_policy_evidence_inheritance":False,"heldout_used":False,"ppo_authorization":False,"bootstrap_authorization":False}
    save_json(root/"UNIFIED_DESCENT_OBSERVATION_HISTORY_SUFFICIENCY_AUDIT_V1_REPORT.json",report);print(json.dumps(report,indent=2))


if __name__=="__main__":main()
