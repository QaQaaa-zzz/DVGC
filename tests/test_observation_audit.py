import copy

import numpy as np
import pytest

from dvgc.delay_probe import active_prefix_repeat_comparison
from dvgc.observation_audit import array_sha256, history_alignment
from dvgc.snapshot_timing import (
    J12_DELAY_SEQUENCE,
    SNAPSHOT_SCHEMA_NAME,
    SNAPSHOT_SCHEMA_VERSION,
    causal_prior_packet,
    select_delayed_packet,
    validate_snapshot_v4,
    validate_transfer_eligibility,
)


def test_history_alignment_detects_post_current_off_by_one():
    frames=np.arange(16,dtype=np.float32).reshape(4,4)
    result=history_alignment(frames.reshape(-1),frames[1:],frame_dim=4)
    assert not result["saved_equals_required_pre_current"]
    assert result["saved_equals_post_current"]


def test_array_hash_includes_shape_and_dtype():
    value=np.arange(4,dtype=np.float32)
    assert array_sha256(value)==array_sha256(value.copy())
    assert array_sha256(value)!=array_sha256(value.astype(np.float64))


def test_legacy_causal_padding_is_explicitly_diagnostic():
    frames=np.arange(16,dtype=np.float32).reshape(4,4)
    assert np.array_equal(causal_prior_packet(frames.reshape(-1),1,frame_dim=4).reshape(4,4),
                          np.stack((frames[0],frames[0],frames[1],frames[2])))


def test_complete_packet_fifo_never_shifts_individual_signals():
    queue=np.arange(3*8,dtype=np.float32).reshape(3,8)
    assert np.array_equal(select_delayed_packet(queue,0),queue[2])
    assert np.array_equal(select_delayed_packet(queue,1),queue[1])
    assert np.array_equal(select_delayed_packet(queue,2),queue[0])


def test_v4_delay_requires_three_real_packets():
    with pytest.raises(ValueError):
        select_delayed_packet(np.zeros((2,8),np.float32),1)


def test_jitter_schedule_is_fixed_and_causal():
    assert len(J12_DELAY_SEQUENCE)==24
    assert set(J12_DELAY_SEQUENCE)=={1,2}


def _estimator():
    scalar={
        "phase":2,"estimated_phase":2,"had_airborne":1,"had_valid_landing":0,
        "airborne_count":3,"prelaunch_airborne_count":0,"landing_bounce_count":0,
        "invalid_wheel_count":0,"recovery_count":0,"contact_age":0,
        "landing_entry_age":0,"landing_phase_step":0,"prev_acc_z":0.2,
        "prev_vz":-0.1,"prev_front_tire_bottom_z":0.3,"prev_rear_tire_bottom_z":0.3,
        "positive_pitch_count":0,"wheelie_count":0,"dual_wheel_liftoff_seen":True,
        "stage_entry_ever":0,"apex_seen":1,"jump_signal_latched":True,
        "jump_window_start_x":2.0,"jump_window_end_x":3.0,"chain_ever":0,
        "recovery_success":0,"episode_step":7,"end_code":0,
    }
    return scalar|{"phase_probs":np.array([0,0,1,0],np.float32)}


def _record():
    # Five consecutive frames provide three real overlapping four-frame packets.
    frames=np.arange(6*4,dtype=np.float32).reshape(6,4)
    fifo=np.stack([frames[0:4].reshape(-1),frames[1:5].reshape(-1),frames[2:6].reshape(-1)])
    pre=frames[2:5];current=frames[5];action=np.array([.1,.2,.3,.4],np.float32)
    hashes={name:"hash" for name in ("xml_sha256","config_sha256","policy_params_sha256","policy_config_sha256","policy_manifest_sha256","normalizer_sha256","source_fingerprint")}
    hashes["action_mapping_version"]="mapping"
    return {
        "schema_name":SNAPSHOT_SCHEMA_NAME,"schema_version":SNAPSHOT_SCHEMA_VERSION,
        "physical_state_t":{"qpos":np.zeros(12,np.float32),"qvel":np.zeros(11,np.float32),"act":np.zeros(0,np.float32),"ctrl_previous":np.zeros(4,np.float32),"qacc_warmstart":np.zeros(11,np.float32),"sensordata":np.zeros(32,np.float32),"time":np.array(0.14,np.float32)},
        "obs_history_pre_t":pre,"current_frame_t":current,"actor_observation_t":fifo[2],
        "obs_history_post_t":frames[3:6],"actor_packet_fifo_t":fifo,
        "estimator_state_pre_t":_estimator(),"estimator_state_post_t":_estimator(),
        "last_normalized_command_t":np.zeros(4,np.float32),"policy_action_t":action,
        "ctrl_applied_t":action*2,"rng_state_t":np.array([1,2],np.uint32),
        "field_ticks":{"physical_state_t":7,"actor_observation_t":7,"current_frame_t":7,"policy_action_t":7,"ctrl_applied_t":7,"ctrl_previous":6,"actor_packet_fifo_t":[5,6,7]},
        "simulation_timestamps":{"physical_state_t":.14,"ctrl_applied_t":.14},
        "provenance":hashes,
    }


def _validate(record):
    hashes=record["provenance"]
    return validate_snapshot_v4(record,frame_dim=4,expected_shapes={"qpos":(12,),"qvel":(11,),"act":(0,),"ctrl_previous":(4,),"qacc_warmstart":(11,),"sensordata":(32,)},expected_hashes=hashes,actor_action_fn=lambda _:np.array([.1,.2,.3,.4],np.float32),ctrl_from_action_fn=lambda action:action*2,current_frame_fn=lambda _:np.arange(20,24,dtype=np.float32))


def test_snapshot_v4_positive_contract():
    assert _validate(_record())["valid"]


@pytest.mark.parametrize("mutation",[
    "schema","qpos_shape","qvel_shape","nan","pre","current","post","fifo",
    "tick","policy_action","ctrl","xml_hash","normalizer_hash","policy_hash",
    "estimator_counter","actor_sidecar",
])
def test_snapshot_v4_negative_controls(mutation):
    row=copy.deepcopy(_record())
    if mutation=="schema":row["schema_version"]=3
    elif mutation=="qpos_shape":row["physical_state_t"]["qpos"]=np.zeros(11,np.float32)
    elif mutation=="qvel_shape":row["physical_state_t"]["qvel"]=np.zeros(10,np.float32)
    elif mutation=="nan":row["physical_state_t"]["qpos"][0]=np.nan
    elif mutation=="pre":row["obs_history_pre_t"][0,0]+=1
    elif mutation=="current":row["current_frame_t"][0]+=1
    elif mutation=="post":row["obs_history_post_t"][0,0]+=1
    elif mutation=="fifo":row["actor_packet_fifo_t"][0,4]+=1
    elif mutation=="tick":row["field_ticks"]["policy_action_t"]=8
    elif mutation=="policy_action":row["policy_action_t"][0]+=1
    elif mutation=="ctrl":row["ctrl_applied_t"][0]+=1
    elif mutation=="xml_hash":row["provenance"]["xml_sha256"]="bad"
    elif mutation=="normalizer_hash":row["provenance"]["normalizer_sha256"]="bad"
    elif mutation=="policy_hash":row["provenance"]["policy_params_sha256"]="bad"
    elif mutation=="estimator_counter":del row["estimator_state_pre_t"]["recovery_count"]
    elif mutation=="actor_sidecar":row["actor_observation_t"][0]+=1
    expected=_record()["provenance"]
    result=validate_snapshot_v4(row,frame_dim=4,expected_shapes={"qpos":(12,),"qvel":(11,),"act":(0,),"ctrl_previous":(4,),"qacc_warmstart":(11,),"sensordata":(32,)},expected_hashes=expected,actor_action_fn=lambda _:np.array([.1,.2,.3,.4],np.float32),ctrl_from_action_fn=lambda action:action*2,current_frame_fn=lambda _:np.arange(20,24,dtype=np.float32))
    assert not result["valid"],mutation


def test_terminal_tail_does_not_invalidate_active_prefix():
    one={name:np.array([value]) for name,value in {"survival":2,"minimum_margin":.1,"terminal_margin":.2,"end_code":5,"termination_tick":3,"landing_entry":False,"chain":False,"recovery_success":False,"final_recovery":False}.items()}
    one.update({"actions":np.zeros((5,1,4)),"active_action_mask":np.array([[1],[1],[1],[0],[0]],bool),"phase_trace":np.zeros((5,1),int),"contact_age_trace":np.zeros((5,1),int)})
    two=copy.deepcopy(one);two["actions"][4,0]=99;two["phase_trace"][4,0]=99
    assert active_prefix_repeat_comparison(one,two)["exact"]
    two["actions"][1,0]=1
    assert not active_prefix_repeat_comparison(one,two)["exact"]


def test_pair_asset_or_semantics_change_fails_eligibility():
    row={"source_snapshot_index":0,"target_snapshot_index":1,"source_snapshot_hash":"a","target_snapshot_hash":"b","correction_sha256":"c","phase":2,"contact_mode":0,"failure_precursor":"pitch","delay_semantics":"packet_fifo_v4"}
    assert validate_transfer_eligibility([row],[copy.deepcopy(row)],expected_artifact_sha256="h",actual_artifact_sha256="h")["valid"]
    changed=copy.deepcopy(row);changed["correction_sha256"]="bad"
    assert not validate_transfer_eligibility([row],[changed],expected_artifact_sha256="h",actual_artifact_sha256="h")["valid"]
    assert not validate_transfer_eligibility([row],[row],expected_artifact_sha256="h",actual_artifact_sha256="bad")["valid"]
