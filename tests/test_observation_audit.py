import numpy as np

from dvgc.observation_audit import array_sha256, history_alignment
from dvgc.snapshot_timing import (
    J12_DELAY_SEQUENCE,
    causal_prior_packet,
    select_delayed_packet,
    snapshot_v2_contract,
    validate_snapshot_v2,
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


def test_causal_delay_moves_complete_frames_and_holds_oldest():
    frames=np.arange(16,dtype=np.float32).reshape(4,4)
    assert np.array_equal(causal_prior_packet(frames.reshape(-1),1,frame_dim=4).reshape(4,4),
                          np.stack((frames[0],frames[0],frames[1],frames[2])))
    assert np.array_equal(causal_prior_packet(frames.reshape(-1),2,frame_dim=4).reshape(4,4),
                          np.stack((frames[0],frames[0],frames[0],frames[1])))


def test_complete_packet_fifo_never_shifts_individual_signals():
    queue=np.arange(3*8,dtype=np.float32).reshape(3,8)
    assert np.array_equal(select_delayed_packet(queue,0),queue[2])
    assert np.array_equal(select_delayed_packet(queue,1),queue[1])
    assert np.array_equal(select_delayed_packet(queue,2),queue[0])


def test_jitter_schedule_is_fixed_and_causal():
    assert len(J12_DELAY_SEQUENCE)==24
    assert set(J12_DELAY_SEQUENCE)=={1,2}
    assert J12_DELAY_SEQUENCE[:8]==(1,2,1,1,2,1,2,1)


def test_snapshot_v2_requires_logged_and_independent_identity():
    pre=np.arange(12,dtype=np.float32).reshape(3,4);frame=np.arange(4,dtype=np.float32)+20
    record={key:{} for key in snapshot_v2_contract()["required_fields"]}
    record.update({"obs_history_pre_t":pre,"current_frame_t":frame,
                   "actor_observation_t":np.concatenate((pre.reshape(-1),frame)),
                   "obs_history_post_t":np.concatenate((pre[1:],frame[None])),
                   "field_ticks":{"physical_state_t":7}})
    assert validate_snapshot_v2(record,frame_dim=4)["valid"]
    record["actor_observation_t"]=np.zeros(16,np.float32)
    result=validate_snapshot_v2(record,frame_dim=4)
    assert not result["valid"]
    assert not result["checks"]["logged_equals_independent_reconstruction"]
