import numpy as np

from dvgc.observation_audit import array_sha256, history_alignment


def test_history_alignment_detects_post_current_off_by_one():
    frames=np.arange(16,dtype=np.float32).reshape(4,4)
    result=history_alignment(frames.reshape(-1),frames[1:],frame_dim=4)
    assert not result["saved_equals_required_pre_current"]
    assert result["saved_equals_post_current"]


def test_array_hash_includes_shape_and_dtype():
    value=np.arange(4,dtype=np.float32)
    assert array_sha256(value)==array_sha256(value.copy())
    assert array_sha256(value)!=array_sha256(value.astype(np.float64))
