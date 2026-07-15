import numpy as np

from dvgc.flight_augmentation import (
    clone_as_anchors, interpolate_state, normalized_distance,
    proportional_quotas, quat_slerp,
)


def _row(index=1):
    return {
        "id":str(index),"reference_index":index,"qpos":np.r_[np.zeros(3),[1.,0,0,0],np.zeros(2)],
        "qvel":np.zeros(8),"ctrl":np.zeros(4),"qacc_warmstart":np.zeros(8),
        "physical_feature":np.arange(16,dtype=np.float32),"candidate_kind":"reference_envelope",
    }


def test_reference_proportional_quotas_are_deterministic():
    assert proportional_quotas({"ascent":79,"apex":25,"descent":97},160)=={"ascent":63,"apex":20,"descent":77}


def test_anchor_clone_changes_no_state_arrays():
    source=_row(); cloned=clone_as_anchors([source])[0]
    assert cloned["candidate_kind"]=="reference_anchor" and cloned["source_index"]==1
    for key in ("qpos","qvel","ctrl","qacc_warmstart","physical_feature"):
        np.testing.assert_array_equal(source[key],cloned[key])


def test_quaternion_slerp_and_state_interpolation_follow_manifold():
    start=np.array([1.,0,0,0]); end=np.array([0.,0,0,1.])
    middle=quat_slerp(start,end,.5)
    assert np.linalg.norm(middle)==1.0
    np.testing.assert_allclose(np.abs(middle),[2**-.5,0,0,2**-.5],atol=1e-7)
    a=_row(); b=_row(2); b["qpos"]=a["qpos"].copy(); b["qpos"][3:7]=end; b["qvel"][:]=2
    q,v,_=interpolate_state(a,b,.5)
    np.testing.assert_allclose(np.abs(q[3:7]),np.abs(middle)); np.testing.assert_allclose(v,1)


def test_normalized_nearest_neighbor_uses_all_existing_candidates():
    existing=np.asarray([[0.,0.],[1.,1.]])
    assert normalized_distance([.5,.5],existing,[1.,2.])==np.linalg.norm([.5,.25])
