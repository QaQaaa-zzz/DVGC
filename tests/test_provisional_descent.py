import copy

import numpy as np

from dvgc.provisional_descent import (
    SCHEMA_VERSION, StratifiedRSISampler, greedy_clusters, state_identity,
    tolerance_unique, validate_candidate,
)


def _row(index=0, label="provisional_core", layer="early"):
    feature = np.zeros(16, np.float32); feature[0] = index
    policy = {
        "last_action": np.zeros(4, np.float32),
        "obs_history": np.zeros((3, 35), np.float32),
        "actor_observation": np.zeros(140, np.float32),
        "filter_phase": 2,
        "phase_probs": np.asarray([0, 0, 1, 0], np.float32),
        "delay_buffer": np.asarray([[0, 0, 1, 0]], np.float32),
        "prev_acc_z": 0.0, "prev_vz": -0.1,
    }
    return {
        "candidate_schema": SCHEMA_VERSION, "provisional_label": label,
        "descent_layer": layer, "candidate_source": "natural_continuous",
        "artifact_role": "proposal_support_bank", "formal_tube_member": False,
        "formal_jel_member": False, "qpos": np.zeros(9, np.float32),
        "qvel": np.zeros(8, np.float32), "ctrl": np.zeros(4, np.float32),
        "qacc_warmstart": np.zeros(8, np.float32),
        "physical_feature": feature, "policy_state": policy,
    }


def test_schema_identity_dedup_and_clusters():
    one = _row(); two = copy.deepcopy(one); two["physical_feature"][0] = .001
    validate_candidate(one)
    assert state_identity(one) == state_identity(copy.deepcopy(one))
    unique, rejected = tolerance_unique([one, two], np.ones(16), .01)
    assert len(unique) == 1 and rejected == 1
    assert greedy_clusters([one, _row(5)], np.ones(16), 1.0) == [0, 1]


def test_stratified_sampler_resume_is_exact():
    rows = [_row(i, label, layer) for i, (label, layer) in enumerate((
        ("provisional_core", "early"), ("provisional_core", "late"),
        ("provisional_frontier", "middle"),
    ))]
    sampler = StratifiedRSISampler(rows, seed=7)
    sampler.sample_indices(5); state = sampler.state_dict()
    expected = sampler.sample_indices(12)
    resumed = StratifiedRSISampler(rows, seed=99); resumed.load_state_dict(state)
    assert resumed.sample_indices(12) == expected
