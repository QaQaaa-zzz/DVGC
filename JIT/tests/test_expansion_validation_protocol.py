from __future__ import annotations

import numpy as np
import pytest


def test_disjointness_rejects_train_parent_or_exact_state():
    from jit_dvgc.expansion_validation_protocol import audit_group_disjointness

    train_rows = [
        {
            "parent_group_id": "train_parent",
            "state_sha256": "a" * 64,
            "actor_observation": [0.0, 1.0],
        }
    ]
    with pytest.raises(ValueError, match="TRAIN parent group"):
        audit_group_disjointness(
            train_rows,
            [{"parent_group_id": "train_parent", "state_sha256": "b" * 64,
              "actor_observation": np.asarray([2.0, 3.0])}],
            observation_atol=0.01,
        )
    with pytest.raises(ValueError, match="TRAIN physical state"):
        audit_group_disjointness(
            train_rows,
            [{"parent_group_id": "validation_parent", "state_sha256": "a" * 64,
              "actor_observation": np.asarray([2.0, 3.0])}],
            observation_atol=0.01,
        )


def test_disjointness_rejects_near_duplicate_observation():
    from jit_dvgc.expansion_validation_protocol import audit_group_disjointness

    train_rows = [
        {
            "parent_group_id": "train_parent",
            "state_sha256": "a" * 64,
            "actor_observation": [0.0, 1.0],
        }
    ]
    with pytest.raises(ValueError, match="near-duplicate"):
        audit_group_disjointness(
            train_rows,
            [{"parent_group_id": "validation_parent", "state_sha256": "b" * 64,
              "actor_observation": np.asarray([0.005, 1.005])}],
            observation_atol=0.01,
        )


def test_real_iter0_validation_protocol_audit(jit_root):
    from jit_dvgc.expansion_validation_protocol import audit_expansion_validation_protocol

    audit = audit_expansion_validation_protocol(
        jit_root / "configs/envelope_iter0_expansion_validation.json"
    )
    assert audit["status"] == "protocol_ready"
    assert audit["validation_parent_group_count_by_phase"] == {
        "upstream": 3,
        "downstream": 2,
    }
    assert audit["attempt_count"] == 160
    assert audit["maximum_acquisition_environment_interactions"] == 696
    assert audit["maximum_labeling_environment_interactions"] == 64000
    assert audit["train_parent_overlap_count"] == 0
    assert audit["exact_state_overlap_count"] == 0
    assert audit["near_duplicate_overlap_count"] == 0
    assert audit["test_data_used"] is False
    assert audit["final_evaluation_data_used"] is False
    assert audit["claim_boundary"]["continuation_field_trained"] is False
