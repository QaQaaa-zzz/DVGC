from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import numpy as np


def test_locked_validation_attempt_schedule_is_exact(jit_root):
    from jit_dvgc.expansion_validation_protocol import (
        load_expansion_validation_protocol_config,
    )
    from jit_dvgc.expansion_validation_runtime import enumerate_validation_attempts

    config = load_expansion_validation_protocol_config(
        jit_root / "configs/envelope_iter0_expansion_validation.json"
    )
    attempts = enumerate_validation_attempts(config["protocol"])
    assert len(attempts) == 160
    assert Counter(row["phase"] for row in attempts) == {
        "upstream": 144,
        "downstream": 16,
    }

    upstream = [row for row in attempts if row["phase"] == "upstream"]
    assert {row["action_name"] for row in upstream} == {
        "steer",
        "rear_wheel_drive",
        "hip",
        "knee",
    }
    assert {row["sign"] for row in upstream} == {-1, 1}
    assert {row["strength"] for row in upstream} == {0.025, 0.1}
    assert {row["duration"] for row in upstream} == {4, 8, 16}
    assert len({row["parent_group_id"] for row in upstream}) == 3

    downstream = [row for row in attempts if row["phase"] == "downstream"]
    assert {row["action_name"] for row in downstream} == {"hip"}
    assert {row["sign"] for row in downstream} == {1}
    assert {row["duration"] for row in downstream} == {30}
    assert {row["strength"] for row in downstream} == {
        0.15,
        0.2,
        0.3,
        0.32,
        0.35,
        0.4,
        0.45,
        0.5,
    }
    assert len({row["parent_group_id"] for row in downstream}) == 2


def test_validation_schedule_indices_are_stable_and_unique(jit_root):
    from jit_dvgc.expansion_validation_protocol import (
        load_expansion_validation_protocol_config,
    )
    from jit_dvgc.expansion_validation_runtime import enumerate_validation_attempts

    config = load_expansion_validation_protocol_config(
        jit_root / "configs/envelope_iter0_expansion_validation.json"
    )
    attempts = enumerate_validation_attempts(config["protocol"])
    assert [row["attempt_index"] for row in attempts] == list(range(160))
    identities = {
        (
            row["phase"],
            row["parent_group_id"],
            row["action_name"],
            row["sign"],
            row["strength"],
            row["duration"],
        )
        for row in attempts
    }
    assert len(identities) == 160


def test_train_near_duplicate_filter_uses_all_actor_features():
    from jit_dvgc.expansion_validation_runtime import _near_train_observation

    train = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    assert _near_train_observation(
        np.asarray([0.005, -0.005, 0.009], np.float32),
        train,
        atol=0.01,
    )
    assert not _near_train_observation(
        np.asarray([0.005, -0.005, 0.011], np.float32),
        train,
        atol=0.01,
    )


def test_unified_actor_observation_uses_phase_task_signal():
    from jit_dvgc.expansion_validation_runtime_preflight import (
        unified_actor_observation_from_legacy_snapshot,
    )

    frames = np.arange(75, dtype=np.float32).reshape(3, 25)
    snapshot = SimpleNamespace(
        observation_fifo=frames,
        events={"jump_signal": np.asarray(True)},
    )
    upstream = unified_actor_observation_from_legacy_snapshot(snapshot, phase="upstream")
    downstream = unified_actor_observation_from_legacy_snapshot(snapshot, phase="downstream")
    np.testing.assert_array_equal(upstream[:-1], frames.reshape(-1))
    np.testing.assert_array_equal(downstream[:-1], frames.reshape(-1))
    assert upstream[-1] == 1.0
    assert downstream[-1] == 0.0


def test_real_runtime_preflight_is_zero_interaction_and_outcome_blind(jit_root):
    from jit_dvgc.expansion_validation_runtime_preflight import (
        audit_expansion_validation_runtime_preflight,
    )

    audit = audit_expansion_validation_runtime_preflight(
        jit_root / "configs/envelope_iter0_expansion_validation.json"
    )
    assert audit["status"] == "runtime_preflight_ready"
    assert audit["scientific_protocol_sha256"] == (
        "9ec0a1e8c314cc5710688a3537fbd339f520d4db3c4268d20715bcde938586b0"
    )
    assert audit["validation_anchor_count"] == 5
    assert audit["validation_anchor_unified_observation_match_count"] == 5
    assert audit["validation_parent_group_count_by_phase"] == {
        "upstream": 3,
        "downstream": 2,
    }
    assert audit["attempt_count"] == 160
    assert audit["maximum_acquisition_environment_interactions"] == 1824
    assert audit["maximum_labeling_environment_interactions"] == 64000
    assert audit["train_parent_overlap_count"] == 0
    assert audit["exact_state_overlap_count"] == 0
    assert audit["near_duplicate_overlap_count"] == 0
    assert audit["environment_interactions"] == 0
    assert audit["training_transitions"] == 0
    assert audit["validation_outcomes_inspected"] is False
    assert audit["test_data_used"] is False
    assert audit["final_evaluation_data_used"] is False


def test_strict_continuation_semantics_remain_phase_aware():
    from jit_dvgc.unified_continuation_labels import classify_unified_continuation_outcome

    positive, outcome = classify_unified_continuation_outcome(
        start_phase=0,
        terminal_success=True,
        physical_failure=False,
        timeout=False,
        done=True,
        apex_seen=True,
        phase_transitioned=True,
        recovery_success=True,
        reached_rollout_horizon=False,
    )
    assert positive is True
    assert outcome == "success"

    positive, outcome = classify_unified_continuation_outcome(
        start_phase=0,
        terminal_success=True,
        physical_failure=False,
        timeout=False,
        done=True,
        apex_seen=True,
        phase_transitioned=False,
        recovery_success=True,
        reached_rollout_horizon=False,
    )
    assert positive is False
    assert outcome == "task_failure"

    positive, outcome = classify_unified_continuation_outcome(
        start_phase=1,
        terminal_success=True,
        physical_failure=False,
        timeout=False,
        done=True,
        apex_seen=False,
        phase_transitioned=False,
        recovery_success=True,
        reached_rollout_horizon=False,
    )
    assert positive is True
    assert outcome == "success"
