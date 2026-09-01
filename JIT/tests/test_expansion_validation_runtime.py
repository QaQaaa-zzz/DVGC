from __future__ import annotations

from collections import Counter

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
    assert {row["strength"] for row in upstream} == {0.025, 0.05, 0.1}
    assert {row["duration"] for row in upstream} == {1, 2}
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
