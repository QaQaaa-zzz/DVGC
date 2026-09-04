from __future__ import annotations

import json

import pytest

from jit_dvgc.causal_frontier_protocol import _policy_family_capability_phase_counts
from jit_dvgc.iterative_acceptance_gate import (
    _acceptance_policy_success_criterion,
    _candidate_core_success_criterion,
    _outcome,
    _reuse_core_baseline,
    _selected_family_outcome,
)
from jit_dvgc.policy_family_landing import (
    _load_completed_evaluator,
    merge_any_policy_landing_labels,
)


def _row(policy: str, candidate: str, state: str, label: int) -> dict:
    return {
        "candidate_id": candidate,
        "state_sha256": state,
        "phase": "downstream",
        "phase_index": 1,
        "parent_group_id": "group-a",
        "parent_state_sha256": "f" * 64,
        "snapshot": f"snapshots/{candidate}",
        "source_bank": "boundary_bank",
        "actor_observation": [0.0] * 76,
        "label": label,
        "continuation_success": bool(label),
        "outcome_class": "first_valid_landing" if label else "airborne_physical_failure",
        "environment_interactions": 5,
        "evaluator_policy_name": policy,
        "evaluator_actor_sha256": policy[0] * 64,
        "evaluator_payload_sha256": policy[-1] * 64,
    }


def test_any_policy_merge_marks_candidate_positive_and_records_controller():
    merged, identity = merge_any_policy_landing_labels(
        {
            "pi_0": [_row("pi_0", "c0", "a" * 64, 0)],
            "pi_1": [_row("pi_1", "c0", "a" * 64, 1)],
            "pi_2": [_row("pi_2", "c0", "a" * 64, 0)],
        }
    )

    assert len(merged) == 1
    assert merged[0]["label"] == 1
    assert merged[0]["successful_policy_names"] == ["pi_1"]
    assert set(merged[0]["per_policy_outcomes"]) == {"pi_0", "pi_1", "pi_2"}
    assert identity["policy_names"] == ["pi_0", "pi_1", "pi_2"]
    assert len(identity["policy_family_sha256"]) == 64
    assert merged[0]["policy_actor_sha256"] == identity["actor_family_sha256"]
    assert merged[0]["policy_payload_sha256"] == identity["payload_family_sha256"]


def test_any_policy_merge_rejects_candidate_identity_drift():
    with pytest.raises(ValueError, match="candidate identity drift"):
        merge_any_policy_landing_labels(
            {
                "pi_0": [_row("pi_0", "c0", "a" * 64, 0)],
                "pi_1": [_row("pi_1", "c0", "b" * 64, 1)],
            }
        )


def test_completed_evaluator_can_be_reused_after_family_run_interruption(tmp_path):
    evaluator = {
        "name": "pi_0",
        "actor_sha256": "a" * 64,
        "payload_sha256": "b" * 64,
    }
    rows = [_row("pi_0", "c0", "c" * 64, 1)]
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "evaluator_policy_name": "pi_0",
                "policy_actor_sha256": "a" * 64,
                "policy_payload_sha256": "b" * 64,
                "success_criterion": "first_valid_landing",
                "label_count": 1,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "labels.json").write_text(json.dumps(rows), encoding="utf-8")

    report, loaded = _load_completed_evaluator(tmp_path, evaluator=evaluator)

    assert report["status"] == "completed"
    assert loaded == rows


def test_policy_family_capability_support_does_not_require_artificial_negatives():
    rows = [
        {"phase": phase, "label": 1, "parent_group_id": f"{phase}-group"}
        for phase in ("upstream", "downstream")
    ]

    counts = _policy_family_capability_phase_counts(rows)

    assert counts["upstream"]["negative_count"] == 0
    assert counts["downstream"]["positive_count"] == 1


def test_acceptance_gate_extracts_selected_member_from_policy_family():
    selected = {
        "policy_name": "pi_1",
        "actor_sha256": "a" * 64,
        "payload_sha256": "b" * 64,
    }
    manifest = {
        "policy_identity_kind": "frozen_policy_family",
        "continuation_success_criterion": "first_valid_landing_before_physical_failure",
        "continuation_policy_family": {
            "members": [
                {
                    "name": "pi_1",
                    "actor_sha256": "a" * 64,
                    "payload_sha256": "b" * 64,
                }
            ]
        },
    }
    row = {
        "per_policy_outcomes": {
            "pi_1": {
                "label": 0,
                "outcome_class": "airborne_physical_failure",
                "environment_interactions": 7,
            }
        }
    }

    assert _acceptance_policy_success_criterion(manifest, selected) == (
        "first_valid_landing"
    )
    assert _selected_family_outcome(row, "pi_1") == {
        "label": 0,
        "outcome_class": "airborne_physical_failure",
        "environment_interactions": 7,
    }


def test_acceptance_gate_reuses_locked_core_without_new_rollouts():
    lock = {
        "schema": "jit_iterative_acceptance_baseline_lock_v1",
        "status": "locked_before_candidate_training",
        "source_iteration": 1,
        "baseline_actor_sha256": "a" * 64,
        "baseline_payload_sha256": "b" * 64,
        "source_tube_manifest_sha256": "c" * 64,
        "core_state_count": 1,
        "core": [{"state_sha256": "state-1", "baseline_success": True}],
    }

    rows = _reuse_core_baseline(
        lock,
        selected={"iteration": 1},
        policy_record={"actor_sha256": "a" * 64, "payload_sha256": "b" * 64},
        source_tube_manifest_sha256="c" * 64,
        source_state_sha256=["state-1"],
    )

    assert rows == lock["core"]


def test_candidate_boundary_uses_first_valid_landing_not_recovery():
    result = _outcome(
        (
            4,      # tick
            False,  # done
            False,  # terminal success
            False,  # physical failure
            False,  # timeout
            True,   # apex seen
            True,   # phase transitioned
            False,  # recovery success
            False,  # expert switching
            True,   # valid landing contact seen
        ),
        start_phase=0,
        max_ticks=100,
        success_criterion="first_valid_landing",
    )

    assert result == {
        "success": True,
        "outcome_class": "first_valid_landing",
        "environment_interactions": 4,
    }


def test_active_family_landing_gate_uses_landing_for_core_preservation():
    assert _candidate_core_success_criterion(
        {"acceptance_success_criterion": "first_valid_landing"}
    ) == "first_valid_landing"
    assert _candidate_core_success_criterion(
        {"acceptance_success_criterion": "stable_recovery"}
    ) == "stable_recovery"
