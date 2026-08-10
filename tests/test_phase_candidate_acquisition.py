from __future__ import annotations

import pytest

from dvgc.phase_candidate_acquisition import (
    AcquisitionParentSummary,
    build_provisional_continuation_label,
    evaluate_candidate_acquisition_gate,
    require_candidate_acquisition_integrity,
)
from dvgc.feasibility import validate_continuation_label


def _parents(count: int, *, success: bool = True):
    return tuple(
        AcquisitionParentSummary(
            seed=920_000 + index,
            trajectory_hash=f"{index + 1:064x}",
            success=success,
            contract_valid=True,
            candidate_count=3,
        )
        for index in range(count)
    )


def test_acquisition_gate_requires_fixed_apex_success_and_eight_unique_successful_parents():
    """One lucky or repeated trajectory must never open candidate snapshot acquisition."""
    fixed = {"physical_metrics": {"apex_band_success_rate": 0.125}}
    report = evaluate_candidate_acquisition_gate(fixed, _parents(8))
    assert report == {
        "eligible": True,
        "fixed_apex_success": True,
        "successful_parent_count": 8,
        "successful_parent_candidate_count": 8,
        "unique_successful_seed_count": 8,
        "unique_successful_trajectory_count": 8,
        "all_parent_contracts_valid": True,
        "minimum_independent_successful_parents": 8,
        "failed": [],
    }

    lucky = evaluate_candidate_acquisition_gate(fixed, _parents(1))
    assert not lucky["eligible"]
    assert "minimum_successful_parents" in lucky["failed"]

    repeated = list(_parents(8))
    repeated[-1] = AcquisitionParentSummary(
        seed=repeated[-2].seed,
        trajectory_hash=repeated[-2].trajectory_hash,
        success=True,
        contract_valid=True,
        candidate_count=3,
    )
    duplicate = evaluate_candidate_acquisition_gate(fixed, tuple(repeated))
    assert not duplicate["eligible"]
    assert set(duplicate["failed"]) >= {
        "unique_successful_seeds",
        "unique_successful_trajectories",
    }

    missing_candidates = list(_parents(8))
    missing_candidates[-1] = AcquisitionParentSummary(
        seed=missing_candidates[-1].seed,
        trajectory_hash=missing_candidates[-1].trajectory_hash,
        success=True,
        contract_valid=True,
        candidate_count=0,
    )
    no_coverage = evaluate_candidate_acquisition_gate(
        fixed, tuple(missing_candidates)
    )
    assert not no_coverage["eligible"]
    assert "successful_parent_candidate_coverage" in no_coverage["failed"]


def test_acquisition_gate_rejects_no_fixed_success_or_any_parent_contract_failure():
    """Stochastic success cannot override held-out failure or a corrupt timing contract."""
    no_fixed = evaluate_candidate_acquisition_gate(
        {"physical_metrics": {"apex_band_success_rate": 0.0}}, _parents(8)
    )
    assert not no_fixed["eligible"]
    assert "fixed_apex_success" in no_fixed["failed"]

    parents = list(_parents(8))
    parents[3] = AcquisitionParentSummary(
        seed=parents[3].seed,
        trajectory_hash=parents[3].trajectory_hash,
        success=True,
        contract_valid=False,
        candidate_count=3,
    )
    invalid = evaluate_candidate_acquisition_gate(
        {"physical_metrics": {"apex_band_success_rate": 0.125}}, tuple(parents)
    )
    assert not invalid["eligible"]
    assert "parent_contracts" in invalid["failed"]
    with pytest.raises(RuntimeError, match="snapshot.*contract"):
        require_candidate_acquisition_integrity(invalid)


def test_acquisition_parent_identity_is_fail_closed():
    """Empty hashes, negative counts, and boolean seeds are not auditable parents."""
    with pytest.raises(ValueError):
        AcquisitionParentSummary(
            seed=True,
            trajectory_hash="x",
            success=True,
            contract_valid=True,
            candidate_count=0,
        )


def test_provisional_continuation_label_uses_closed_outcomes_and_frozen_policy_identity():
    """A screen label must count downstream completion, not mere survival."""
    label = build_provisional_continuation_label(
        (
            {"outcome": "success", "termination_reason": "apex_band_entered"},
            {"outcome": "physical_failure", "termination_reason": "roll_limit"},
            {"outcome": "timeout", "termination_reason": "continuation_horizon"},
            {"outcome": "other_failure", "termination_reason": "missed_liftoff"},
        ),
        phase="propulsion_ascent",
        source_policy_hash="a" * 64,
        protocol_hash="b" * 64,
    )
    assert label["outcome_counts"] == {
        "success": 1,
        "physical_failure": 1,
        "timeout": 1,
        "other_failure": 1,
    }
    assert label["num_rollouts"] == 4
    assert label["num_successes"] == 1
    assert label["empirical_rate"] == 0.25
    assert label["physical_failure_rate"] == 0.25
    assert label["timeout_rate"] == 0.25
    assert label["provisional"] is True
    record = {
        "two_phase_context": {"source_phase": "propulsion_ascent"},
        "continuation_label": label,
    }
    assert validate_continuation_label(record)["valid"]
