import math

import pytest

from dvgc.certification import (
    assert_disjoint_branch_seeds,
    branch_evidence,
    branch_seed,
    summarize_branches,
)


def outcome(*, chain, final, terminated, truncated, steps=12):
    return {
        "chain": chain,
        "final": final,
        "terminated": terminated,
        "truncated": truncated,
        "steps": steps,
    }


def test_branch_terminal_causes_and_summary_are_separate():
    raw = [
        outcome(chain=1, final=1, terminated=1, truncated=0),
        outcome(chain=1, final=0, terminated=1, truncated=0),
        outcome(chain=1, final=0, terminated=0, truncated=1),
        outcome(chain=0, final=0, terminated=0, truncated=0),
    ]
    rows = [
        branch_evidence(
            branch_index=i,
            seed=branch_seed(100, 0, i),
            seed_namespace="build:landing",
            dynamics_variant="nominal",
            outcome=value,
        )
        for i, value in enumerate(raw)
    ]
    assert [row["terminal_cause"] for row in rows] == [
        "final_recovery",
        "physical_failure",
        "timeout",
        "horizon_exhausted",
    ]
    summary = summarize_branches(rows)
    assert summary["branches"] == 4
    assert summary["physical_failures"] == 1
    assert summary["timeouts"] == 1
    assert summary["horizon_exhaustions"] == 1
    assert summary["false_progress_rate"] == 0.5
    assert summary["missed_success_rate"] == 0.0


def test_branch_evidence_rejects_overlapping_terminal_masks():
    with pytest.raises(ValueError, match="both physically terminated"):
        branch_evidence(
            branch_index=0,
            seed=0,
            seed_namespace="build:landing",
            dynamics_variant="nominal",
            outcome=outcome(chain=0, final=0, terminated=1, truncated=1),
        )


def test_audit_branch_seeds_must_be_disjoint():
    construction = [{"branch_seed": 10}, {"branch_seed": 11}]
    assert_disjoint_branch_seeds(construction, [20, 21])
    with pytest.raises(ValueError, match="reuses construction"):
        assert_disjoint_branch_seeds(construction, [11, 20])


def test_empty_summary_uses_nan_rates():
    summary = summarize_branches([])
    assert summary["branches"] == 0
    assert math.isnan(summary["physical_failure_rate"])
