import pytest

from cli.descent_tube_controller import (
    failure_fuse_update,
    planned_branch_seeds,
    pointwise_seed,
)
from dvgc.certification import assert_disjoint_branch_seeds, branch_seed


def _construction(base_seed, states=139, branches=32):
    return [
        {"branch_seed": branch_seed(base_seed, state, branch)}
        for state in range(states)
        for branch in range(branches)
    ]


def test_round2_pointwise_seed_is_disjoint_from_construction_global_indices():
    construction = _construction(9_630_000)
    with pytest.raises(ValueError, match="reuses construction"):
        assert_disjoint_branch_seeds(construction, planned_branch_seeds(9_330_000, 98))
    assert_disjoint_branch_seeds(construction, planned_branch_seeds(pointwise_seed(2), 98))


def test_pointwise_seed_requires_an_explicit_reviewed_round():
    assert pointwise_seed(1) == 9_310_000
    assert pointwise_seed(2) == 200_000_000
    assert pointwise_seed(3) == 600_000_000
    with pytest.raises(ValueError, match="No independent pointwise seed"):
        pointwise_seed(4)


def test_identical_controller_failure_fuses_on_third_restart():
    state = {}
    first = RuntimeError("same deterministic error")
    signature, count = failure_fuse_update(state, "audit", first)
    assert count == 1
    state.update(failure_signature=signature, consecutive_failure_count=count)
    signature, count = failure_fuse_update(state, "audit", first)
    assert count == 2
    state.update(failure_signature=signature, consecutive_failure_count=count)
    _, count = failure_fuse_update(state, "audit", first)
    assert count == 3
    _, changed = failure_fuse_update(state, "merge", RuntimeError("different"))
    assert changed == 1


def test_post_audit_resume_waits_for_old_controller_to_exit():
    text=open("scripts/resume_descent_tube_after_current_audit.sh",encoding="utf-8").read()
    assert '"$active" != "active"' in text
    assert '"$active" != "activating"' in text
    assert 'systemctl --user start "$UNIT"' in text
    assert "Round-2 exact pointwise Tube precision" in text
