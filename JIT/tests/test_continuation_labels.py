from __future__ import annotations

import pytest


def test_parent_splits_are_disjoint_complete_and_deterministic():
    from jit_dvgc.continuation_labels import assign_parent_splits

    groups = [f"parent-{index:02d}" for index in range(24)]
    first = assign_parent_splits(groups, split_seed=820301)
    second = assign_parent_splits(reversed(groups), split_seed=820301)
    assert first == second
    assert set(first) == set(groups)
    assert set(first.values()) == {"train", "validation", "test"}
    counts = {name: sum(value == name for value in first.values()) for name in set(first.values())}
    assert counts == {"train": 16, "validation": 4, "test": 4}


def test_parent_splits_require_three_groups():
    from jit_dvgc.continuation_labels import assign_parent_splits

    with pytest.raises(ValueError, match="at least three parent groups"):
        assign_parent_splits(["a", "b"])


def test_branch_key_is_stable_and_branch_specific():
    from jit_dvgc.continuation_labels import derive_branch_key

    seed0, _ = derive_branch_key(protocol_seed=7, candidate_id="abc", branch_index=0)
    seed0_again, _ = derive_branch_key(protocol_seed=7, candidate_id="abc", branch_index=0)
    seed1, _ = derive_branch_key(protocol_seed=7, candidate_id="abc", branch_index=1)
    assert seed0 == seed0_again
    assert seed0 != seed1
