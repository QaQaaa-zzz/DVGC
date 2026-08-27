from __future__ import annotations

import pytest


def test_global_seed_splits_are_disjoint_complete_and_deterministic():
    from jit_dvgc.continuation_labels import assign_global_seed_splits

    observed = [1000001, 1000002, 1000003, 1000004, 1000005, 1000006, 1000007, 1000008]
    first = assign_global_seed_splits(observed)
    second = assign_global_seed_splits(reversed(observed))
    assert first == second
    assert first == {
        1000001: "train",
        1000002: "train",
        1000003: "train",
        1000004: "train",
        1000005: "train",
        1000006: "validation",
        1000007: "test",
        1000008: "test",
    }


def test_global_seed_split_keeps_same_seed_across_source_checkpoints():
    from jit_dvgc.continuation_labels import assign_global_seed_splits

    source_checkpoints = ("transition_4988928", "transition_7987200", "transition_9977856")
    seed_splits = assign_global_seed_splits(range(1000001, 1000009))
    parent_splits = {
        f"{checkpoint}__{seed}": seed_splits[seed]
        for checkpoint in source_checkpoints
        for seed in range(1000001, 1000009)
    }
    for seed in range(1000001, 1000009):
        assert {
            parent_splits[f"{checkpoint}__{seed}"] for checkpoint in source_checkpoints
        } == {seed_splits[seed]}


def test_global_seed_splits_reject_overlap_and_incomplete_coverage():
    from jit_dvgc.continuation_labels import assign_global_seed_splits

    with pytest.raises(ValueError, match="must be disjoint"):
        assign_global_seed_splits(
            [1, 2, 3],
            train_seeds=(1,),
            validation_seeds=(2,),
            test_seeds=(2, 3),
        )
    with pytest.raises(ValueError, match="exactly cover observed"):
        assign_global_seed_splits(
            [1, 2, 3],
            train_seeds=(1,),
            validation_seeds=(2,),
            test_seeds=(4,),
        )


def test_branch_key_is_stable_and_branch_specific():
    from jit_dvgc.continuation_labels import derive_branch_key

    seed0, _ = derive_branch_key(protocol_seed=7, candidate_id="abc", branch_index=0)
    seed0_again, _ = derive_branch_key(protocol_seed=7, candidate_id="abc", branch_index=0)
    seed1, _ = derive_branch_key(protocol_seed=7, candidate_id="abc", branch_index=1)
    assert seed0 == seed0_again
    assert seed0 != seed1
