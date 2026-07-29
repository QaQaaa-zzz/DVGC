import pytest

from cli.select_reachability_ranked_bank import select


def rows():
    return [
        {"id": f"{kind}-{parent}-{index}", "candidate_kind": kind,
         "trajectory_parent_id": parent}
        for kind in ("canonical", "reference")
        for parent in ("p0", "p1", "p2")
        for index in range(2)
    ]


def proposals(records):
    return [{"candidate_id": row["id"], "predicted_p_next": 1.0-index / 100}
            for index, row in enumerate(records)]


def test_selection_is_parent_bounded_and_kind_balanced():
    records = rows()
    chosen = select(records, proposals(records), target=6, max_per_parent=2,
                    minimum_per_kind=2)
    parents = [parent for _, _, parent in chosen]
    assert len(chosen) == 6
    assert max(parents.count(parent) for parent in set(parents)) == 2
    assert {row["candidate_kind"] for row, _, _ in chosen} == {"canonical", "reference"}


def test_selection_fails_instead_of_implicitly_relaxing_diversity():
    records = rows()
    with pytest.raises(ValueError, match="diversity limits"):
        select(records, proposals(records), target=7, max_per_parent=2,
               minimum_per_kind=2)
