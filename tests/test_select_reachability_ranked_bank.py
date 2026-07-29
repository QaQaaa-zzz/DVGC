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


def test_selection_excludes_out_of_support_model_extrapolation():
    records = rows()
    ranked = proposals(records)
    ranked[0]["acquisition_eligible"] = False
    chosen = select(records, ranked, target=5, max_per_parent=2,
                    minimum_per_kind=2)
    assert records[0]["id"] not in {row["id"] for row, _, _ in chosen}


def test_generated_seed_suffixes_do_not_create_fake_parent_diversity():
    records = [
        {"id": f"x{i}", "candidate_kind": "dynamic",
         "trajectory_parent_id": f"root:seed:{i}", "source_parent_id": "root"}
        for i in range(3)
    ]
    with pytest.raises(ValueError, match="diversity limits"):
        select(records, proposals(records), target=2, max_per_parent=1,
               minimum_per_kind=0)
