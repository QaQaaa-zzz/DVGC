from cli.select_ood_boundary_candidates import select_boundary


def test_boundary_selection_is_nearest_and_root_parent_distinct():
    records = [{"id": name} for name in ("a", "b", "c")]
    scores = [
        {"candidate_id": "a", "parent": "p0", "unseen_parent": True,
         "normalized_training_distance": 3.0},
        {"candidate_id": "b", "parent": "p0", "unseen_parent": True,
         "normalized_training_distance": 2.0},
        {"candidate_id": "c", "parent": "p1", "unseen_parent": True,
         "normalized_training_distance": 4.0},
    ]
    chosen = select_boundary(records, scores, 2)
    assert [row["id"] for row, _ in chosen] == ["b", "c"]
