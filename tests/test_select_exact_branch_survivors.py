from cli.select_exact_branch_survivors import exact_survivor_ids


def test_only_exact_all_success_positive_states_advance():
    labels = [
        {"candidate_id": "safe", "label": "positive", "s": 4, "n": 4},
        {"candidate_id": "almost", "label": "positive", "s": 3, "n": 4},
        {"candidate_id": "wrong-level", "label": "positive", "s": 8, "n": 8},
        {"candidate_id": "wrong-label", "label": "unknown", "s": 4, "n": 4},
    ]
    assert exact_survivor_ids(labels, 4) == {"safe"}
