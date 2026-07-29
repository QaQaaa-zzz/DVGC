from cli.build_stage_audit_funnel import survivor_ids


def test_only_all_branch_positive_states_survive():
    reports = [{"labels": [
        {"candidate_id": "safe", "label": "positive", "s": 8, "n": 8},
        {"candidate_id": "almost", "label": "boundary", "s": 7, "n": 8},
        {"candidate_id": "inconsistent", "label": "positive", "s": 7, "n": 8},
    ]}]
    assert survivor_ids(reports) == {"safe"}
