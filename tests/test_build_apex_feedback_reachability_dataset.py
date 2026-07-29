from cli.build_apex_feedback_reachability_dataset import first_stable_tick


def test_first_stable_tick_uses_event_not_final_termination():
    outcome = {"trace": [
        {"tick": 1, "stable_physical_descent": False},
        {"tick": 2, "stable_physical_descent": True},
    ], "termination_reason": "roll_limit"}
    assert first_stable_tick(outcome) == 2
    assert first_stable_tick({"trace": []}) is None
