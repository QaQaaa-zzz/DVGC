from cli.freeze_ranked_candidate_controllers import assign_controller


def test_assignment_uses_best_rate_and_registry_tie_break():
    label = {"branches": [
        {"controller_id": "a", "success": True},
        {"controller_id": "a", "success": False},
        {"controller_id": "b", "success": True},
        {"controller_id": "b", "success": True},
    ]}
    chosen, successes = assign_controller(label, ["a", "b"])
    assert chosen == "b"
    assert successes == {"a": 1, "b": 2}

    label["branches"][1]["success"] = True
    chosen, _ = assign_controller(label, ["a", "b"])
    assert chosen == "a"
