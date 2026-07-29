from cli.train_phase_balanced_unified_rsi_pilot import acceptance, select_parent_diverse


def _records():
    return [{"id": f"{stage}-{index}", "phase_rsi_stage": stage,
             "reset_parent_id": f"{stage}:p{index}"}
            for stage in ("takeoff", "ascent", "apex", "descent", "landing")
            for index in range(4)]


def test_fixed_evaluation_is_three_parent_diverse_states_per_phase():
    selected = select_parent_diverse(_records(), 3)
    assert len(selected) == 15
    for stage in ("takeoff", "ascent", "apex", "descent", "landing"):
        assert len([row for row in selected if row["phase_rsi_stage"] == stage]) == 3


def test_promotion_requires_downstream_retention_and_final_improvement():
    before = {stage: {"final_states": 0} for stage in
              ("takeoff", "ascent", "apex", "descent", "landing")}
    before["descent"]["final_states"] = 2; before["landing"]["final_states"] = 3
    after = {stage: dict(value) for stage, value in before.items()}
    after["apex"]["final_states"] = 1
    result = acceptance({"by_stage": before, "final_states": 5},
                        {"by_stage": after, "final_states": 6, "nonfinite": 0}, True)
    assert result["promote"] is True
    after["landing"]["final_states"] = 2
    result = acceptance({"by_stage": before, "final_states": 5},
                        {"by_stage": after, "final_states": 4, "nonfinite": 0}, True)
    assert result["promote"] is False
