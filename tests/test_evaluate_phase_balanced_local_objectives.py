from cli.evaluate_phase_balanced_local_objectives import fixed_parent_diverse


def test_local_objective_probe_is_parent_diverse_and_excludes_landing():
    rows = [
        {"id": f"{stage}-{index}", "phase_rsi_stage": stage,
         "reset_parent_id": f"{stage}:p{index}"}
        for stage in ("takeoff", "ascent", "apex", "descent", "landing")
        for index in range(4)
    ]
    selected = fixed_parent_diverse(rows, 3)
    assert len(selected) == 12
    assert {row["phase_rsi_stage"] for row in selected} == {
        "takeoff", "ascent", "apex", "descent"
    }
    assert len({row["reset_parent_id"] for row in selected}) == 12
