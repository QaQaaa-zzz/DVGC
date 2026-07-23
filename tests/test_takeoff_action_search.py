from pathlib import Path


def test_action_search_is_policy_free_and_checks_real_wheel_clearance():
    text = Path("cli/search_takeoff_actions.py").read_text()
    assert '"policy_used": False' in text
    assert "GroundSupportSolver" in text
    assert "actuator_force" in text
    assert "next_stage_entry" in text
    assert "hip_full_knee_half" in text
    assert "reference_time_aligned" in text
    assert "row.get(\"reference_index\"" in text
