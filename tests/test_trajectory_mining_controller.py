from pathlib import Path


def test_new_route_is_bounded_and_does_not_repeat_old_ppo():
    source=Path("cli/trajectory_mining_controller.py").read_text()
    assert "trajectory_mining" in source and "roll_controllability" in source
    assert "single_roll_targeted_ppo" in source
    assert source.count("25600")==1
    assert "descent_acquisition_max_rounds" not in source


def test_persistent_start_script_uses_new_unit_and_active_pointer():
    source=Path("scripts/start_trajectory_mining_controller.sh").read_text()
    assert "dvgc-trajectory-mining-controller" in source
    assert "ACTIVE_PIPELINE.json" in source
