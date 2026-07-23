from pathlib import Path

from cli.stage_label_pilot import evenly,terminal_is_physical_failure

def test_even_selection_is_deterministic_and_covers_endpoints():
 rows=list(range(20));assert evenly(rows,6)==evenly(rows,6)
 assert evenly(rows,6)[0]==0 and evenly(rows,6)[-1]==19

def test_successful_recovery_termination_is_not_physical_failure():
 assert not terminal_is_physical_failure(True,'recovery')
 assert not terminal_is_physical_failure(True,'next_stage_entry')
 assert terminal_is_physical_failure(True,'pitch_limit')

def test_takeoff_entry_uses_geometry_derived_dual_wheel_event():
 text=Path("cli/stage_label_pilot.py").read_text()
 assert 'state.metrics["event/dual_wheel_liftoff"]' in text
 assert "nominal_base_z_ground)+float(env._config.imu_airborne_height_margin)" not in text
