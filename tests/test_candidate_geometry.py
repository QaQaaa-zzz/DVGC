from pathlib import Path

import numpy as np

from dvgc.candidate_geometry import TerrainClearanceSolver


XML = "assets/orange_bike_4kg_horizontal.xml"


def _pose(solver, *, x, z):
    qpos = solver.model.qpos0.copy()
    qpos[0], qpos[2] = x, z
    return qpos


def test_already_legal_state_is_not_lifted():
    solver = TerrainClearanceSolver(XML, margin=0.002, max_correction=0.20)
    result = solver.solve(_pose(solver, x=0.0, z=0.25), com_z_tolerance=0.20)
    assert result.accepted and result.reason == "already_clear"
    assert result.root_z_shift == 0.0


def test_penetration_gets_minimal_correction_or_is_rejected():
    solver = TerrainClearanceSolver(XML, margin=0.002, max_correction=0.20, tolerance=1e-5)
    qpos = _pose(solver, x=4.05, z=0.25)
    initial = solver._measure(qpos, None, None)
    assert initial["contacts"] > 0
    result = solver.solve(qpos, com_z_tolerance=0.20)
    assert result.accepted and result.root_z_shift > 0
    assert result.robot_terrain_contacts == 0
    assert result.wheel_clearance >= solver.margin
    assert result.nonwheel_clearance >= solver.margin
    below = result.qpos.copy(); below[solver.root_z_index] -= 2 * solver.tolerance
    assert not (solver._measure(below, None, None)["contacts"] == 0
                and solver._measure(below, None, None)["wheel"] >= solver.margin
                and solver._measure(below, None, None)["nonwheel"] >= solver.margin)
    limited = TerrainClearanceSolver(XML, margin=0.002, max_correction=0.01)
    rejected = limited.solve(_pose(limited, x=4.05, z=0.25), com_z_tolerance=0.20)
    assert not rejected.accepted and rejected.reason == "max_correction_exceeded"


def test_real_terrain_height_changes_placement(tmp_path):
    source = Path(XML).read_text(encoding="utf-8")
    assert 'material="rampplane" pos="5.6 0 0"' in source
    raised = source.replace(
        'material="rampplane" pos="5.6 0 0"',
        'material="rampplane" pos="5.6 0 0.05"',
    )
    (tmp_path / "meshes").symlink_to(Path(XML).resolve().parent / "meshes", target_is_directory=True)
    raised_xml = tmp_path / "raised.xml"; raised_xml.write_text(raised, encoding="utf-8")
    base = TerrainClearanceSolver(XML, margin=0.002, max_correction=0.30)
    high = TerrainClearanceSolver(str(raised_xml), margin=0.002, max_correction=0.30)
    base_result = base.solve(_pose(base, x=4.05, z=0.25), com_z_tolerance=0.30)
    high_result = high.solve(_pose(high, x=4.05, z=0.25), com_z_tolerance=0.30)
    assert base_result.accepted and high_result.accepted
    assert high_result.root_z_shift > base_result.root_z_shift


def test_solver_is_exactly_reproducible_and_clears_body_and_wheels():
    solver = TerrainClearanceSolver(XML, margin=0.002, max_correction=0.20)
    qpos = _pose(solver, x=4.05, z=0.25)
    one = solver.solve(qpos, com_z_tolerance=0.20)
    two = solver.solve(qpos, com_z_tolerance=0.20)
    assert one.accepted == two.accepted and one.reason == two.reason
    assert one.root_z_shift == two.root_z_shift
    np.testing.assert_array_equal(one.qpos, two.qpos)
    np.testing.assert_array_equal(one.corrected_com, two.corrected_com)
    assert one.robot_terrain_contacts == 0
    assert one.wheel_clearance >= 0.002 and one.nonwheel_clearance >= 0.002
