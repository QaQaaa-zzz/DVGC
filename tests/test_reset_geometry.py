import mujoco
import numpy as np

from dvgc.reset_geometry import GroundSupportSolver


XML = "assets/orange_bike_4kg_horizontal.xml"


def test_key_posture_is_placed_on_wheels_without_copying_root_x():
    model = mujoco.MjModel.from_xml_path(XML)
    qpos = model.key_qpos[0].copy()
    qvel = model.key_qvel[0].copy()
    qpos[0] = 2.25
    solver = GroundSupportSolver(XML)
    result = solver.solve(qpos, qvel, np.zeros(model.nu))
    assert result.accepted
    assert result.qpos[0] == 2.25
    assert result.root_z_shift_m < 0
    assert result.wheel_terrain_contacts >= 1
    assert result.body_terrain_contacts == 0
    assert result.minimum_penetration_m >= -0.002


def test_nonfinite_ground_proposal_rejects():
    model = mujoco.MjModel.from_xml_path(XML)
    qpos = model.key_qpos[0].copy()
    qpos[0] = np.nan
    result = GroundSupportSolver(XML).solve(qpos, model.key_qvel[0], np.zeros(model.nu))
    assert not result.accepted
    assert result.reason == "nonfinite"
