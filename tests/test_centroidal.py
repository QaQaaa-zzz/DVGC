import mujoco
import numpy as np

from dvgc.centroidal import replay_centroidal
from cli.audit_apex_centroidal_contact import _terminal_calibration


def test_explicit_centroidal_momentum_matches_mujoco_subtree():
    model = mujoco.MjModel.from_xml_path(
        "assets/orange_bike_4kg_horizontal.xml"
    )
    rng = np.random.default_rng(17)
    qpos = model.qpos0.copy()
    qvel = rng.normal(scale=.25, size=model.nv)
    result = replay_centroidal(model, qpos, qvel, np.zeros(model.nu))
    np.testing.assert_allclose(
        result["centroidal_angular_momentum"],
        result["mujoco_subtree_angular_momentum"],
        atol=1e-10,
    )
    assert result["angular_momentum_crosscheck_linf"] < 1e-10
    assert len(result["body_contributions"]) == model.nbody - 1


def test_centroidal_replay_does_not_modify_input():
    model = mujoco.MjModel.from_xml_path(
        "assets/orange_bike_4kg_horizontal.xml"
    )
    qpos = model.qpos0.copy()
    qvel = np.zeros(model.nv)
    qpos_before = qpos.copy()
    replay_centroidal(model, qpos, qvel)
    np.testing.assert_array_equal(qpos, qpos_before)


def test_terminal_calibration_uses_data_and_physical_floors():
    values = np.zeros((3, 10))
    center, scale, threshold = _terminal_calibration(values)
    np.testing.assert_allclose(center, 0.)
    assert np.all(scale > 0.)
    assert threshold >= 1.
