from types import SimpleNamespace

import mujoco
import numpy as np

from dvgc.reference_joints import apply_stage_joint_state, stage_joint_state


XML = "assets/orange_bike_4kg_horizontal.xml"


def test_takeoff_uses_key_joints_without_copying_key_root():
    model = mujoco.MjModel.from_xml_path(XML)
    row = SimpleNamespace(
        hip_position=-0.9, knee_position=1.7,
        hip_velocity=8.0, knee_velocity=-12.0,
    )
    joints = stage_joint_state(model, row, "takeoff")
    assert joints.source == "xml_key:initial_state"
    assert np.allclose(
        [joints.hip, joints.knee, joints.hip_velocity, joints.knee_velocity],
        [-1.2, 2.5, 0.0, 0.0],
    )
    qpos = np.arange(model.nq, dtype=float) + 10
    qvel = np.arange(model.nv, dtype=float) + 20
    changed_qpos, changed_qvel = apply_stage_joint_state(model, qpos, qvel, joints)
    root_joint = int(model.joint("floating_base_joint").id)
    root_qpos = int(model.jnt_qposadr[root_joint])
    root_qvel = int(model.jnt_dofadr[root_joint])
    assert np.array_equal(changed_qpos[root_qpos:root_qpos + 7], qpos[root_qpos:root_qpos + 7])
    assert np.array_equal(changed_qvel[root_qvel:root_qvel + 6], qvel[root_qvel:root_qvel + 6])


def test_non_takeoff_uses_corresponding_reference_joint_state():
    model = mujoco.MjModel.from_xml_path(XML)
    row = SimpleNamespace(
        hip_position=-1.05, knee_position=1.82,
        hip_velocity=3.25, knee_velocity=-7.5,
    )
    for phase in ("approach", "flight", "landing"):
        joints = stage_joint_state(model, row, phase)
        assert joints.source == "reference_trajectory"
        assert np.allclose(
            [joints.hip, joints.knee, joints.hip_velocity, joints.knee_velocity],
            [-1.05, 1.82, 3.25, -7.5],
        )
