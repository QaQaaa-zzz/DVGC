from __future__ import annotations

from typing import NamedTuple

import jax
from jax import numpy as jp
import numpy as np

from jit_dvgc.constants import ACTOR_FRAME_FIELDS
from jit_dvgc.model import ModelIndex
from jit_dvgc.observation import (
    ObservableGeometry,
    actor_observation,
    advance_history,
    initial_history,
    observable_frame,
    privileged_observation,
)


class FakeData(NamedTuple):
    qpos: jax.Array
    qvel: jax.Array
    sensordata: jax.Array


def _model_index() -> ModelIndex:
    return ModelIndex(
        keyframe_id=0,
        root_qpos_address=0,
        root_dof_address=0,
        rearwheel_qpos_address=7,
        rearwheel_dof_address=6,
        steering_qpos_address=8,
        steering_dof_address=7,
        frontwheel_qpos_address=9,
        frontwheel_dof_address=8,
        hip_qpos_address=10,
        hip_dof_address=9,
        knee_qpos_address=11,
        knee_dof_address=10,
        floor_geom_id=0,
        obstacle_geom_id=1,
        frontwheel_geom_id=15,
        rearwheel_geom_id=9,
        sensor_addresses=(0, 1, 2, 3, 4, 7, 10, 14, 17, 20, 23, 24, 25, 26),
    )


def _geometry() -> ObservableGeometry:
    return ObservableGeometry(
        obstacle_relative_x=jp.array(0.7),
        root_height=jp.array(0.42),
        roll=jp.array(0.01),
        pitch=jp.array(-0.02),
        yaw=jp.array(0.03),
        illegal_contact=jp.array(False),
    )


def test_fifo_contains_three_consecutive_real_25_value_frames():
    history = initial_history()
    for value in (1.0, 2.0, 3.0, 4.0):
        history = advance_history(history, jp.full((25,), value))

    frames = history.frames
    assert frames.shape == (3, 25)
    np.testing.assert_array_equal(frames[:, 0], [2.0, 3.0, 4.0])
    np.testing.assert_array_equal(frames[:, -1], [1.0, 1.0, 1.0])
    assert int(history.valid_count) == 3


def test_jump_signal_is_appended_once_outside_history():
    history = advance_history(initial_history(), jp.zeros(25))
    off = actor_observation(history, jp.array(False))
    on = actor_observation(history, jp.array(True))

    assert off.shape == (76,)
    np.testing.assert_array_equal(off[:-1], on[:-1])
    assert float(off[-1]) == 0.0
    assert float(on[-1]) == 1.0
    assert int(np.count_nonzero(np.asarray(on))) == 2  # history-valid + signal


def test_reset_history_is_empty_but_current_task_signal_is_visible():
    history = initial_history()
    natural = actor_observation(history, jp.array(False))
    airborne = actor_observation(history, jp.array(True))
    np.testing.assert_array_equal(natural, np.zeros(76))
    np.testing.assert_array_equal(airborne[:-1], np.zeros(75))
    assert float(airborne[-1]) == 1.0


def test_history_valid_mask_fills_only_after_real_control_ticks():
    history = advance_history(initial_history(), jp.full((25,), 7.0))
    np.testing.assert_array_equal(history.frames[:, -1], [0.0, 0.0, 1.0])
    history = advance_history(history, jp.full((25,), 8.0))
    np.testing.assert_array_equal(history.frames[:, -1], [0.0, 1.0, 1.0])


def test_observable_frame_has_exact_25_value_order_and_compiles():
    qpos = jp.array([1.5, 0.0, 0.42, 1.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, -1.2, 2.5])
    qvel = jp.arange(11, dtype=jp.float32)
    sensor = jp.arange(31, dtype=jp.float32)
    data = FakeData(qpos=qpos, qvel=qvel, sensordata=sensor)
    last_action = jp.array([0.1, 0.2, 0.3, 0.4])
    build = jax.jit(
        lambda d: observable_frame(
            d, _model_index(), _geometry(), last_action, jp.array(True)
        )
    )

    frame = build(data)
    assert frame.shape == (25,)
    np.testing.assert_allclose(frame[:3], [0.0, 0.0, -1.0], atol=1e-6)
    np.testing.assert_allclose(frame[17:21], last_action, atol=1e-6)
    np.testing.assert_allclose(frame[21:25], [0.0, 0.7, 0.42, 1.0])


def test_actor_contract_excludes_support_outcomes_and_privileged_identifiers():
    forbidden = {
        "front_wheel_support",
        "rear_wheel_support",
        "estimated_structure_clearance",
        "reward",
        "success",
        "terminated",
        "truncated",
        "end_code",
        "reference_index",
        "policy_hash",
        "trajectory_id",
    }
    assert forbidden.isdisjoint(ACTOR_FRAME_FIELDS)


def test_privileged_observation_has_fixed_106_value_shape():
    data = FakeData(
        qpos=jp.array([1.5, 0.0, 0.42, 1.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, -1.2, 2.5]),
        qvel=jp.arange(11, dtype=jp.float32),
        sensordata=jp.zeros(31),
    )
    history = advance_history(initial_history(), jp.zeros(25))
    actor = actor_observation(history, jp.array(True))
    privileged = privileged_observation(data, actor, _geometry())
    assert privileged.shape == (106,)
    assert float(privileged[75]) == 1.0
    assert bool(jp.isfinite(privileged).all())
