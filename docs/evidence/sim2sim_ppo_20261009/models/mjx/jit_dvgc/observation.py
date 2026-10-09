"""Observable Actor frames, real FIFO history, and privileged critic input."""

from __future__ import annotations

from flax import struct
import jax
from jax import numpy as jp

from .constants import (
    ACTOR_FRAME_SIZE,
    ACTOR_OBSERVATION_SIZE,
    PRIVILEGED_OBSERVATION_SIZE,
)
from .model import ModelIndex


@struct.dataclass
class HistoryState:
    frames: jax.Array
    valid_count: jax.Array


@struct.dataclass
class ObservableGeometry:
    obstacle_relative_x: jax.Array
    root_height: jax.Array
    roll: jax.Array
    pitch: jax.Array
    yaw: jax.Array
    illegal_contact: jax.Array = False


def initial_history() -> HistoryState:
    return HistoryState(
        frames=jp.zeros((3, ACTOR_FRAME_SIZE), dtype=jp.float32),
        valid_count=jp.asarray(0, dtype=jp.int32),
    )


def advance_history(history: HistoryState, frame: jax.Array) -> HistoryState:
    candidate = jp.asarray(frame, dtype=jp.float32).at[-1].set(0.0)
    frames = jp.concatenate((history.frames[1:], candidate[None, :]), axis=0)
    valid_count = jp.minimum(history.valid_count + 1, jp.asarray(3, jp.int32))
    valid_mask = (jp.arange(3, dtype=jp.int32) >= 3 - valid_count).astype(jp.float32)
    frames = frames.at[:, -1].set(valid_mask)
    return HistoryState(frames=frames, valid_count=valid_count)


def actor_observation(history: HistoryState, jump_signal: jax.Array) -> jax.Array:
    observation = jp.concatenate(
        (
            history.frames.reshape((-1,)),
            jp.asarray((jump_signal,), dtype=jp.float32),
        )
    )
    if observation.shape != (ACTOR_OBSERVATION_SIZE,):
        raise ValueError(f"Actor observation shape drifted: {observation.shape}")
    return observation


def _gravity_in_body_frame(quaternion: jax.Array) -> jax.Array:
    w, x, y, z = quaternion
    return jp.asarray(
        (
            -2.0 * (x * z - w * y),
            -2.0 * (y * z + w * x),
            -(1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=jp.float32,
    )


def observable_frame(
    data,
    model_index: ModelIndex,
    geometry: ObservableGeometry,
    last_action: jax.Array,
    history_valid: jax.Array,
) -> jax.Array:
    addresses = model_index.sensor_addresses
    acceleration = data.sensordata[addresses[4] : addresses[4] + 3]
    angular_velocity = data.sensordata[addresses[5] : addresses[5] + 3]
    gravity = _gravity_in_body_frame(data.qpos[3:7])
    joint_positions = jp.asarray(
        (
            data.qpos[model_index.steering_qpos_address],
            data.qpos[model_index.hip_qpos_address],
            data.qpos[model_index.knee_qpos_address],
        )
    )
    joint_velocities = jp.asarray(
        (
            data.qvel[model_index.steering_dof_address],
            data.qvel[model_index.hip_dof_address],
            data.qvel[model_index.knee_dof_address],
        )
    )
    wheel_velocities = jp.asarray(
        (
            data.qvel[model_index.frontwheel_dof_address],
            data.qvel[model_index.rearwheel_dof_address],
        )
    )
    tail = jp.asarray(
        (
            data.qvel[model_index.root_dof_address],
            geometry.obstacle_relative_x,
            geometry.root_height,
            history_valid,
        ),
        dtype=jp.float32,
    )
    return jp.concatenate(
        (
            gravity,
            angular_velocity,
            acceleration,
            joint_positions,
            joint_velocities,
            wheel_velocities,
            jp.asarray(last_action, dtype=jp.float32),
            tail,
        )
    ).astype(jp.float32)


def privileged_observation(
    data,
    actor_obs: jax.Array,
    geometry: ObservableGeometry,
) -> jax.Array:
    privileged = jp.concatenate(
        (
            jp.asarray(actor_obs, dtype=jp.float32),
            jp.asarray(data.qpos, dtype=jp.float32),
            jp.asarray(data.qvel, dtype=jp.float32),
            jp.asarray((geometry.roll, geometry.pitch, geometry.yaw), dtype=jp.float32),
            jp.asarray(data.qvel[:3], dtype=jp.float32),
            jp.asarray((geometry.illegal_contact,), dtype=jp.float32),
        )
    )
    if privileged.shape != (PRIVILEGED_OBSERVATION_SIZE,):
        raise ValueError(
            f"privileged observation shape drifted: {privileged.shape}"
        )
    return privileged
