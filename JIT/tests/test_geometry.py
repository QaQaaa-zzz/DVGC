from __future__ import annotations

from typing import NamedTuple

import jax
from jax import numpy as jp
import mujoco
import numpy as np
import pytest

from jit_dvgc.config import load_config
from jit_dvgc.geometry import (
    GEOM_BOX,
    ContactSignals,
    GeometrySignals,
    build_geometry_contract,
    collision_support_bounds,
    extract_geometry,
    full_structure_metrics,
)
from jit_dvgc.model import load_host_model


class WarpLikeData(NamedTuple):
    qpos: jax.Array
    qvel: jax.Array
    geom_xpos: jax.Array
    geom_xmat: jax.Array


def test_runtime_geometry_contract_contains_no_wheel_support_booleans():
    forbidden = {"front_wheel_support", "rear_wheel_support"}
    assert forbidden.isdisjoint(ContactSignals.__dataclass_fields__)
    assert forbidden.isdisjoint(GeometrySignals.__dataclass_fields__)


def test_box_support_bounds_are_hand_checkable():
    bounds = collision_support_bounds(
        jp.array([[1.0, 0.0, 2.0]]),
        jp.eye(3)[None, :, :],
        jp.array([GEOM_BOX]),
        jp.array([[0.2, 0.3, 0.4]]),
    )

    np.testing.assert_allclose(bounds.min_x, [0.8], atol=1e-6)
    np.testing.assert_allclose(bounds.max_x, [1.2], atol=1e-6)
    np.testing.assert_allclose(bounds.min_z, [1.6], atol=1e-6)
    np.testing.assert_allclose(bounds.max_z, [2.4], atol=1e-6)


def test_relative_x_uses_frontmost_collision_structure_and_full_clearance():
    positions = jp.array([[3.0, 0.0, 0.5], [3.4, 0.0, 0.5]])
    rotations = jp.broadcast_to(jp.eye(3), (2, 3, 3))
    types = jp.array([GEOM_BOX, GEOM_BOX])
    sizes = jp.array([[0.1, 0.1, 0.1], [0.2, 0.1, 0.1]])

    signals = full_structure_metrics(
        positions,
        rotations,
        types,
        sizes,
        obstacle_front_x=3.6,
        obstacle_top_z=0.16,
    )

    assert float(signals.robot_frontmost_x) == pytest.approx(3.6, abs=1e-6)
    assert float(signals.obstacle_relative_x) == pytest.approx(0.0, abs=1e-6)
    assert float(signals.full_structure_clearance) == pytest.approx(0.24, abs=1e-6)


def test_support_metric_compiles_and_batches():
    positions = jp.zeros((8, 2, 3)).at[:, 1, 0].set(1.0)
    rotations = jp.broadcast_to(jp.eye(3), (8, 2, 3, 3))
    types = jp.broadcast_to(jp.array([GEOM_BOX, GEOM_BOX]), (8, 2))
    sizes = jp.broadcast_to(jp.array([[0.1, 0.1, 0.1], [0.2, 0.1, 0.1]]), (8, 2, 3))
    calculate = jax.jit(
        jax.vmap(
            lambda p, r, t, s: full_structure_metrics(
                p, r, t, s, obstacle_front_x=3.6, obstacle_top_z=0.16
            )
        )
    )

    metrics = calculate(positions, rotations, types, sizes)
    assert metrics.obstacle_relative_x.shape == (8,)
    assert bool(jp.isfinite(metrics.full_structure_clearance).all())


def test_authoritative_manifest_covers_every_collision_robot_geom(jit_root):
    config = load_config(jit_root / "configs" / "phase_u_smoke.json")
    model = load_host_model(config).mj_model
    contract = build_geometry_contract(model)

    assert set(contract.robot_geom_names) == {
        "base_collision",
        "rearwheel_collision",
        "steer_collision",
        "frontwheel_collision",
        "downarm_collision",
        "knee_motor_collision",
        "uparm_collision",
    }
    assert contract.obstacle_front_x == pytest.approx(3.6)
    assert contract.obstacle_back_x == pytest.approx(7.6)
    assert contract.obstacle_top_z == pytest.approx(0.16)


def test_training_geometry_does_not_require_a_contact_array(jit_root):
    config = load_config(jit_root / "configs" / "phase_u_smoke.json")
    bundle = load_host_model(config)
    host_data = mujoco.MjData(bundle.mj_model)
    host_data.qpos[:] = bundle.mj_model.key_qpos[bundle.model_index.keyframe_id]
    host_data.qvel[:] = bundle.mj_model.key_qvel[bundle.model_index.keyframe_id]
    mujoco.mj_forward(bundle.mj_model, host_data)
    warp_like = WarpLikeData(
        qpos=jp.asarray(host_data.qpos),
        qvel=jp.asarray(host_data.qvel),
        geom_xpos=jp.asarray(host_data.geom_xpos),
        geom_xmat=jp.asarray(host_data.geom_xmat),
    )
    contract = build_geometry_contract(bundle.mj_model)

    signals = jax.jit(lambda data: extract_geometry(data, contract))(warp_like)

    assert bool(jp.isfinite(signals.obstacle_relative_x))
    assert bool(jp.isfinite(signals.structure_clearance))
    assert not bool(signals.prohibited_contact)
