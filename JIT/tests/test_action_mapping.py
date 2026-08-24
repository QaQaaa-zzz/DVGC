from __future__ import annotations

import jax
from jax import numpy as jp
import numpy as np
import pytest

from jit_dvgc.action_mapping import ActionMapping, map_action


@pytest.fixture
def mapping() -> ActionMapping:
    return ActionMapping(
        ctrl_min=jp.array([-0.8, -5.0, -1.3, -1.5]),
        ctrl_max=jp.array([0.8, 40.0, 0.5, 2.5]),
        hip_initial=-1.2,
        base_rear_speed=12.0,
        rear_speed_delta=12.0,
        knee_target_delta=0.2,
    )


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ([0.0, 0.0, 0.0, 0.0], [0.0, 12.0, -1.2, 2.5]),
        ([1.0, 1.0, 1.0, 1.0], [0.8, 24.0, 0.5, 2.3]),
        ([-1.0, -1.0, -1.0, -1.0], [-0.8, 0.0, -1.3, 2.5]),
    ],
)
def test_action_order_piecewise_hip_and_incremental_knee(mapping, action, expected):
    ctrl = map_action(jp.asarray(action), jp.asarray(2.5), mapping)
    np.testing.assert_allclose(ctrl, expected, atol=1e-6)


def test_all_controls_are_clipped_to_authoritative_ranges(mapping):
    ctrl = map_action(jp.array([100.0, 100.0, -100.0, 100.0]), jp.array(-1.5), mapping)
    assert bool((ctrl >= mapping.ctrl_min).all())
    assert bool((ctrl <= mapping.ctrl_max).all())


def test_mapping_compiles_and_batches_without_host_conversion(mapping):
    actions = jp.array(
        [[0.0, 0.0, 0.0, 0.0], [1.0, -1.0, -1.0, 1.0]],
        dtype=jp.float32,
    )
    knees = jp.array([2.5, 1.0], dtype=jp.float32)
    mapped = jax.jit(jax.vmap(map_action, in_axes=(0, 0, None)))(actions, knees, mapping)

    assert mapped.shape == (2, 4)
    assert bool(jp.isfinite(mapped).all())
    assert float(mapped[0, 3]) == pytest.approx(2.5)
    assert float(mapped[1, 3]) == pytest.approx(0.8)
