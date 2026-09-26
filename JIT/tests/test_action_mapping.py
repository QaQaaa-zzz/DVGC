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
        knee_initial=2.5,
        base_rear_speed=12.0,
        rear_speed_delta=12.0,
        joint_target_semantics="keyframe_centered_absolute",
        knee_target_delta=0.0,
    )


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ([0.0, 0.0, 0.0, 0.0], [0.0, 12.0, -1.2, 2.5]),
        ([1.0, 1.0, 1.0, 1.0], [0.8, 24.0, 0.5, 2.5]),
        ([-1.0, -1.0, -1.0, -1.0], [-0.8, 0.0, -1.3, -1.5]),
        ([0.0, 0.0, -0.5, -0.5], [0.0, 12.0, -1.25, 0.5]),
    ],
)
def test_action_order_uses_piecewise_absolute_hip_and_knee(mapping, action, expected):
    ctrl = map_action(jp.asarray(action), jp.asarray(-0.25), mapping)
    np.testing.assert_allclose(ctrl, expected, atol=1e-6)


def test_absolute_knee_target_does_not_depend_on_live_knee_position(mapping):
    action = jp.array([0.0, 0.0, 0.0, -0.5])
    from_lower_live_position = map_action(action, jp.asarray(-1.5), mapping)
    from_upper_live_position = map_action(action, jp.asarray(2.5), mapping)

    assert float(from_lower_live_position[3]) == pytest.approx(0.5)
    assert float(from_upper_live_position[3]) == pytest.approx(0.5)


def test_retained_v2_incremental_knee_semantics_remain_truthful(mapping):
    legacy = mapping.replace(
        joint_target_semantics="incremental_knee",
        knee_target_delta=0.2,
    )

    positive = map_action(
        jp.array([0.0, 0.0, 0.0, 1.0]), jp.asarray(1.0), legacy
    )
    negative = map_action(
        jp.array([0.0, 0.0, 0.0, -1.0]), jp.asarray(1.0), legacy
    )

    assert float(positive[3]) == pytest.approx(0.8)
    assert float(negative[3]) == pytest.approx(1.2)


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
    assert float(mapped[1, 3]) == pytest.approx(2.5)
