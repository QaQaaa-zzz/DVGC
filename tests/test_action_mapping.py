import numpy as np
import pytest

from dvgc.action_mapping import knee_position_target


def _target(position: float, action: float) -> float:
    _, target, _ = knee_position_target(
        position,
        action,
        target_delta=0.20,
        knee_min=-1.5,
        knee_max=2.5,
        xp=np,
    )
    return float(target)


def test_positive_action_is_not_dead_at_xml_upper_limit():
    assert _target(2.5, 0.25) == 2.45
    assert _target(2.5, 0.50) == 2.40
    assert _target(2.5, 1.00) == 2.30


def test_zero_holds_and_negative_extends_after_flexion():
    assert _target(2.2, 0.0) == 2.2
    assert _target(2.2, -0.5) == pytest.approx(2.3)
    assert _target(2.2, -1.0) == pytest.approx(2.4)


def test_real_joint_limits_still_clip_targets():
    assert _target(-1.45, 1.0) == -1.5
    assert _target(2.45, -1.0) == 2.5
