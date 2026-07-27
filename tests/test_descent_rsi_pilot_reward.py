import jax.numpy as jp

from dvgc.config import load_config
from dvgc.rewards import compute_descent_rsi_pilot_reward


def _reward(**updates):
    cfg = load_config("configs/default.json")
    values = dict(
        cfg=cfg, roll=jp.asarray(0.0), pitch=jp.asarray(0.0),
        gyro=jp.zeros(3), vz=jp.asarray(-0.5), action=jp.zeros(4),
        previous_action=jp.zeros(4), hard_failure=jp.asarray(False),
    )
    values.update(updates)
    return compute_descent_rsi_pilot_reward(**values)


def test_descent_rsi_reward_is_bounded_and_matcher_free():
    nominal = _reward()
    unstable = _reward(
        roll=jp.deg2rad(40.0), pitch=jp.deg2rad(45.0),
        gyro=jp.asarray([5.0, 5.0, 5.0]), vz=jp.asarray(-2.5),
        action=jp.ones(4),
    )
    failed = _reward(hard_failure=jp.asarray(True))
    assert float(nominal["reward"]) > float(unstable["reward"])
    assert float(failed["reward"]) < float(nominal["reward"]) - 1.0
    assert float(nominal["shaping"]) <= 0.2
    assert "distance" not in nominal and "chain" not in nominal


def test_descent_rsi_reward_terms_are_finite():
    assert all(bool(jp.isfinite(value)) for value in _reward().values())
