from __future__ import annotations

import jax.numpy as jp
import pytest

import dvgc.env as env_module


@pytest.mark.parametrize(
    "active",
    (
        "prohibited_contact",
        "invalid_wheel_contact",
        "roll_limit",
        "pitch_limit",
        "backward_motion",
        "platform_back_edge_exit",
        "takeoff_task_failure",
        "nonfinite",
    ),
)
def test_each_retained_global_failure_remains_hard(active):
    assert hasattr(env_module, "_hard_failure_flags")
    flags = {
        name: jp.asarray(name == active)
        for name in (
            "prohibited_contact",
            "invalid_wheel_contact",
            "roll_limit",
            "pitch_limit",
            "backward_motion",
            "platform_back_edge_exit",
            "takeoff_task_failure",
            "nonfinite",
        )
    }

    assert bool(env_module._hard_failure_flags(**flags)) is True


def test_prelaunch_airborne_is_not_a_hard_failure_input():
    assert hasattr(env_module, "_hard_failure_flags")
    assert "prelaunch_airborne" not in env_module._hard_failure_flags.__annotations__
