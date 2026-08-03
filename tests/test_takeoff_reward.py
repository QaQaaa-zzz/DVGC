from __future__ import annotations

import inspect
import jax.numpy as jp

from dvgc.config import default_config
from dvgc.rewards import compute_takeoff_reward_profile


def _failure_signals():
    return {
        "front_tire_vz": jp.asarray(1.0),
        "rear_tire_vz": jp.asarray(1.0),
        "front_tire_bottom_z": jp.asarray(0.2),
        "rear_tire_bottom_z": jp.asarray(0.2),
        "min_tire_bottom_z": jp.asarray(0.2),
        "positive_pitch_bad": jp.asarray(True),
        "knee_vel": jp.asarray(1.0),
        "pitch_signed_deg": jp.asarray(30.0),
        "wheelie_detected": jp.asarray(True),
        "wheel_height_diff": jp.asarray(0.0),
        "positive_pitch_hard": jp.asarray(True),
        "dual_wheel_liftoff": jp.asarray(True),
        "frontmost_x": jp.asarray(10.0),
        "liftoff_deadline_x": jp.asarray(2.0),
        "clearance_deadline_x": jp.asarray(2.5),
        "wheel_clearance_ready": jp.asarray(False),
    }


def _profile(*, jump_latched, positive_pitch_count=0, wheelie_count=0):
    assert "jump_latched" in inspect.signature(
        compute_takeoff_reward_profile
    ).parameters
    return compute_takeoff_reward_profile(
        cfg=default_config(),
        signals=_failure_signals(),
        action=jp.zeros(4),
        phase0=jp.asarray(1),
        phase1=jp.asarray(1),
        jump_latched=jp.asarray(jump_latched),
        dual_wheel_liftoff_seen=jp.asarray(False),
        positive_pitch_count=jp.asarray(positive_pitch_count),
        wheelie_count=jp.asarray(wheelie_count),
    )


def test_takeoff_counters_liftoff_and_deadlines_are_inactive_before_latch():
    result = _profile(jump_latched=False, positive_pitch_count=4, wheelie_count=4)

    assert int(result["positive_pitch_count"]) == 0
    assert int(result["wheelie_count"]) == 0
    assert bool(result["dual_wheel_liftoff_seen"]) is False
    assert bool(result["positive_pitch_failure"]) is False
    assert bool(result["wheelie_failure"]) is False
    assert bool(result["missed_liftoff_deadline"]) is False
    assert bool(result["missed_wheel_clearance_deadline"]) is False


def test_takeoff_counters_and_deadlines_activate_after_latch():
    result = _profile(jump_latched=True, positive_pitch_count=4, wheelie_count=4)

    assert bool(result["dual_wheel_liftoff_seen"]) is True
    assert bool(result["positive_pitch_failure"]) is True
    assert bool(result["wheelie_failure"]) is True
    assert bool(result["missed_wheel_clearance_deadline"]) is True
