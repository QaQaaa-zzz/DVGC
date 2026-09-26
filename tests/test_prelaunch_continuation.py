from __future__ import annotations

import jax
import jax.numpy as jp
import numpy as np
import pytest

from dvgc.bank import SnapshotBank
from dvgc.config import STAGE_ID, load_config
from dvgc.env import (
    END_NONE,
    END_PITCH_LIMIT,
    END_PRETAKEOFF_AIRBORNE,
    END_ROLL_LIMIT,
    OrangeBikeDVGC,
    _quat_from_euler_xyz,
)
from dvgc.reference import ReferenceTrajectory
from dvgc.two_phase_guideline import run_guideline_event_trace
from dvgc.two_phase_runtime import TwoPhaseThresholds, build_two_phase_geometry
from dvgc.two_phase_semantics import ApexBandThresholds, RecoveryThresholds


def _runtime():
    cfg = load_config(
        "configs/default.json",
        {
            "domain_randomization": False,
            "obs_noise_enable": False,
            "use_bank_resets": False,
        },
    )
    env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank())
    reference = ReferenceTrajectory.load("data/reference_jump.csv")
    geometry = build_two_phase_geometry(env.mj_model, cfg)
    thresholds = TwoPhaseThresholds(
        apex=ApexBandThresholds(
            max_abs_com_vz=0.07,
            min_clearance=0.11,
            max_abs_roll=0.08,
            max_abs_pitch=0.06,
            max_angular_speed=1.25,
            min_forward_velocity=3.5,
            relative_x_min=0.15,
            relative_x_max=0.37,
        ),
        recovery=RecoveryThresholds(
            max_abs_roll=0.28,
            max_abs_pitch=0.09,
            max_angular_speed=2.10,
            min_forward_velocity=2.58,
            required_hold_ticks=25,
        ),
    )
    return env, reference, geometry, thresholds


def test_early_airborne_diagnostic_does_not_terminate_guideline():
    env, reference, geometry, thresholds = _runtime()

    report = run_guideline_event_trace(
        env,
        reference,
        geometry,
        thresholds,
        seed=44_000,
        maximum_control_ticks=12,
    )

    assert report["environment_transitions"] == 12
    assert report["terminal"] is False
    assert report["end_code"] == END_NONE


def test_upright_early_airborne_state_can_latch_jump_window_without_support():
    env, _, _, _ = _runtime()
    natural = env.reset(jax.random.PRNGKey(20))
    qpos = np.asarray(natural.data.qpos).copy()
    qvel = np.zeros_like(np.asarray(natural.data.qvel))
    qpos[0] = env._config.step_front_x - 0.5 * (
        env._config.takeoff_window_far + env._config.takeoff_window_near
    )
    qpos[2] = 0.5
    qvel[0] = 1.0
    state = env.reset_from_snapshot(
        jp.asarray(qpos),
        jp.asarray(qvel),
        natural.data.ctrl,
        jax.random.PRNGKey(21),
        jp.asarray(STAGE_ID["approach"], jp.int32),
        jp.asarray(1, jp.int32),
        jp.asarray(0, jp.int32),
        jp.asarray(0, jp.int32),
        airborne_count=jp.asarray(env._config.airborne_confirm_steps, jp.int32),
        prelaunch_airborne_count=jp.asarray(
            env._config.pretakeoff_airborne_fail_steps + 2, jp.int32
        ),
        jump_signal_latched=jp.asarray(False),
    )

    assert bool(state.info["jump_signal_latched"]) is False
    inside = env.step(state, jp.zeros(env.action_size, jp.float32))
    after_qpos = inside.data.qpos.at[0].set(env._config.step_front_x + 0.5)
    after_state = inside.replace(data=inside.data.replace(qpos=after_qpos))
    continued = env.step(after_state, jp.zeros(env.action_size, jp.float32))

    assert bool(inside.info["jump_signal_latched"]) is True
    assert bool(continued.info["jump_signal_latched"]) is True
    assert int(continued.info["end_code"]) != END_PRETAKEOFF_AIRBORNE


@pytest.mark.parametrize(
    ("roll_deg", "pitch_deg", "expected_end_code"),
    ((36.0, 0.0, END_ROLL_LIMIT), (0.0, 76.0, END_PITCH_LIMIT)),
)
def test_existing_roll_and_pitch_hard_limits_still_terminate(
    roll_deg, pitch_deg, expected_end_code
):
    env, _, _, _ = _runtime()
    natural = env.reset(jax.random.PRNGKey(9))
    qpos = np.asarray(natural.data.qpos).copy()
    qvel = np.zeros_like(np.asarray(natural.data.qvel))
    qpos[2] = 1.0
    qpos[3:7] = np.asarray(
        _quat_from_euler_xyz(
            jp.deg2rad(roll_deg), jp.deg2rad(pitch_deg), jp.asarray(0.0)
        )
    )
    state = env.reset_from_snapshot(
        jp.asarray(qpos),
        jp.asarray(qvel),
        natural.data.ctrl,
        jax.random.PRNGKey(10),
        jp.asarray(STAGE_ID["approach"], jp.int32),
        jp.asarray(1, jp.int32),
        jp.asarray(0, jp.int32),
        jp.asarray(0, jp.int32),
    )

    terminal = env.step(state, jp.zeros(env.action_size, jp.float32))

    assert int(terminal.info["terminated"]) == 1
    assert int(terminal.info["end_code"]) == expected_end_code
