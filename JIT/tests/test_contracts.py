from __future__ import annotations

import json

import pytest

from jit_dvgc.config import load_config
from jit_dvgc.constants import (
    ACTION_ORDER,
    ACTOR_FRAME_FIELDS,
    CTRL_DT,
    N_SUBSTEPS,
    SIM_DT,
)


def test_timing_and_orders_drive_the_required_runtime_contract():
    assert SIM_DT == 0.005
    assert CTRL_DT == 0.020
    assert N_SUBSTEPS == 4
    assert ACTION_ORDER == ("steer", "rear_wheel_drive", "hip", "knee")
    assert len(ACTOR_FRAME_FIELDS) == 25
    assert "root_height" in ACTOR_FRAME_FIELDS
    assert "estimated_structure_clearance" not in ACTOR_FRAME_FIELDS
    assert "front_wheel_support" not in ACTOR_FRAME_FIELDS
    assert "rear_wheel_support" not in ACTOR_FRAME_FIELDS


def test_smoke_config_resolves_to_one_exact_1024_environment_block(jit_root):
    cfg = load_config(jit_root / "configs" / "phase_u_smoke.json")

    assert cfg.ppo.num_parallel_envs == 1024
    assert cfg.ppo.block_transitions == 25_600
    assert cfg.ppo.requested_transitions == 25_600
    assert cfg.ppo.requested_transitions // cfg.ppo.block_transitions == 1
    assert cfg.schema == "jit_phase_u_engineering_smoke_v2"
    assert cfg.reset.airborne_rsi_probability == pytest.approx(0.05)
    assert cfg.events.jump_zone_x_min == pytest.approx(2.5)
    assert cfg.events.jump_zone_x_max == pytest.approx(3.1)


def test_invalid_ppo_layout_is_rejected_before_a_run_is_created(jit_root, tmp_path):
    payload = json.loads((jit_root / "configs" / "phase_u_smoke.json").read_text())
    payload["ppo"]["batch_size"] = 127
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="divisible by num_parallel_envs"):
        load_config(invalid)


def test_active_v4_smoke_resolves_to_one_exact_384_environment_block(jit_root):
    config = load_config(jit_root / "configs" / "phase_u_continuation_smoke.json")

    assert config.ppo.num_parallel_envs == 384
    assert config.ppo.block_transitions == 24_576
    assert config.ppo.requested_transitions == 24_576
    assert config.ppo.requested_transitions // config.ppo.block_transitions == 1
    assert config.schema == "jit_phase_u_engineering_smoke_v4"
    assert config.ppo.seed == 820900
    assert config.reset.airborne_rsi_probability == pytest.approx(0.08)
    assert config.events.jump_zone_x_max == pytest.approx(3.9)
    assert config.reward.height_coeff == pytest.approx(40.0)
    assert config.reward.speed_coeff == pytest.approx(1.0)
    assert config.reward.desired_velocity == pytest.approx(2.0)
    assert config.reward.total_min == pytest.approx(-400.0)
    assert config.reward.low_height_penalty == pytest.approx(5.0)
    assert config.reset.airborne_rsi_z_min == pytest.approx(0.38)
    assert config.reset.airborne_rsi_z_max == pytest.approx(0.45)
    assert config.reset.airborne_rsi_vz_min == pytest.approx(3.0)
    assert config.reset.airborne_rsi_vz_max == pytest.approx(3.6)
    assert config.reward.steering_action_diff_penalty == pytest.approx(1.0)
    assert config.reward.steering_magnitude_penalty == pytest.approx(0.25)
    assert config.reward.rear_wheel_action_diff_penalty == pytest.approx(2.0)
    assert config.reward.roll_pitch_failure_penalty == pytest.approx(400.0)
    assert config.reward.jump_zone_missed_penalty == pytest.approx(200.0)
    assert config.reward.failed_episode_return == pytest.approx(-100.0)
    assert config.reward.illegal_contact_penalty == 0.0
    assert config.physical_limits.terminate_on_prohibited_contact is False
    assert config.ppo.episode_horizon == 400
    assert config.model["naccdmax"] == 512


def test_descent_recovery_config_is_supported(jit_root):
    config = load_config(jit_root / "configs" / "descent_recovery_smoke.json")
    assert config.phase == "descent_recovery"
    assert config.descent.recovery_ticks == 25
    assert config.ppo.episode_horizon > config.descent.recovery_ticks


def test_descent_recovery_reward_contract_reads_real_config(jit_root):
    from jit_dvgc.descent_rewards import DescentRewardInputs, descent_recovery_reward
    import jax.numpy as jp

    config = load_config(jit_root / "configs/descent_recovery_smoke.json")
    descent = config.descent
    expected = {
        "reward_contact": 5.0,
        "reward_recovery_tick": 1.0,
        "penalty_bad_contact": 10.0,
        "penalty_failure": 30.0,
        "penalty_timeout": 5.0,
        "reward_forward_progress": 2.0,
        "reward_success": 30.0,
        "roll_posture_coeff": 3.0,
        "pitch_posture_coeff": 0.5,
        "roll_rate_penalty_coeff": 0.5,
        "pitch_rate_penalty_coeff": 0.15,
        "action_smoothness_penalty_coeff": 0.01,
        "total_min": -100.0,
        "total_max": 50.0,
    }
    for field, value in expected.items():
        assert getattr(descent, field) == pytest.approx(value)
    for field in expected:
        if field not in {"total_min", "total_max"}:
            assert getattr(descent, field) > 0.0

    zeros = dict(
        x_delta=0.0,
        valid_contact=False,
        previous_valid_contact=False,
        post_contact=False,
        recovery_success=False,
        previous_recovery_success=False,
        bad_contact=False,
        physical_failure=False,
        timeout=False,
        roll=0.0,
        pitch=0.0,
        roll_rate=0.0,
        pitch_rate=0.0,
        action=jp.zeros(4),
        last_action=jp.zeros(4),
    )

    def reward(**overrides):
        values = {**zeros, **overrides}
        return descent_recovery_reward(
            DescentRewardInputs(**{key: jp.asarray(value) for key, value in values.items()}),
            descent,
        )

    assert float(reward(bad_contact=True).components.bad_contact) == pytest.approx(-10.0)
    assert float(reward(physical_failure=True).components.failure) == pytest.approx(-30.0)
    assert float(reward(timeout=True).components.timeout) == pytest.approx(-5.0)
    assert float(reward(valid_contact=True).components.contact) == pytest.approx(5.0)
    assert float(reward(post_contact=True).components.recovery_tick) == pytest.approx(1.0)
    assert float(reward(recovery_success=True).components.success) == pytest.approx(30.0)
    assert float(reward(x_delta=1.0).components.forward_progress) == pytest.approx(2.0)
    assert float(reward(x_delta=100.0, valid_contact=True, recovery_success=True).total) == pytest.approx(50.0)
    assert float(
        reward(
            bad_contact=True,
            roll_rate=100.0,
            pitch_rate=100.0,
            action=jp.ones(4) * 100.0,
        ).total
    ) == pytest.approx(-100.0)


def test_descent_recovery_rejects_invalid_threshold(jit_root, tmp_path):
    payload = json.loads((jit_root / "configs" / "descent_recovery_smoke.json").read_text())
    payload["descent"]["recovery_ticks"] = payload["ppo"]["episode_horizon"]
    bad = tmp_path / "bad_descent.json"
    bad.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="recovery_ticks"):
        load_config(bad)

def test_legacy_descent_resolved_config_remains_loadable(jit_root):
    legacy = jit_root / "runs/phase_d/descent_recovery_smoke_25600_seed920001_20260827_wrapfix/resolved_config.json"
    config = load_config(legacy)
    assert config.config_sha256 == "3ba3031d162882df89f90ab3c4095f0fe9966de5f903355f24d1c118f5029017"
    assert config.descent.reward_success == 10.0
    assert config.descent.roll_posture_coeff == 0.0

def test_new_descent_config_does_not_silently_fill_missing_dense_fields(jit_root, tmp_path):
    payload = json.loads((jit_root / "configs/descent_recovery_smoke.json").read_text())
    del payload["descent"]["roll_posture_coeff"]
    bad = tmp_path / "missing_dense.json"
    bad.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="descent.*missing"):
        load_config(bad)

def test_deprecated_descent_flag_cannot_bypass_new_reward_contract(jit_root, tmp_path):
    payload = json.loads((jit_root / "configs/descent_recovery_smoke.json").read_text())
    payload["descent"]["terminate_on_prohibited_contact"] = True
    del payload["descent"]["pitch_rate_penalty_coeff"]
    bad = tmp_path / "mixed_legacy.json"
    bad.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="deprecated"):
        load_config(bad)


def test_phase_u_config_hash_is_unchanged(jit_root):
    config = load_config(jit_root / "configs" / "phase_u_smoke.json")
    assert config.phase == "propulsion_ascent"
    assert config.config_sha256 == "b0e2f7cbbf061bde07d8cbcbc20db3d9a534a46610a57d92583fee5fd750cfe8"
