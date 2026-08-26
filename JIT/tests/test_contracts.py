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
    assert config.ppo.seed == 820600
    assert config.reset.airborne_rsi_probability == pytest.approx(0.08)
    assert config.events.jump_zone_x_max == pytest.approx(4.0)
    assert config.reward.height_coeff == pytest.approx(40.0)
    assert config.reset.airborne_rsi_z_min == pytest.approx(0.38)
    assert config.reset.airborne_rsi_z_max == pytest.approx(0.45)
    assert config.reset.airborne_rsi_vz_min == pytest.approx(3.0)
    assert config.reset.airborne_rsi_vz_max == pytest.approx(3.6)
    assert config.reward.steering_action_diff_penalty == pytest.approx(2.0)
    assert config.reward.steering_magnitude_penalty == pytest.approx(0.5)
    assert config.reward.rear_wheel_action_diff_penalty == pytest.approx(2.0)
    assert config.reward.failed_episode_return == pytest.approx(-100.0)
    assert config.reward.illegal_contact_penalty == 0.0
    assert config.physical_limits.terminate_on_prohibited_contact is False
    assert config.ppo.episode_horizon == 400
    assert config.model["naccdmax"] == 320
