from __future__ import annotations

import json
import math

import pytest

from jit_dvgc.config import load_config


FORMAL_CHECKPOINTS = (0, 102_400, 256_000, 512_000, 742_400, 998_400)
FORMAL_EVALUATIONS = FORMAL_CHECKPOINTS[1:]
ABSOLUTE_5M_CHECKPOINTS = (
    0,
    245_760,
    983_040,
    2_506_752,
    3_981_312,
    4_988_928,
)
ABSOLUTE_5M_EVALUATIONS = ABSOLUTE_5M_CHECKPOINTS[1:]
V4_15M_CHECKPOINTS = (
    0,
    737_280,
    2_998_272,
    7_495_680,
    11_993_088,
    15_015_936,
)
V4_15M_EVALUATIONS = V4_15M_CHECKPOINTS[1:]


def test_formal_config_is_exactly_39_aligned_blocks(jit_root):
    config = load_config(jit_root / "configs" / "phase_u_formal.json")

    assert config.formal is not None
    assert config.ppo.seed == 820101
    assert config.ppo.requested_transitions == 998_400
    assert config.ppo.block_transitions == 25_600
    assert config.formal.formal_blocks == 39
    assert config.ppo.num_evals == 40
    assert config.formal.checkpoint_transitions == FORMAL_CHECKPOINTS
    assert config.formal.fixed_evaluation_transitions == FORMAL_EVALUATIONS
    assert config.ppo.held_out_seeds == tuple(range(920001, 920009))
    assert config.formal.resume_semantics == "parameter_warm_start_optimizer_reset"
    assert config.schema == "jit_phase_u_formal_v2"
    assert config.events.apex_height == pytest.approx(0.5)
    assert config.events.min_ascent_velocity == pytest.approx(0.05)
    assert config.events.min_descent_velocity == pytest.approx(0.05)


def test_smoke_config_has_no_formal_contract(jit_root):
    config = load_config(jit_root / "configs" / "phase_u_smoke.json")
    assert config.formal is None
    assert config.schema == "jit_phase_u_engineering_smoke_v2"


def test_absolute_smoke_config_is_one_translated_ppo_block(jit_root):
    config = load_config(jit_root / "configs" / "phase_u_absolute_smoke.json")

    assert config.formal is None
    assert config.schema == "jit_phase_u_engineering_smoke_v3"
    assert config.action.joint_target_semantics == "keyframe_centered_absolute"
    assert config.action.knee_target_delta is None
    assert config.ppo.block_transitions == 24_576
    assert config.ppo.requested_transitions == 24_576
    assert config.ppo.num_parallel_envs == 384
    assert config.ppo.unroll_length == 64
    assert config.ppo.batch_size == 16
    assert config.ppo.num_minibatches == 24
    assert config.ppo.num_updates_per_batch == 8


def test_absolute_5m_config_is_exactly_203_aligned_blocks(jit_root):
    config = load_config(jit_root / "configs" / "phase_u_absolute_5m.json")

    assert config.formal is not None
    assert config.schema == "jit_phase_u_formal_v3"
    assert config.action.joint_target_semantics == "keyframe_centered_absolute"
    assert config.action.knee_target_delta is None
    assert config.ppo.seed == 820201
    assert config.ppo.held_out_seeds == tuple(range(930001, 930009))
    assert config.ppo.block_transitions == 24_576
    assert config.ppo.requested_transitions == 4_988_928
    assert config.formal.formal_blocks == 203
    assert config.ppo.num_evals == 204
    assert config.ppo.num_parallel_envs == 384
    assert config.ppo.unroll_length == 64
    assert config.ppo.batch_size == 16
    assert config.ppo.num_minibatches == 24
    assert config.ppo.num_updates_per_batch == 8
    assert config.ppo.learning_rate == pytest.approx(0.0001)
    assert config.ppo.entropy_cost == pytest.approx(0.01)
    assert config.ppo.discounting == pytest.approx(0.99)
    assert config.ppo.gae_lambda == pytest.approx(0.95)
    assert config.ppo.clipping_epsilon == pytest.approx(0.2)
    assert config.ppo.max_grad_norm == pytest.approx(0.5)
    assert config.formal.checkpoint_transitions == ABSOLUTE_5M_CHECKPOINTS
    assert config.formal.fixed_evaluation_transitions == ABSOLUTE_5M_EVALUATIONS


def test_continuation_v4_configs_use_the_active_15m_contract(jit_root):
    smoke = load_config(jit_root / "configs" / "phase_u_continuation_smoke.json")
    formal = load_config(jit_root / "configs" / "phase_u_continuation_10m.json")

    assert smoke.schema == "jit_phase_u_engineering_smoke_v4"
    assert smoke.raw["training_wrapper"] == {
        "full_reset": True,
        "preserve_episode_evidence": True,
    }
    assert smoke.ppo.seed == 820800
    assert smoke.reset.airborne_rsi_probability == pytest.approx(0.08)
    assert smoke.events.jump_zone_x_max == pytest.approx(4.0)
    assert smoke.events.stuck_window_steps is None
    assert smoke.events.stuck_min_progress is None
    assert smoke.events.stuck_forward_velocity_threshold == pytest.approx(0.3)
    assert smoke.physical_limits.max_abs_yaw == pytest.approx(math.radians(45.0))
    assert smoke.physical_limits.terminate_on_prohibited_contact is False
    assert smoke.reward.height_coeff == pytest.approx(40.0)
    assert smoke.reward.low_height_penalty == pytest.approx(5.0)
    assert smoke.reward.stuck_penalty == pytest.approx(100.0)
    assert smoke.reward.yaw_limit_penalty == pytest.approx(100.0)
    assert smoke.reward.steering_action_diff_penalty == pytest.approx(1.0)
    assert smoke.reward.steering_magnitude_penalty == pytest.approx(0.25)
    assert smoke.reward.rear_wheel_action_diff_penalty == pytest.approx(2.0)
    assert smoke.reward.speed_coeff == pytest.approx(1.0)
    assert smoke.reward.desired_velocity == pytest.approx(2.0)
    assert smoke.reward.total_min == pytest.approx(-400.0)
    assert smoke.reward.roll_pitch_failure_penalty == pytest.approx(400.0)
    assert smoke.reward.jump_zone_missed_penalty == pytest.approx(200.0)
    assert smoke.reward.failed_episode_return == pytest.approx(-100.0)
    assert smoke.reward.illegal_contact_penalty == 0.0
    assert smoke.ppo.episode_horizon == 400
    assert smoke.reset.airborne_rsi_z_min == pytest.approx(0.38)
    assert smoke.reset.airborne_rsi_z_max == pytest.approx(0.45)
    assert smoke.reset.airborne_rsi_vz_min == pytest.approx(3.0)
    assert smoke.reset.airborne_rsi_vz_max == pytest.approx(3.6)
    assert smoke.model["naccdmax"] == 512
    assert formal.schema == "jit_phase_u_formal_v4"
    assert formal.ppo.seed == 820801
    assert formal.ppo.held_out_seeds == tuple(range(990001, 990009))
    assert formal.ppo.requested_transitions == 15_015_936
    assert formal.ppo.episode_horizon == 400
    assert formal.formal is not None
    assert formal.formal.formal_blocks == 611
    assert formal.ppo.num_evals == 612
    assert formal.events.jump_zone_x_max == pytest.approx(4.0)
    assert formal.reward.height_coeff == pytest.approx(40.0)
    assert formal.reward.speed_coeff == pytest.approx(1.0)
    assert formal.reward.desired_velocity == pytest.approx(2.0)
    assert formal.reward.total_min == pytest.approx(-400.0)
    assert formal.reward.low_height_penalty == pytest.approx(5.0)
    assert formal.reward.steering_action_diff_penalty == pytest.approx(1.0)
    assert formal.reward.steering_magnitude_penalty == pytest.approx(0.25)
    assert formal.reward.roll_pitch_failure_penalty == pytest.approx(400.0)
    assert formal.reward.jump_zone_missed_penalty == pytest.approx(200.0)
    assert formal.reset.airborne_rsi_probability == pytest.approx(0.08)
    assert formal.model["naccdmax"] == 512
    assert formal.formal.checkpoint_transitions == V4_15M_CHECKPOINTS
    assert formal.formal.fixed_evaluation_transitions == V4_15M_EVALUATIONS
    assert formal.formal.resume_semantics == "fresh_only"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["ppo"].update(seed=820201),
        lambda p: p["ppo"].update(held_out_seeds=list(range(930001, 930009))),
        lambda p: p["events"].update(apex_height=0.6),
        lambda p: p["events"].update(jump_zone_x_max=3.1),
        lambda p: p["events"].update(stuck_forward_velocity_threshold=0.25),
        lambda p: p["physical_limits"].update(max_abs_yaw=0.8),
        lambda p: p["physical_limits"].update(terminate_on_prohibited_contact=True),
        lambda p: p["reward"].update(height_coeff=20.0),
        lambda p: p["reward"].update(low_height_penalty=2.0),
        lambda p: p["reward"].update(stuck_penalty=99.0),
        lambda p: p["reward"].update(yaw_limit_penalty=99.0),
        lambda p: p["reward"].update(steering_action_diff_penalty=0.5),
        lambda p: p["reward"].update(steering_magnitude_penalty=0.1),
        lambda p: p["reward"].update(roll_pitch_failure_penalty=399.0),
        lambda p: p["reward"].update(jump_zone_missed_penalty=199.0),
        lambda p: p["reward"].update(failed_episode_return=-90.0),
        lambda p: p["reward"].update(illegal_contact_penalty=30.0),
        lambda p: p["ppo"].update(episode_horizon=200),
        lambda p: p["reset"].update(airborne_rsi_z_min=0.4),
        lambda p: p["reset"].update(airborne_rsi_vz_max=3.5),
        lambda p: p["reset"].update(airborne_rsi_probability=0.05),
        lambda p: p["reward"].update(apex_success_bonus=49.0),
        lambda p: p["model"].update(naccdmax=48),
        lambda p: p["model"].update(njmax=128),
        lambda p: p["training_wrapper"].update(full_reset=False),
        lambda p: p["training_wrapper"].update(preserve_episode_evidence=False),
    ],
)
def test_continuation_v4_rejects_contract_drift(jit_root, tmp_path, mutate):
    payload = json.loads(
        (jit_root / "configs" / "phase_u_continuation_10m.json").read_text(
            encoding="utf-8"
        )
    )
    mutate(payload)
    path = tmp_path / "mutated_v4.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="approved v4|formal"):
        load_config(path)


def _mutated_absolute_5m(jit_root, tmp_path, mutate):
    source = jit_root / "configs" / "phase_u_absolute_5m.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    mutate(payload)
    path = tmp_path / "mutated_absolute_5m.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["action"].update(joint_target_semantics="incremental_knee"),
        lambda p: p["ppo"].update(requested_transitions=4_964_352),
        lambda p: p["ppo"].update(learning_rate=0.0003),
        lambda p: p["ppo"].update(num_updates_per_batch=7),
        lambda p: p["ppo"].update(seed=820202),
        lambda p: p["model"].update(naconmax=1),
        lambda p: p["reward"].update(height_coeff=21.0),
        lambda p: p["reset"].update(airborne_rsi_probability=0.10),
        lambda p: p["formal"]["checkpoint_transitions"].__setitem__(1, 270_336),
    ],
)
def test_absolute_5m_rejects_any_approved_contract_drift(
    jit_root, tmp_path, mutate
):
    path = _mutated_absolute_5m(jit_root, tmp_path, mutate)
    with pytest.raises(ValueError, match="approved v3|formal|PPO"):
        load_config(path)


def _mutated_formal(jit_root, tmp_path, mutate):
    source = jit_root / "configs" / "phase_u_formal.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    mutate(payload)
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p["ppo"].update(requested_transitions=972_800), "998400"),
        (lambda p: p["ppo"].update(num_evals=39), "num_evals"),
        (
            lambda p: p["formal"]["checkpoint_transitions"].__setitem__(1, 100_000),
            "block-aligned",
        ),
        (
            lambda p: p["formal"].update(
                checkpoint_transitions=[0, 25_600, 998_400],
                fixed_evaluation_transitions=[25_600, 998_400],
            ),
            "exact checkpoint schedule",
        ),
        (
            lambda p: p["formal"].update(checkpoint_transitions=[102_400, 998_400]),
            "start at zero",
        ),
        (
            lambda p: p["formal"].update(
                fixed_evaluation_transitions=[102_400, 256_000]
            ),
            "nonzero checkpoint",
        ),
        (
            lambda p: p["ppo"].update(held_out_seeds=list(range(920001, 920008))),
            "held-out seeds",
        ),
        (
            lambda p: p["formal"].update(resume_semantics="bit_exact"),
            "resume_semantics",
        ),
    ],
)
def test_invalid_formal_contract_is_rejected(jit_root, tmp_path, mutate, message):
    path = _mutated_formal(jit_root, tmp_path, mutate)
    with pytest.raises(ValueError, match=message):
        load_config(path)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["reset"].update(airborne_rsi_probability=0.50),
        lambda p: p["reset"].update(airborne_rsi_vz_min=0.0),
        lambda p: p["events"].update(jump_zone_x_max=3.2),
        lambda p: p["events"].update(apex_height=0.6),
        lambda p: p["reward"].update(height_coeff=21.0),
        lambda p: p["reward"].update(peak_reward_height=0.6),
        lambda p: p["ppo"].update(learning_rate=0.0002),
        lambda p: p["model"].update(naconmax=1),
        lambda p: p["model"].update(mjx_impl="jax"),
    ],
)
def test_formal_rejects_approved_v2_method_constant_drift(
    jit_root, tmp_path, mutate
):
    path = _mutated_formal(jit_root, tmp_path, mutate)
    with pytest.raises(ValueError, match="approved v2"):
        load_config(path)


def test_smoke_schema_rejects_a_formal_section(jit_root, tmp_path):
    smoke = jit_root / "configs" / "phase_u_smoke.json"
    payload = json.loads(smoke.read_text(encoding="utf-8"))
    payload["formal"] = {
        "checkpoint_transitions": list(FORMAL_CHECKPOINTS),
        "fixed_evaluation_transitions": list(FORMAL_EVALUATIONS),
        "resume_semantics": "parameter_warm_start_optimizer_reset",
    }
    path = tmp_path / "invalid_smoke.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="smoke config must not contain formal"):
        load_config(path)


def test_formal_changes_only_approved_budget_seed_and_schedule_fields(jit_root):
    smoke = json.loads(
        (jit_root / "configs" / "phase_u_smoke.json").read_text(encoding="utf-8")
    )
    formal = json.loads(
        (jit_root / "configs" / "phase_u_formal.json").read_text(encoding="utf-8")
    )
    assert formal.pop("formal")["checkpoint_transitions"] == list(FORMAL_CHECKPOINTS)
    smoke["schema"] = formal["schema"]
    for key in ("requested_transitions", "num_evals", "seed"):
        smoke["ppo"][key] = formal["ppo"][key]
    assert formal == smoke


def test_active_v2_config_contains_no_target_or_deceleration_fields(jit_root):
    for name in ("phase_u_smoke.json", "phase_u_formal.json"):
        payload = json.loads((jit_root / "configs" / name).read_text(encoding="utf-8"))
        encoded = json.dumps(payload, sort_keys=True)
        for forbidden in (
            "target_position",
            "target_threshold",
            "target_reach_reward",
            "target_height",
            "direction_coeff",
            "position_coeff",
            "platform_start",
            "platform_back_margin",
            "deceleration_zone",
        ):
            assert forbidden not in encoded
