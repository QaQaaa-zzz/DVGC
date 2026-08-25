from __future__ import annotations

import json

import pytest

from jit_dvgc.config import load_config


FORMAL_CHECKPOINTS = (0, 102_400, 256_000, 512_000, 742_400, 998_400)
FORMAL_EVALUATIONS = FORMAL_CHECKPOINTS[1:]


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
