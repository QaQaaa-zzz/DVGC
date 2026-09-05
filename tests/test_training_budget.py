from __future__ import annotations

from dataclasses import asdict, replace
import importlib

import pytest

from dvgc import training_budget as budget


def budget_kwargs():
    return {
        "requested_total_transitions": 50_000,
        "num_parallel_envs": 160,
        "episode_horizon": 500,
        "unroll_length": 32,
        "batch_size": 80,
        "num_minibatches": 10,
        "num_updates_per_batch": 2,
        "num_evals": 5,
        "experiment_level": "smoke",
    }


def test_budget_report_uses_total_transitions_and_hand_calculated_brax_alignment():
    importlib.import_module("dvgc.training_budget")

    report = budget.build_ppo_budget_report(**budget_kwargs())

    assert report.requested_total_transitions == 50_000
    assert report.effective_total_transitions == 102_400
    assert report.requested_timesteps == 50_000
    assert report.effective_timesteps == 102_400
    assert report.alignment_overhead == 52_400
    assert report.ppo_rollout_block_size == 25_600
    assert report.ppo_rollout_blocks == 4
    assert report.ppo_optimizer_updates == 80
    assert report.mean_steps_per_env == pytest.approx(640.0)
    assert report.episode_equivalents == pytest.approx(204.8)
    assert report.wall_clock_seconds is None


def test_budget_report_public_fields_are_complete_and_aliases_are_not_inputs():
    report = budget.build_ppo_budget_report(**budget_kwargs())

    assert set(asdict(report)) == {
        "requested_total_transitions",
        "effective_total_transitions",
        "requested_timesteps",
        "effective_timesteps",
        "alignment_overhead",
        "num_parallel_envs",
        "mean_steps_per_env",
        "episode_horizon",
        "episode_equivalents",
        "ppo_rollout_block_size",
        "ppo_rollout_blocks",
        "ppo_optimizer_updates",
        "unroll_length",
        "batch_size",
        "num_minibatches",
        "num_updates_per_batch",
        "num_evals",
        "experiment_level",
        "wall_clock_seconds",
    }
    with pytest.raises(TypeError, match="requested_timesteps"):
        budget.build_ppo_budget_report(**budget_kwargs(), requested_timesteps=1)


def test_experiment_level_vocabulary_is_explicit():
    assert budget.EXPERIMENT_LEVELS == (
        "static",
        "smoke",
        "learnability_pilot",
        "formal_expert",
        "formal_unified",
        "final_evaluation",
    )
    invalid = budget_kwargs() | {"experiment_level": "formal"}
    with pytest.raises(ValueError, match="experiment_level"):
        budget.build_ppo_budget_report(**invalid)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("requested_total_transitions", 0, "requested_total_transitions"),
        ("num_parallel_envs", 0, "num_parallel_envs"),
        ("episode_horizon", 0, "episode_horizon"),
        ("num_updates_per_batch", 0, "num_updates_per_batch"),
        ("num_evals", 0, "num_evals"),
    ],
)
def test_budget_builder_rejects_nonpositive_dimensions(field, value, message):
    kwargs = budget_kwargs() | {field: value}
    with pytest.raises(ValueError, match=message):
        budget.build_ppo_budget_report(**kwargs)


def test_budget_builder_reuses_brax_batch_layout_validation():
    kwargs = budget_kwargs() | {"num_parallel_envs": 192}
    with pytest.raises(ValueError, match="not divisible"):
        budget.build_ppo_budget_report(**kwargs)


def test_prerun_and_completed_budget_reports_validate_wall_clock_at_the_right_time():
    report = budget.build_ppo_budget_report(**budget_kwargs())
    assert budget.validate_ppo_budget_report(report, completed=False)["valid"] is True
    missing = budget.validate_ppo_budget_report(report, completed=True)
    assert missing["valid"] is False
    assert "wall_clock_seconds" in missing["failed"]

    for value in (-1.0, float("nan"), float("inf")):
        invalid = budget.validate_ppo_budget_report(
            replace(report, wall_clock_seconds=value), completed=True
        )
        assert invalid["valid"] is False
        assert "wall_clock_seconds" in invalid["failed"]

    complete = replace(report, wall_clock_seconds=12.5)
    assert budget.validate_ppo_budget_report(complete, completed=True)["valid"] is True


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        (lambda row: row.update(requested_timesteps=49_999), "requested_alias"),
        (lambda row: row.update(effective_timesteps=102_399), "effective_alias"),
        (lambda row: row.update(alignment_overhead=1), "alignment_overhead"),
        (lambda row: row.update(ppo_rollout_blocks=3), "rollout_blocks"),
        (lambda row: row.update(ppo_optimizer_updates=79), "optimizer_updates"),
        (lambda row: row.update(mean_steps_per_env=1.0), "mean_steps_per_env"),
        (lambda row: row.update(episode_equivalents=1.0), "episode_equivalents"),
    ],
)
def test_budget_validator_rejects_alias_or_derived_accounting_divergence(
    mutation, failed_check
):
    payload = asdict(budget.build_ppo_budget_report(**budget_kwargs()))
    mutation(payload)

    result = budget.validate_ppo_budget_report(payload, completed=False)

    assert result["valid"] is False
    assert failed_check in result["failed"]


def test_budget_validator_recomputes_brax_alignment_from_reported_inputs():
    payload = asdict(budget.build_ppo_budget_report(**budget_kwargs()))
    payload.update(
        effective_total_transitions=25_600,
        effective_timesteps=25_600,
        alignment_overhead=-24_400,
        ppo_rollout_blocks=1,
        ppo_optimizer_updates=20,
        mean_steps_per_env=160.0,
        episode_equivalents=51.2,
    )

    result = budget.validate_ppo_budget_report(payload, completed=False)

    assert result["valid"] is False
    assert "effective_alignment" in result["failed"]


def test_budget_validator_recomputes_rollout_block_size_from_layout():
    payload = asdict(budget.build_ppo_budget_report(**budget_kwargs()))
    payload["ppo_rollout_block_size"] = 12_800
    payload["ppo_rollout_blocks"] = 8
    payload["ppo_optimizer_updates"] = 160

    result = budget.validate_ppo_budget_report(payload, completed=False)

    assert result["valid"] is False
    assert "rollout_block_size" in result["failed"]
