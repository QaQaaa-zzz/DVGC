"""Unambiguous total-environment-transition accounting for DVGC PPO runs."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math
from typing import Any

from .runtime import (
    ppo_effective_timesteps,
    ppo_rollout_block_steps,
    validate_ppo_batch_layout,
)


EXPERIMENT_LEVELS = (
    "static",
    "smoke",
    "learnability_pilot",
    "formal_expert",
    "formal_unified",
    "final_evaluation",
)


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class PPOBudgetReport:
    requested_total_transitions: int
    effective_total_transitions: int
    requested_timesteps: int
    effective_timesteps: int
    alignment_overhead: int
    num_parallel_envs: int
    mean_steps_per_env: float
    episode_horizon: int
    episode_equivalents: float
    ppo_rollout_block_size: int
    ppo_rollout_blocks: int
    ppo_optimizer_updates: int
    unroll_length: int
    batch_size: int
    num_minibatches: int
    num_updates_per_batch: int
    num_evals: int
    experiment_level: str
    wall_clock_seconds: float | None


def build_ppo_budget_report(
    *,
    requested_total_transitions: int,
    num_parallel_envs: int,
    episode_horizon: int,
    unroll_length: int,
    batch_size: int,
    num_minibatches: int,
    num_updates_per_batch: int,
    num_evals: int,
    experiment_level: str,
    wall_clock_seconds: float | None = None,
) -> PPOBudgetReport:
    """Build one report whose public unit is total environment transitions."""
    requested = _positive_integer(
        "requested_total_transitions", requested_total_transitions
    )
    num_parallel_envs = _positive_integer("num_parallel_envs", num_parallel_envs)
    episode_horizon = _positive_integer("episode_horizon", episode_horizon)
    unroll_length = _positive_integer("unroll_length", unroll_length)
    batch_size = _positive_integer("batch_size", batch_size)
    num_minibatches = _positive_integer("num_minibatches", num_minibatches)
    num_updates_per_batch = _positive_integer(
        "num_updates_per_batch", num_updates_per_batch
    )
    num_evals = _positive_integer("num_evals", num_evals)
    if experiment_level not in EXPERIMENT_LEVELS:
        raise ValueError(f"experiment_level must be one of {EXPERIMENT_LEVELS}")
    validate_ppo_batch_layout(
        num_envs=num_parallel_envs,
        batch_size=batch_size,
        num_minibatches=num_minibatches,
    )
    block = ppo_rollout_block_steps(
        unroll_length=unroll_length,
        batch_size=batch_size,
        num_minibatches=num_minibatches,
    )
    effective = ppo_effective_timesteps(
        requested,
        unroll_length=unroll_length,
        batch_size=batch_size,
        num_minibatches=num_minibatches,
        num_evals=num_evals,
    )
    blocks = effective // block
    return PPOBudgetReport(
        requested_total_transitions=requested,
        effective_total_transitions=effective,
        requested_timesteps=requested,
        effective_timesteps=effective,
        alignment_overhead=effective - requested,
        num_parallel_envs=int(num_parallel_envs),
        mean_steps_per_env=effective / int(num_parallel_envs),
        episode_horizon=int(episode_horizon),
        episode_equivalents=effective / int(episode_horizon),
        ppo_rollout_block_size=block,
        ppo_rollout_blocks=blocks,
        ppo_optimizer_updates=blocks * int(num_minibatches) * int(num_updates_per_batch),
        unroll_length=int(unroll_length),
        batch_size=int(batch_size),
        num_minibatches=int(num_minibatches),
        num_updates_per_batch=int(num_updates_per_batch),
        num_evals=int(num_evals),
        experiment_level=str(experiment_level),
        wall_clock_seconds=wall_clock_seconds,
    )


def _matches(actual: Any, expected: float) -> bool:
    try:
        value = float(actual)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12)


def validate_ppo_budget_report(
    report: PPOBudgetReport | Mapping[str, Any], *, completed: bool
) -> dict[str, Any]:
    """Validate aliases, derived accounting, run level, and wall-clock state."""
    values = asdict(report) if isinstance(report, PPOBudgetReport) else dict(report)
    required = set(PPOBudgetReport.__dataclass_fields__)
    missing = sorted(required - set(values))

    def positive(name: str) -> bool:
        value = values.get(name)
        return isinstance(value, int) and not isinstance(value, bool) and value > 0

    requested = values.get("requested_total_transitions")
    effective = values.get("effective_total_transitions")
    block = values.get("ppo_rollout_block_size")
    minibatches = values.get("num_minibatches")
    updates_per_batch = values.get("num_updates_per_batch")
    environments = values.get("num_parallel_envs")
    horizon = values.get("episode_horizon")
    unroll_length = values.get("unroll_length")
    batch_size = values.get("batch_size")
    num_evals = values.get("num_evals")
    arithmetic_ready = all(
        positive(name)
        for name in (
            "requested_total_transitions",
            "effective_total_transitions",
            "ppo_rollout_block_size",
            "unroll_length",
            "batch_size",
            "num_minibatches",
            "num_updates_per_batch",
            "num_evals",
            "num_parallel_envs",
            "episode_horizon",
        )
    )
    layout_valid = False
    expected_block = None
    expected_effective = None
    if arithmetic_ready:
        try:
            validate_ppo_batch_layout(
                num_envs=environments,
                batch_size=batch_size,
                num_minibatches=minibatches,
            )
            layout_valid = True
            expected_block = ppo_rollout_block_steps(
                unroll_length=unroll_length,
                batch_size=batch_size,
                num_minibatches=minibatches,
            )
            expected_effective = ppo_effective_timesteps(
                requested,
                unroll_length=unroll_length,
                batch_size=batch_size,
                num_minibatches=minibatches,
                num_evals=num_evals,
            )
        except ValueError:
            layout_valid = False
    expected_blocks = effective // block if arithmetic_ready and effective % block == 0 else None
    wall_clock = values.get("wall_clock_seconds")
    wall_clock_valid = (
        isinstance(wall_clock, (int, float))
        and not isinstance(wall_clock, bool)
        and math.isfinite(float(wall_clock))
        and float(wall_clock) >= 0.0
    )
    checks = {
        "required_fields": not missing,
        "positive_dimensions": arithmetic_ready,
        "batch_layout": layout_valid,
        "experiment_level": values.get("experiment_level") in EXPERIMENT_LEVELS,
        "requested_alias": values.get("requested_timesteps") == requested,
        "effective_alias": values.get("effective_timesteps") == effective,
        "rollout_block_size": layout_valid and block == expected_block,
        "effective_alignment": layout_valid
        and effective == expected_effective
        and effective >= requested,
        "alignment_overhead": arithmetic_ready
        and effective >= requested
        and values.get("alignment_overhead") == effective - requested,
        "rollout_blocks": expected_blocks is not None
        and values.get("ppo_rollout_blocks") == expected_blocks,
        "optimizer_updates": expected_blocks is not None
        and values.get("ppo_optimizer_updates")
        == expected_blocks * minibatches * updates_per_batch,
        "mean_steps_per_env": arithmetic_ready
        and _matches(values.get("mean_steps_per_env"), effective / environments),
        "episode_equivalents": arithmetic_ready
        and _matches(values.get("episode_equivalents"), effective / horizon),
        "wall_clock_seconds": wall_clock_valid if completed else wall_clock is None or wall_clock_valid,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {"valid": not failed, "checks": checks, "failed": failed, "missing": missing}
