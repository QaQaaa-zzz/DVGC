"""Pure helpers for auditable frozen-policy branch certification."""
from __future__ import annotations

from typing import Any, Iterable, Mapping


# These are the declared physical variants used by both Tube construction and
# independent audit.  Keeping the identifiers next to the values makes every
# Bernoulli outcome traceable to the actual dynamics used for that branch.
DYNAMICS_VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "id": "low_mass_low_friction_low_force",
        "mass_scale": 0.95,
        "friction_scale": 0.90,
        "actuator_force_scale": 0.95,
        "gravity_scale": 1.0,
    },
    {
        "id": "nominal",
        "mass_scale": 1.0,
        "friction_scale": 1.0,
        "actuator_force_scale": 1.0,
        "gravity_scale": 1.0,
    },
    {
        "id": "high_mass_high_friction_high_force",
        "mass_scale": 1.05,
        "friction_scale": 1.10,
        "actuator_force_scale": 1.05,
        "gravity_scale": 1.0,
    },
)


def branch_seed(base_seed: int, state_index: int, branch_index: int) -> int:
    """Return the stable per-state branch seed used by CLI workflows."""
    values = (int(base_seed), int(state_index), int(branch_index))
    if any(value < 0 for value in values):
        raise ValueError(f"Branch seed inputs must be non-negative, got {values}")
    if values[2] >= 10_000:
        raise ValueError("branch_index must be below the per-state seed stride (10000)")
    return values[0] + values[1] * 10_000 + values[2]


def branch_evidence(
    *,
    branch_index: int,
    seed: int,
    seed_namespace: str,
    dynamics_variant: str,
    outcome: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize one rollout result into persistent branch-level evidence."""
    chain = bool(outcome["chain"])
    final = bool(outcome["final"])
    terminated = bool(outcome["terminated"])
    truncated = bool(outcome["truncated"])
    if terminated and truncated:
        raise ValueError("A branch cannot be both physically terminated and timeout-truncated")
    if final:
        cause = "final_recovery"
    elif terminated:
        cause = "physical_failure"
    elif truncated:
        cause = "timeout"
    else:
        cause = "horizon_exhausted"
    return {
        "branch_index": int(branch_index),
        "branch_seed": int(seed),
        "seed_namespace": str(seed_namespace),
        "dynamics_variant": str(dynamics_variant),
        "chain_success": chain,
        "final_recovery": final,
        "terminated": terminated,
        "truncated": truncated,
        "steps": int(outcome["steps"]),
        "terminal_cause": cause,
    }


def summarize_branches(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize mutually exclusive branch outcomes without hiding timeouts."""
    branches = list(rows)
    total = len(branches)
    causes = {
        name: sum(str(row["terminal_cause"]) == name for row in branches)
        for name in ("final_recovery", "physical_failure", "timeout", "horizon_exhausted", "handoff_missed_final")
    }
    if sum(causes.values()) != total:
        raise ValueError("Every branch must have one recognized terminal_cause")

    chain_successes = sum(bool(row["chain_success"]) for row in branches)
    final_successes = sum(bool(row["final_recovery"]) for row in branches)
    false_progress = sum(
        bool(row["chain_success"]) and not bool(row["final_recovery"]) for row in branches
    )
    missed_success = sum(
        not bool(row["chain_success"]) and bool(row["final_recovery"]) for row in branches
    )

    def rate(count: int) -> float:
        return float(count / total) if total else float("nan")

    return {
        "branches": total,
        "chain_successes": chain_successes,
        "final_successes": final_successes,
        "final_recoveries": causes["final_recovery"],
        "physical_failures": causes["physical_failure"],
        "timeouts": causes["timeout"],
        "horizon_exhaustions": causes["horizon_exhausted"],
        "handoff_missed_finals": causes["handoff_missed_final"],
        "final_recovery_rate": rate(causes["final_recovery"]),
        "physical_failure_rate": rate(causes["physical_failure"]),
        "timeout_rate": rate(causes["timeout"]),
        "horizon_exhaustion_rate": rate(causes["horizon_exhausted"]),
        "false_progress_rate": rate(false_progress),
        "missed_success_rate": rate(missed_success),
    }


def assert_disjoint_branch_seeds(
    construction: Iterable[Mapping[str, Any]], audit_seeds: Iterable[int]
) -> None:
    """Reject audit branches that reuse a Tube-construction random trial."""
    construction_seeds = {int(row["branch_seed"]) for row in construction}
    overlap = construction_seeds.intersection(int(seed) for seed in audit_seeds)
    if overlap:
        preview = sorted(overlap)[:5]
        raise ValueError(f"Audit reuses construction branch seeds: {preview}")
