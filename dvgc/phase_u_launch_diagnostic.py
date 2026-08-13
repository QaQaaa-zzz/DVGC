"""Pure contracts for the bounded Phase U feedback-braking diagnostic."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Any, Mapping, Sequence

from jax import numpy as jp


@dataclass(frozen=True)
class FeedbackLaunchSpec:
    launch_bias: float
    knee_ratio: float
    pitch_gain: float
    pitch_rate_gain: float
    active_ticks: int
    action_limit: float = 0.8


def feedback_launch_specs() -> tuple[FeedbackLaunchSpec, ...]:
    """Return the predeclared 384-branch Cartesian grid in stable order."""
    return tuple(
        FeedbackLaunchSpec(*values)
        for values in product(
            (0.2, 0.3, 0.4, 0.5),
            (0.0, 0.5, 1.0, 1.5),
            (0.0, 0.5, 1.0),
            (0.0, 0.03, 0.06, 0.1),
            (4, 7),
        )
    )


def feedback_launch_action(
    spec: FeedbackLaunchSpec,
    *,
    pitch: Any,
    pitch_rate: Any,
    window_latched: Any,
    active_age: Any,
) -> Any:
    """Compute one deployable feedback action without outcome information."""
    active = jp.asarray(window_latched, bool) & (
        jp.asarray(active_age, jp.int32) < spec.active_ticks
    )
    hip = jp.clip(
        spec.launch_bias
        - spec.pitch_gain * jp.asarray(pitch)
        - spec.pitch_rate_gain * jp.asarray(pitch_rate),
        -spec.action_limit,
        spec.action_limit,
    )
    knee = jp.clip(
        spec.knee_ratio * jp.maximum(hip, 0.0), 0.0, spec.action_limit
    )
    command = jp.asarray([0.0, 0.0, hip, knee], jp.float32)
    return jp.where(active, command, jp.zeros((4,), jp.float32))


def close_diagnostic_outcomes(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Close standard mutually exclusive outcome accounting."""
    counts = {
        "success": 0,
        "physical_failure": 0,
        "timeout": 0,
        "other_failure": 0,
    }
    for row in rows:
        flags = (
            bool(row.get("success")),
            bool(row.get("physical_failure")),
            bool(row.get("timeout")),
        )
        if sum(flags) > 1:
            raise ValueError("diagnostic outcomes must be mutually exclusive")
        outcome = (
            "success"
            if flags[0]
            else "physical_failure"
            if flags[1]
            else "timeout"
            if flags[2]
            else "other_failure"
        )
        counts[outcome] += 1
    total = len(rows)
    return {
        "num_branches": total,
        "outcome_counts": counts,
        "empirical_success_rate": counts["success"] / total if total else 0.0,
        "physical_failure_rate": (
            counts["physical_failure"] / total if total else 0.0
        ),
        "timeout_rate": counts["timeout"] / total if total else 0.0,
    }


def _finite_or_infinity(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.inf
    return result if math.isfinite(result) else math.inf


def rank_diagnostic_row(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Rank Apex, safe stable progress, then residual/rate/energy."""
    success = bool(row.get("success"))
    safe_progress = (
        bool(row.get("stable_airborne_reached"))
        and bool(row.get("ascending_reached"))
        and not bool(row.get("physical_failure"))
    )
    return (
        0 if success else 1 if safe_progress else 2,
        _finite_or_infinity(row.get("minimum_apex_contract_residual")),
        _finite_or_infinity(row.get("maximum_angular_speed")),
        _finite_or_infinity(row.get("action_energy")),
        str(row.get("branch_id", "")),
    )


def select_representative_rows(
    rows: Sequence[Mapping[str, Any]], *, maximum: int = 8
) -> list[Mapping[str, Any]]:
    """Choose outcome-driven representatives independently of rendering."""
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    ordered = sorted(rows, key=rank_diagnostic_row)
    successes = [row for row in ordered if bool(row.get("success"))][:maximum]
    selected = list(successes)
    represented = {str(row.get("terminal_reason", "unknown")) for row in selected}
    for reason in sorted(
        {str(row.get("terminal_reason", "unknown")) for row in ordered}
    ):
        if len(selected) >= maximum or reason in represented:
            continue
        candidates = [
            row
            for row in ordered
            if str(row.get("terminal_reason", "unknown")) == reason
        ]
        if candidates:
            selected.append(candidates[0])
            represented.add(reason)
    return selected
