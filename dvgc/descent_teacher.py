"""Auditable helpers for the bounded Descent CEM-teacher bootstrap."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


ACTION_ORDER = ("steer", "rear_wheel_drive", "hip", "knee")


def fixed_candidate_folds(records: Sequence[Mapping[str, Any]], folds: int = 3) -> list[list[str]]:
    """Deterministically stratifies whole candidates without tick leakage."""
    grouped: dict[tuple[str, str], list[str]] = {}
    for row in records:
        key = (str(row.get("provisional_label", "")), str(row.get("descent_layer", "")))
        grouped.setdefault(key, []).append(str(row["id"]))
    result = [[] for _ in range(int(folds))]
    offset = 0
    for key in sorted(grouped):
        for index, candidate_id in enumerate(sorted(grouped[key])):
            result[(offset + index) % folds].append(candidate_id)
        offset = (offset + len(grouped[key])) % folds
    return [sorted(fold) for fold in result]


def normalized_observation(observation: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    observation = np.asarray(observation, np.float32)
    return (observation - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)


def nearest_neighbor_audit(
    observations: np.ndarray, actions: np.ndarray, candidate_ids: Sequence[str],
) -> dict[str, Any]:
    """Reports nearest-label consistency without using candidate ID as an input."""
    obs = np.asarray(observations, np.float64)
    action = np.asarray(actions, np.float64)
    if len(obs) < 2:
        return {"pairs": 0, "representable": False, "reason": "fewer_than_two_samples"}
    distances = np.linalg.norm(obs[:, None] - obs[None, :], axis=-1) / np.sqrt(obs.shape[1])
    np.fill_diagonal(distances, np.inf)
    nearest = np.argmin(distances, axis=1)
    nearest_distance = distances[np.arange(len(obs)), nearest]
    action_delta = np.linalg.norm(action - action[nearest], axis=1)
    opposite = np.sum(action * action[nearest], axis=1) < 0
    cross_candidate = np.asarray([candidate_ids[i] != candidate_ids[j] for i, j in enumerate(nearest)])
    close = nearest_distance <= np.quantile(nearest_distance, .25)
    conflicts = close & opposite & (action_delta > .10)
    conflict_fraction = float(np.mean(conflicts))
    return {
        "pairs": int(len(obs)),
        "distance": {
            "min": float(nearest_distance.min()), "median": float(np.median(nearest_distance)),
            "p95": float(np.quantile(nearest_distance, .95)), "max": float(nearest_distance.max()),
        },
        "nearest_action_delta": {
            "median": float(np.median(action_delta)), "p95": float(np.quantile(action_delta, .95)),
            "max": float(action_delta.max()),
        },
        "cross_candidate_fraction": float(np.mean(cross_candidate)),
        "close_opposite_conflict_fraction": conflict_fraction,
        "representable": bool(conflict_fraction <= .20),
        "criterion": "at most 20% of closest-quartile neighbors have opposite residual direction and action delta >0.10",
    }


def trajectory_support_radius(observations: np.ndarray) -> float:
    """95th percentile of within-teacher-trajectory nearest-neighbor distance."""
    obs = np.asarray(observations, np.float64)
    if len(obs) < 2:
        return 0.0
    distance = np.linalg.norm(obs[:, None] - obs[None, :], axis=-1) / np.sqrt(obs.shape[1])
    np.fill_diagonal(distance, np.inf)
    return float(np.quantile(np.min(distance, axis=1), .95))


def relabel_support_gate(
    *, normalized_distance: float, support_p95: float, phase_equal: bool,
    contact_mode_equal: bool, delay_buffer_equal: bool, precursor_equal: bool,
    counterfactual_survival_gain: int, counterfactual_margin_gain: float,
    excluded_or_heldout: bool,
) -> tuple[bool, list[str]]:
    """Applies the user-authorized one-shot student-state relabel contract."""
    reasons = []
    if excluded_or_heldout: reasons.append("excluded_or_heldout")
    if float(normalized_distance) > float(support_p95): reasons.append("outside_teacher_support_p95")
    if not phase_equal: reasons.append("phase_mismatch")
    if not contact_mode_equal: reasons.append("contact_mode_mismatch")
    if not delay_buffer_equal: reasons.append("delay_buffer_mismatch")
    if not precursor_equal: reasons.append("failure_precursor_mismatch")
    if int(counterfactual_survival_gain) <= 0 and float(counterfactual_margin_gain) <= 0:
        reasons.append("counterfactual_not_strictly_better")
    return not reasons, reasons


def physical_improvement_key(before: Mapping[str, Any], after: Mapping[str, Any]) -> tuple[float, ...]:
    """Lexicographic checkpoint key independent of training loss and held-out data."""
    b, a = before["summary"]["overall"], after["summary"]["overall"]
    rows_before = {row["candidate_id"]: row for row in before["rows"]}
    gains = [row["survived_ticks"] - rows_before[row["candidate_id"]]["survived_ticks"] for row in after["rows"]]
    margin_gain = np.median([
        min(row["minimum_margins"]["roll_rad"], row["minimum_margins"]["pitch_rad"])
        - min(rows_before[row["candidate_id"]]["minimum_margins"]["roll_rad"],
              rows_before[row["candidate_id"]]["minimum_margins"]["pitch_rad"])
        for row in after["rows"]
    ])
    return (
        float(a["survival_counts"]["16"] - b["survival_counts"]["16"]),
        float(a["survival_counts"]["12"] - b["survival_counts"]["12"]),
        float(np.median(gains)), float(np.sum(np.maximum(gains, 0))), float(margin_gain),
    )
