"""Identity-only overlap audit for completed fresh validation candidates.

No consumed validation labels, scores, outcomes, or predictions are read.  The
locked fresh-validation protocol says exact/near duplicates of the already-
consumed validation bank are candidate exclusions with no replacement.  This
module therefore identifies the exclusions before labeling; the caller applies
them to the in-memory validation catalog and continues with the remaining fixed
attempts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import file_sha256


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def audit_fresh_candidate_vs_consumed_validation(
    config_path: Path,
    output: Path,
) -> dict[str, Any]:
    config = _read_object(Path(config_path))
    protocol = config["protocol"]
    old_path = Path(str(protocol["consumed_validation_identity_catalog"]))
    if file_sha256(old_path) != protocol["consumed_validation_identity_catalog_sha256"]:
        raise ValueError("consumed validation identity catalog drift")
    old = _read_object(old_path)
    old_rows = old.get("entries")
    if not isinstance(old_rows, list):
        raise ValueError("consumed validation identity catalog entries missing")
    fresh = _read_object(Path(output) / "candidate_catalog.json")
    fresh_rows = fresh.get("entries")
    if not isinstance(fresh_rows, list):
        raise ValueError("fresh validation candidate entries missing")

    old_states = {str(row.get("state_sha256", "")) for row in old_rows}
    old_obs = np.asarray([row["actor_observation"] for row in old_rows], dtype=np.float64)
    tolerance = float(protocol["near_duplicate_audit"]["actor_observation_atol"])
    exact: list[str] = []
    near: list[str] = []
    for row in fresh_rows:
        state = str(row.get("state_sha256", ""))
        candidate_id = str(row.get("candidate_id", ""))
        if state in old_states:
            exact.append(candidate_id)
            continue
        obs = np.asarray(row["actor_observation"], dtype=np.float64)
        if np.any(np.all(np.abs(old_obs - obs) <= tolerance, axis=1)):
            near.append(candidate_id)

    excluded = set(exact) | set(near)
    report = {
        "schema": "jit_fresh_validation_consumed_identity_audit_v1",
        "status": "independent" if not excluded else "exclusions_required",
        "raw_fresh_candidate_count": len(fresh_rows),
        "accepted_candidate_count_after_exclusion": len(fresh_rows) - len(excluded),
        "consumed_candidate_count": len(old_rows),
        "prefilter_exact_state_overlap_count": len(exact),
        "prefilter_near_duplicate_observation_overlap_count": len(near),
        "exact_state_overlap_count": len(exact),
        "near_duplicate_observation_overlap_count": len(near),
        "actor_observation_atol": tolerance,
        "exact_overlap_candidate_ids": exact,
        "near_overlap_candidate_ids": near,
        "no_replacement_after_exclusion": True,
        "consumed_validation_outcomes_read": False,
        "consumed_validation_predictions_read": False,
        "training_transitions": 0,
        "test_data_used": False,
        "final_evaluation_data_used": False,
    }
    _write(Path(output) / "consumed_validation_identity_audit.json", report)
    return report
