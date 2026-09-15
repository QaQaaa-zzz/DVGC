#!/usr/bin/env python3
"""Read-only diagnosis of a failed k>=1 continuation calibration.

This tool never fits/refits a model, changes a threshold, acquires states, labels
outcomes, or writes experiment artifacts.  It validates the already-written
partial continuation-field evidence and reports exactly which fixed calibration
gate(s) failed, together with TRAIN/calibration parent support and deterministic
scores from the already-frozen partial field.
"""
from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from jit_dvgc.config import file_sha256
from jit_dvgc.iterative_frontier_protocol import canonical_sha256
from jit_dvgc import iterative_continuation_fields as iterative


PHASES = ("upstream", "downstream")


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _verify_hash(payload: Mapping[str, Any], field: str) -> None:
    declared = str(payload.get(field, ""))
    base = {key: value for key, value in payload.items() if key != field}
    if len(declared) != 64 or canonical_sha256(base) != declared:
        raise ValueError(f"{field} self-hash drift")


def _parent_label_support(rows, phase: str) -> dict[str, Any]:
    selected = [dict(row) for row in rows if row.get("phase") == phase]
    cells = Counter(
        (str(row["parent_group_id"]), int(row["label"])) for row in selected
    )
    groups = sorted({str(row["parent_group_id"]) for row in selected})
    parent_support = {
        group: {
            "positive_count": int(cells[(group, 1)]),
            "negative_count": int(cells[(group, 0)]),
            "observed_label_cell_count": int(cells[(group, 0)] > 0)
            + int(cells[(group, 1)] > 0),
        }
        for group in groups
    }
    positive = sum(int(row["label"]) for row in selected)
    return {
        "candidate_count": len(selected),
        "positive_count": positive,
        "negative_count": len(selected) - positive,
        "parent_group_count": len(groups),
        "outcome_pure_parent_group_count": sum(
            int(value["observed_label_cell_count"] == 1)
            for value in parent_support.values()
        ),
        "parent_support": parent_support,
    }


def _score_support(field_path: Path, rows, phase: str, threshold: float) -> dict[str, Any]:
    selected = [dict(row) for row in rows if row.get("phase") == phase]
    if not selected:
        raise ValueError(f"no {phase} calibration rows")
    scores = iterative._score(field_path, selected)
    y = np.asarray([int(row["label"]) for row in selected], dtype=np.int32)
    accepted = scores > float(threshold)
    groups = sorted({str(row["parent_group_id"]) for row in selected})
    per_group = {}
    for group in groups:
        indices = [
            index
            for index, row in enumerate(selected)
            if str(row["parent_group_id"]) == group
        ]
        gy = y[indices]
        gs = scores[indices]
        ga = accepted[indices]
        positive_scores = gs[gy == 1]
        negative_scores = gs[gy == 0]
        per_group[group] = {
            "candidate_count": len(indices),
            "positive_count": int(np.sum(gy == 1)),
            "negative_count": int(np.sum(gy == 0)),
            "accepted_positive_count": int(np.sum(ga & (gy == 1))),
            "accepted_negative_count": int(np.sum(ga & (gy == 0))),
            "positive_score_min": (
                float(np.min(positive_scores)) if len(positive_scores) else None
            ),
            "positive_score_max": (
                float(np.max(positive_scores)) if len(positive_scores) else None
            ),
            "negative_score_min": (
                float(np.min(negative_scores)) if len(negative_scores) else None
            ),
            "negative_score_max": (
                float(np.max(negative_scores)) if len(negative_scores) else None
            ),
        }
    return {
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "positive_score_min": float(np.min(scores[y == 1])),
        "positive_score_max": float(np.max(scores[y == 1])),
        "negative_score_min": float(np.min(scores[y == 0])),
        "negative_score_max": float(np.max(scores[y == 0])),
        "accepted_positive_count": int(np.sum(accepted & (y == 1))),
        "accepted_negative_count": int(np.sum(accepted & (y == 0))),
        "per_group": per_group,
    }


def _gate_diagnosis(calibration: Mapping[str, Any]) -> dict[str, Any]:
    contract = dict(calibration["contract"])
    metrics = dict(calibration["metrics"])
    parent_support = dict(calibration["parent_support"])
    checks = {
        "roc_auc_at_least_minimum": bool(
            float(metrics["roc_auc"]) >= float(contract["minimum_roc_auc"])
        ),
        "positive_recall_at_least_minimum": bool(
            float(calibration["positive_recall_at_threshold"])
            >= float(contract["minimum_positive_recall"])
        ),
        "accepted_negative_count_zero": int(calibration["accepted_negative_count"]) == 0,
        "positive_support_in_every_parent": all(
            int(value["positive_count"]) > 0 for value in parent_support.values()
        ),
        "accepted_positive_in_every_parent": all(
            int(value["accepted_positive_count"]) > 0
            for value in parent_support.values()
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if not failed:
        category = "no_gate_failure_detected"
    elif "roc_auc_at_least_minimum" in failed:
        category = "ranking_generalization_failure"
    elif "positive_recall_at_least_minimum" in failed:
        category = "conservative_threshold_recall_failure"
    elif "accepted_positive_in_every_parent" in failed:
        category = "parent_local_threshold_coverage_failure"
    else:
        category = "calibration_contract_failure"
    return {
        "checks": checks,
        "failed_checks": failed,
        "failure_category": category,
    }


def analyze(
    *, continuation_root: Path, train_root: Path, calibration_root: Path, phase: str
) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}")
    continuation_root = Path(continuation_root)
    phase_root = continuation_root / phase
    manifest_path = phase_root / "manifest.json"
    calibration_path = phase_root / "calibration.json"
    field_path = phase_root / "field.npz"
    for path in (manifest_path, calibration_path, field_path):
        if not path.is_file():
            raise FileNotFoundError(f"failed continuation artifact missing: {path}")

    manifest = _read(manifest_path)
    calibration = _read(calibration_path)
    _verify_hash(manifest, "manifest_sha256")
    _verify_hash(calibration, "calibration_sha256")
    if calibration.get("phase") != phase or manifest.get("phase") != phase:
        raise ValueError("continuation phase identity drift")
    if calibration.get("field_manifest_sha256") != manifest.get("manifest_sha256"):
        raise ValueError("calibration/field manifest identity drift")
    if calibration.get("field_file_sha256") != file_sha256(field_path):
        raise ValueError("calibration field file identity drift")
    if calibration.get("calibration_passed") is not False:
        raise ValueError("diagnostic expects an explicitly failed calibration")

    train_manifest, train_rows = iterative._load_role(Path(train_root), "train")
    calibration_manifest, calibration_rows = iterative._load_role(
        Path(calibration_root), "calibration"
    )
    for field in (
        "iteration",
        "policy_actor_sha256",
        "policy_payload_sha256",
        "source_tube_manifest_sha256",
        "plan_sha256",
    ):
        if train_manifest.get(field) != calibration_manifest.get(field):
            raise ValueError(f"TRAIN/calibration {field} mismatch")
    if calibration.get("calibration_role_manifest_sha256") != calibration_manifest.get(
        "role_manifest_sha256"
    ):
        raise ValueError("failed calibration role manifest identity drift")

    threshold = float(calibration["acceptance_threshold_exclusive"])
    score_support = _score_support(field_path, calibration_rows, phase, threshold)
    diagnosis = _gate_diagnosis(calibration)
    stored_support = dict(calibration["parent_support"])
    rescored_support = score_support["per_group"]
    for group, stored in stored_support.items():
        if group not in rescored_support:
            raise ValueError("stored calibration parent missing from deterministic rescore")
        rescored = rescored_support[group]
        for field in (
            "candidate_count",
            "positive_count",
            "negative_count",
            "accepted_positive_count",
        ):
            if int(stored[field]) != int(rescored[field]):
                raise ValueError(f"stored/rescored calibration {group} {field} drift")
    if int(calibration["accepted_negative_count"]) != int(
        score_support["accepted_negative_count"]
    ):
        raise ValueError("stored/rescored accepted-negative count drift")

    return {
        "schema": "jit_iterative_calibration_failure_diagnostic_v1",
        "status": "completed_read_only",
        "phase": phase,
        "continuation_root": str(continuation_root),
        "field_manifest_sha256": str(manifest["manifest_sha256"]),
        "field_file_sha256": file_sha256(field_path),
        "calibration_sha256": str(calibration["calibration_sha256"]),
        "policy_actor_sha256": str(manifest.get("policy_actor_sha256", "")),
        "policy_payload_sha256": str(manifest.get("policy_payload_sha256", "")),
        "train_support": _parent_label_support(train_rows, phase),
        "calibration": {
            "candidate_count": int(calibration["candidate_count"]),
            "positive_count": int(calibration["positive_count"]),
            "negative_count": int(calibration["negative_count"]),
            "parent_group_count": int(calibration["parent_group_count"]),
            "acceptance_threshold_exclusive": threshold,
            "metrics": dict(calibration["metrics"]),
            "positive_recall_at_threshold": float(
                calibration["positive_recall_at_threshold"]
            ),
            "accepted_negative_count": int(calibration["accepted_negative_count"]),
            "parent_support": stored_support,
            "deterministic_score_support": score_support,
            "contract": dict(calibration["contract"]),
        },
        "gate_diagnosis": diagnosis,
        "mutations_performed": 0,
        "model_refit_performed": False,
        "threshold_changed": False,
        "new_outcomes_inspected": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuation-root", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, default="upstream")
    args = parser.parse_args()
    result = analyze(
        continuation_root=args.continuation_root,
        train_root=args.train_root,
        calibration_root=args.calibration_root,
        phase=args.phase,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
