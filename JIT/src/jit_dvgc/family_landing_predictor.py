"""Engineering predictor for policy-family first-valid-landing outcomes.

Observed rollout labels remain authoritative.  This model may rank or diagnose
future causal acquisitions, but it cannot establish reachability, create a
positive label, or admit a state into a replay Tube.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import iterative_continuation_fields as iterative
from . import shared_continuation_field_refit as shared
from .config import file_sha256
from .iterative_frontier_protocol import canonical_sha256
from .iterative_weighting_compat import observed_cell_balanced_weights
from .policy_conditioned_continuation_field import _metrics
from .unified_envelope_snapshot import load_unified_envelope_snapshot


SCHEMA = "jit_policy_family_landing_predictor_v1"


def _write(path: Path, value: Mapping[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _verify_self_hash(payload: Mapping[str, Any], field: str) -> None:
    declared = str(payload.get(field, ""))
    base = {key: value for key, value in payload.items() if key != field}
    if len(declared) != 64 or canonical_sha256(base) != declared:
        raise ValueError(f"landing predictor {field} drift")


def _phase_counts(rows: tuple[dict[str, Any], ...]) -> dict[str, dict[str, int]]:
    result = {}
    for phase in ("upstream", "downstream"):
        selected = [row for row in rows if row.get("phase") == phase]
        positive = sum(int(row["label"]) for row in selected)
        result[phase] = {
            "positive_count": positive,
            "negative_count": len(selected) - positive,
            "parent_group_count": len(
                {str(row["parent_group_id"]) for row in selected}
            ),
        }
    return result


def _ranking_metrics(targets: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    targets = np.asarray(targets, dtype=np.int32).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    base = _metrics(targets.astype(np.float64), scores)
    positive_count = int(np.sum(targets == 1))
    if positive_count <= 0:
        average_precision = None
    else:
        order = np.argsort(-scores, kind="stable")
        ranked_positive = targets[order] == 1
        precision = np.cumsum(ranked_positive) / np.arange(1, len(targets) + 1)
        average_precision = float(
            np.sum(precision * ranked_positive) / positive_count
        )
    return {**base, "average_precision": average_precision}


def assess_phase_support(
    counts: Mapping[str, Mapping[str, Mapping[str, int]]],
) -> dict[str, dict[str, Any]]:
    """Declare which phases contain enough observed two-class support to fit."""
    result: dict[str, dict[str, Any]] = {}
    for phase in ("upstream", "downstream"):
        train = counts["train"][phase]
        calibration = counts["calibration"][phase]
        acceptance = counts["acceptance"][phase]
        if int(train["negative_count"]) == 0:
            result[phase] = {
                "fit_authorized": False,
                "status": "not_fitted_single_class_all_positive",
                "reason": "TRAIN has no observed policy-family landing failures",
            }
            continue
        requirements = {
            "train_positive_at_least_20": int(train["positive_count"]) >= 20,
            "train_negative_at_least_20": int(train["negative_count"]) >= 20,
            "train_parent_groups_at_least_3": int(train["parent_group_count"]) >= 3,
            "calibration_has_both_labels": int(calibration["positive_count"]) > 0
            and int(calibration["negative_count"]) > 0,
            "acceptance_has_both_labels": int(acceptance["positive_count"]) > 0
            and int(acceptance["negative_count"]) > 0,
        }
        ready = all(requirements.values())
        result[phase] = {
            "fit_authorized": ready,
            "status": "fit_authorized" if ready else "insufficient_observed_support",
            "requirements": requirements,
        }
    return result


def lock_forward_predictor_scores(
    *,
    catalog_path: Path,
    role: str,
    field_path: Path,
    field_manifest_path: Path,
    calibration_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Score a new unlabeled catalog without consulting continuation outcomes."""
    if role not in {"train", "calibration", "acceptance"}:
        raise ValueError(f"unsupported forward-score role: {role}")
    catalog_path = Path(catalog_path)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if catalog.get("schema") != "jit_unified_boundary_catalog_v1":
        raise ValueError("forward scoring requires a unified boundary catalog")
    if catalog.get("status") != "completed":
        raise ValueError("forward scoring requires a completed acquisition catalog")
    rows = catalog.get("entries")
    if not isinstance(rows, list) or not rows:
        raise ValueError("forward scoring catalog has no candidates")

    field_manifest = json.loads(Path(field_manifest_path).read_text(encoding="utf-8"))
    calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    _verify_self_hash(field_manifest, "manifest_sha256")
    _verify_self_hash(calibration, "calibration_sha256")
    if field_manifest.get("field_file_sha256") != file_sha256(Path(field_path)):
        raise ValueError("landing predictor field file identity drift")
    if calibration.get("field_manifest_sha256") != field_manifest.get(
        "manifest_sha256"
    ):
        raise ValueError("landing predictor calibration/field identity drift")
    threshold = float(calibration["acceptance_threshold_exclusive"])
    upstream_rows = []
    downstream_count = 0
    for row in rows:
        phase = str(row.get("phase"))
        if phase == "downstream":
            downstream_count += 1
            continue
        if phase != "upstream":
            raise ValueError(f"unknown candidate phase: {phase}")
        snapshot_path = (
            catalog_path.parent / str(row["source_bank"]) / str(row["snapshot"])
        )
        snapshot = load_unified_envelope_snapshot(snapshot_path)
        observation = np.asarray(snapshot.observation, dtype=np.float32).reshape(-1)
        if observation.shape != (76,) or not np.isfinite(observation).all():
            raise ValueError("forward scoring requires finite 76-D actor observations")
        upstream_rows.append({**dict(row), "actor_observation": observation.tolist()})
    if not upstream_rows:
        raise ValueError("forward scoring catalog has no upstream candidates")
    scores = iterative._score(Path(field_path), upstream_rows)
    entries = [
        {
            "candidate_id": str(row["candidate_id"]),
            "state_sha256": str(row["state_sha256"]),
            "parent_group_id": str(row["parent_group_id"]),
            "score": float(score),
            "predicted_positive": bool(float(score) > threshold),
        }
        for row, score in zip(upstream_rows, scores, strict=True)
    ]
    report = {
        "schema": "jit_policy_family_landing_forward_scores_v1",
        "status": "locked_before_outcome_analysis",
        "role": role,
        "catalog_path": str(catalog_path),
        "catalog_file_sha256": file_sha256(catalog_path),
        "catalog_protocol_sha256": str(catalog["protocol_sha256"]),
        "field_path": str(Path(field_path)),
        "field_file_sha256": file_sha256(Path(field_path)),
        "field_manifest_sha256": str(field_manifest["manifest_sha256"]),
        "calibration_sha256": str(calibration["calibration_sha256"]),
        "acceptance_threshold_exclusive": threshold,
        "upstream_candidate_count": len(upstream_rows),
        "downstream_candidate_count": downstream_count,
        "outcome_labels_read": False,
        "outcome_labels_used": False,
        "prediction_can_filter_acquisition": False,
        "prediction_can_admit_tube_rows": False,
        "entries": entries,
    }
    report["scores_sha256"] = canonical_sha256(report)
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"forward scores already locked: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write(output_path, report)
    return report


def evaluate_locked_forward_scores(
    *,
    scores_path: Path,
    role_root: Path,
    role: str,
    output_path: Path,
) -> dict[str, Any]:
    """Join immutable pre-outcome scores to newly observed family labels."""
    scores_path = Path(scores_path)
    locked = json.loads(scores_path.read_text(encoding="utf-8"))
    if locked.get("schema") != "jit_policy_family_landing_forward_scores_v1":
        raise ValueError("forward score schema drift")
    if locked.get("status") != "locked_before_outcome_analysis":
        raise ValueError("forward scores were not locked before outcome analysis")
    if locked.get("outcome_labels_read") is not False:
        raise ValueError("forward scores consulted outcome labels")
    if locked.get("role") != role:
        raise ValueError("forward score role mismatch")
    role_manifest, rows = iterative._load_role(Path(role_root), role)
    upstream = {
        (str(row["candidate_id"]), str(row["state_sha256"])): dict(row)
        for row in rows
        if row.get("phase") == "upstream"
    }
    entries = locked.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("forward score entries are empty")
    keys = [
        (str(entry["candidate_id"]), str(entry["state_sha256"]))
        for entry in entries
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate candidate in locked forward scores")
    if set(keys) != set(upstream):
        raise ValueError("locked forward scores do not exactly cover fresh upstream labels")
    ordered_rows = [upstream[key] for key in keys]
    score_values = np.asarray([float(entry["score"]) for entry in entries])
    targets = np.asarray([int(row["label"]) for row in ordered_rows], dtype=np.int32)
    threshold = float(locked["acceptance_threshold_exclusive"])
    accepted = score_values > threshold
    positive_count = int(np.sum(targets == 1))
    negative_count = int(np.sum(targets == 0))
    if positive_count <= 0 or negative_count <= 0:
        raise ValueError("fresh forward audit requires both upstream labels")
    report = {
        "schema": "jit_policy_family_landing_forward_audit_v1",
        "status": "completed_fresh_forward_audit",
        "role": role,
        "scores_path": str(scores_path),
        "scores_file_sha256": file_sha256(scores_path),
        "source_role_manifest_sha256": str(
            role_manifest["role_manifest_sha256"]
        ),
        "fresh_labeled_upstream_count": len(ordered_rows),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_parent_group_count": len(
            {
                str(row.get("parent_group_id", ""))
                for row in ordered_rows
                if int(row["label"]) == 1
            }
        ),
        "negative_parent_group_count": len(
            {
                str(row.get("parent_group_id", ""))
                for row in ordered_rows
                if int(row["label"]) == 0
            }
        ),
        "metrics": _ranking_metrics(targets, score_values),
        "locked_threshold_exclusive": threshold,
        "positive_recall_at_locked_threshold": float(
            np.sum(accepted & (targets == 1)) / positive_count
        ),
        "accepted_negative_count_at_locked_threshold": int(
            np.sum(accepted & (targets == 0))
        ),
        "false_positive_rate_at_locked_threshold": float(
            np.sum(accepted & (targets == 0)) / negative_count
        ),
        "threshold_selected_on_fresh_labels": False,
        "model_refit_on_fresh_labels": False,
        "outcome_labels_loaded_only_after_scores_locked": True,
        "tube_admission_authorized_from_prediction": False,
        "observed_rollout_labels_remain_authoritative": True,
        "test_data_used": False,
        "final_evaluation_data_used": False,
    }
    report["audit_sha256"] = canonical_sha256(report)
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"forward audit already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write(output_path, report)
    return report


def _acceptance_report(
    rows: tuple[dict[str, Any], ...],
    scores: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    targets = np.asarray([int(row["label"]) for row in rows], dtype=np.int32)
    accepted = scores > float(threshold)
    return {
        "candidate_count": len(rows),
        "positive_count": int(np.sum(targets == 1)),
        "negative_count": int(np.sum(targets == 0)),
        "metrics": _ranking_metrics(targets, scores),
        "positive_recall_at_calibration_threshold": float(
            np.sum(accepted & (targets == 1)) / np.sum(targets == 1)
        ),
        "accepted_negative_count_at_calibration_threshold": int(
            np.sum(accepted & (targets == 0))
        ),
        "false_positive_rate_at_calibration_threshold": float(
            np.sum(accepted & (targets == 0)) / np.sum(targets == 0)
        ),
        "threshold_selected_on_acceptance": False,
    }


def fit_family_landing_predictor(
    *,
    train_root: Path,
    calibration_root: Path,
    acceptance_root: Path,
    output_dir: Path,
    model_profile: str = "standard_mlp_64x64_tanh",
) -> dict[str, Any]:
    manifests = {}
    role_rows = {}
    for role, root in (
        ("train", train_root),
        ("calibration", calibration_root),
        ("acceptance", acceptance_root),
    ):
        manifests[role], role_rows[role] = iterative._load_role(Path(root), role)
        if manifests[role].get("policy_identity_kind") != "frozen_policy_family":
            raise ValueError("landing predictor requires policy-family labels")
        if manifests[role].get("continuation_success_criterion") != (
            "first_valid_landing_before_physical_failure"
        ):
            raise ValueError("landing predictor success criterion drift")
    for field in (
        "iteration",
        "policy_actor_sha256",
        "policy_payload_sha256",
        "source_tube_manifest_sha256",
        "plan_sha256",
    ):
        if len({manifests[role][field] for role in manifests}) != 1:
            raise ValueError(f"landing predictor role {field} mismatch")

    counts = {role: _phase_counts(rows) for role, rows in role_rows.items()}
    support = assess_phase_support(counts)
    if support["upstream"]["fit_authorized"] is not True:
        raise ValueError(f"upstream landing predictor support insufficient: {support['upstream']}")
    if model_profile != "standard_mlp_64x64_tanh":
        raise ValueError("active landing predictor architecture must remain 64x64 tanh")

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"landing predictor output already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    iteration = int(manifests["train"]["iteration"])
    train_up, _ = iterative._rows_for_phase(role_rows["train"], "upstream")
    calibration_up, _ = iterative._rows_for_phase(
        role_rows["calibration"], "upstream"
    )
    acceptance_up, _ = iterative._rows_for_phase(
        role_rows["acceptance"], "upstream"
    )
    model = iterative._model_config(model_profile, iteration)
    original_weighting = shared._cell_balanced_weights
    shared._cell_balanced_weights = observed_cell_balanced_weights
    try:
        field_manifest = iterative._fit_standard_mlp_phase(
            train_up,
            phase="upstream",
            model_cfg=model,
            output=output,
        )
    finally:
        shared._cell_balanced_weights = original_weighting

    field_path = output / "upstream" / "field.npz"
    field_manifest = dict(field_manifest)
    field_manifest.update(
        {
            "field_name": f"P_family_land_up^{iteration}",
            "score_semantics": (
                "engineering estimate of first-valid-landing under the frozen "
                "pi_0/pi_1/pi_2 family; not a reachability or safety probability"
            ),
            "source_train_role_manifest_sha256": manifests["train"][
                "role_manifest_sha256"
            ],
            "architecture_predeclared_before_observing_this_batch": False,
            "formal_independent_architecture_selection_claim": False,
        }
    )
    field_manifest.pop("manifest_sha256", None)
    field_manifest["manifest_sha256"] = canonical_sha256(field_manifest)
    _write(output / "upstream" / "manifest.json", field_manifest)

    calibration_scores = iterative._score(field_path, calibration_up)
    calibration = iterative._calibrate(
        "upstream", calibration_up, calibration_scores
    )
    calibration.update(
        {
            "field_manifest_sha256": field_manifest["manifest_sha256"],
            "source_calibration_role_manifest_sha256": manifests["calibration"][
                "role_manifest_sha256"
            ],
            "model_parameters_refit_on_calibration": False,
        }
    )
    calibration["calibration_sha256"] = canonical_sha256(calibration)
    _write(output / "upstream" / "calibration.json", calibration)

    acceptance_scores = iterative._score(field_path, acceptance_up)
    acceptance = _acceptance_report(
        acceptance_up,
        acceptance_scores,
        threshold=float(calibration["acceptance_threshold_exclusive"]),
    )
    acceptance.update(
        {
            "source_acceptance_role_manifest_sha256": manifests["acceptance"][
                "role_manifest_sha256"
            ],
            "acceptance_sha256": "",
        }
    )
    acceptance.pop("acceptance_sha256")
    acceptance["acceptance_sha256"] = canonical_sha256(acceptance)
    _write(output / "upstream" / "acceptance.json", acceptance)

    calibration_passed = bool(calibration["calibration_passed"])
    summary = {
        "schema": SCHEMA,
        "status": "completed_engineering_predictor",
        "iteration": iteration,
        "target": "positive_if_any_pi0_pi1_pi2_reaches_first_valid_landing",
        "model_profile": model_profile,
        "phase_support": support,
        "role_counts": counts,
        "upstream": {
            "field_manifest_sha256": field_manifest["manifest_sha256"],
            "calibration": calibration,
            "acceptance": acceptance,
        },
        "downstream": support["downstream"],
        "acquisition_guidance_authorized": calibration_passed,
        "tube_admission_authorized_from_prediction": False,
        "observed_rollout_labels_remain_authoritative": True,
        "architecture_predeclared_before_observing_this_batch": False,
        "formal_independent_architecture_selection_claim": False,
        "training_transitions": 0,
        "environment_interactions": 0,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "claim_boundary": {
            "forward_reachability_estimator": False,
            "certified_probability": False,
            "certified_safe_set_claim": False,
            "advisory_acquisition_ranking_only": True,
        },
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    _write(output / "summary.json", summary)
    return summary
