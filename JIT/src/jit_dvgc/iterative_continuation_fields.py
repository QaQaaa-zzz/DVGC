"""Fixed-architecture C_up^k/C_down^k fitting and calibration for k >= 1.

Architecture selection is a bootstrap/Iteration-0 scientific decision and must
not be repeated inside the automatic envelope loop.  Later iterations fit the
same 76->8 tanh->1 architecture on the predeclared TRAIN role only, then choose
phase thresholds from the disjoint calibration role without refitting weights.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import file_sha256
from .iterative_frontier_protocol import ROLE_SCHEMA, canonical_sha256
from .policy_conditioned_continuation_field import _metrics
from .shared_continuation_field_refit import _fit_phase
from .upstream_matched_checkpoint_domain_cv import _sigmoid


SUMMARY_SCHEMA = "jit_iterative_continuation_fields_v1"
FIELD_SCHEMA = "jit_shared_continuation_field_v1"
MODEL_BASE = {
    "family": "tiny_mlp_tanh",
    "input": "unified_actor_observation",
    "observation_size": 76,
    "hidden_units": 8,
    "activation": "tanh",
    "parameter_count": 625,
    "normalization": "train_only_zscore_clip10",
    "sample_weighting": "equal_parent_label_cell_mass",
    "l2_weight": 0.01,
    "optimizer": "adam_full_batch_fixed_schedule",
    "steps": 4000,
    "learning_rate": 0.01,
}
CALIBRATION_CONTRACT = {
    "decision_rule": "accept_if_score_strictly_greater_than_max_calibration_negative_score",
    "minimum_roc_auc": 0.70,
    "minimum_positive_recall": 0.20,
    "require_positive_support_in_every_parent": True,
    "require_accepted_positive_in_every_parent": True,
    "accepted_negative_count_must_be_zero": True,
    "model_parameters_refit_on_calibration": False,
    "threshold_is_safety_certificate": False,
}


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _verify_hash(payload: Mapping[str, Any], field: str) -> None:
    declared = str(payload.get(field, ""))
    base = {key: value for key, value in payload.items() if key != field}
    if len(declared) != 64 or canonical_sha256(base) != declared:
        raise ValueError(f"{field} self-hash drift")


def _load_role(root: Path, expected_role: str) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    root = Path(root)
    manifest = _read(root / "role_manifest.json")
    if not isinstance(manifest, dict) or manifest.get("schema") != ROLE_SCHEMA:
        raise ValueError("iterative continuation role manifest schema drift")
    if manifest.get("status") != "completed" or manifest.get("role") != expected_role:
        raise ValueError(f"expected completed {expected_role} frontier role")
    _verify_hash(manifest, "role_manifest_sha256")
    labels_path = root / "logical_labels.json"
    if file_sha256(labels_path) != manifest["logical_labels_file_sha256"]:
        raise ValueError(f"{expected_role} logical labels file drift")
    payload = _read(labels_path)
    if not isinstance(payload, dict) or payload.get("schema") != "jit_iterative_frontier_logical_labels_v1":
        raise ValueError("logical labels schema drift")
    if payload.get("role") != expected_role:
        raise ValueError("logical labels role drift")
    _verify_hash(payload, "labels_sha256")
    rows = payload.get("entries")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{expected_role} logical labels are empty")
    for row in rows:
        if row.get("split") != expected_role or row.get("logical_role") != expected_role:
            raise ValueError(f"{expected_role} logical row split drift")
        if row.get("policy_actor_sha256") != manifest["policy_actor_sha256"]:
            raise ValueError(f"{expected_role} actor identity drift")
        if row.get("policy_payload_sha256") != manifest["policy_payload_sha256"]:
            raise ValueError(f"{expected_role} payload identity drift")
        if int(row.get("label", -1)) not in (0, 1):
            raise ValueError(f"{expected_role} label must be binary")
        obs = np.asarray(row.get("actor_observation"), dtype=np.float32).reshape(-1)
        if obs.shape != (76,) or not np.isfinite(obs).all():
            raise ValueError(f"{expected_role} row requires finite 76-D actor observation")
    return manifest, tuple(dict(row) for row in rows)


def _rows_for_phase(rows, phase: str):
    selected = tuple(dict(row) for row in rows if row.get("phase") == phase)
    if not selected:
        raise ValueError(f"no {phase} continuation rows")
    positive = sum(int(row["label"]) for row in selected)
    groups = {str(row["parent_group_id"]) for row in selected}
    return selected, {
        "candidate_count": len(selected),
        "positive_count": positive,
        "negative_count": len(selected) - positive,
        "parent_group_count": len(groups),
    }


def _score(field_path: Path, rows) -> np.ndarray:
    observations = np.asarray([row["actor_observation"] for row in rows], dtype=np.float32)
    with np.load(field_path) as payload:
        mean = np.asarray(payload["mean"], dtype=np.float32)
        std = np.asarray(payload["std"], dtype=np.float32)
        w1 = np.asarray(payload["w1"], dtype=np.float32)
        b1 = np.asarray(payload["b1"], dtype=np.float32)
        w2 = np.asarray(payload["w2"], dtype=np.float32)
        b2 = float(np.asarray(payload["b2"]))
    x = np.clip((observations - mean) / std, -10.0, 10.0).astype(np.float32)
    hidden = np.tanh(x @ w1 + b1)
    logits = hidden @ w2 + b2
    return _sigmoid(np.asarray(logits, dtype=np.float64))


def _calibrate(phase: str, rows, scores: np.ndarray) -> dict[str, Any]:
    y = np.asarray([int(row["label"]) for row in rows], dtype=np.int32)
    negatives = scores[y == 0]
    positives = scores[y == 1]
    if not len(negatives) or not len(positives):
        raise ValueError(f"{phase} calibration requires both labels")
    threshold = float(np.max(negatives))
    accepted = scores > threshold
    accepted_negative_count = int(np.sum(accepted & (y == 0)))
    positive_recall = float(np.sum(accepted & (y == 1)) / np.sum(y == 1))
    metrics = _metrics(y.astype(np.float64), scores)
    groups = sorted({str(row["parent_group_id"]) for row in rows})
    per_group = {}
    group_pass = True
    for group in groups:
        indices = [i for i, row in enumerate(rows) if str(row["parent_group_id"]) == group]
        gy = y[indices]
        ga = accepted[indices]
        positive_support = int(np.sum(gy == 1))
        accepted_positive = int(np.sum(ga & (gy == 1)))
        per_group[group] = {
            "candidate_count": len(indices),
            "positive_count": positive_support,
            "negative_count": int(np.sum(gy == 0)),
            "accepted_positive_count": accepted_positive,
        }
        group_pass &= positive_support > 0 and accepted_positive > 0
    passed = bool(
        float(metrics["roc_auc"]) >= CALIBRATION_CONTRACT["minimum_roc_auc"]
        and positive_recall >= CALIBRATION_CONTRACT["minimum_positive_recall"]
        and accepted_negative_count == 0
        and group_pass
    )
    return {
        "schema": "jit_iterative_continuation_calibration_v1",
        "status": "completed",
        "phase": phase,
        "acceptance_threshold_exclusive": threshold,
        "decision_rule": CALIBRATION_CONTRACT["decision_rule"],
        "metrics": metrics,
        "positive_recall_at_threshold": positive_recall,
        "accepted_negative_count": accepted_negative_count,
        "candidate_count": len(rows),
        "positive_count": int(np.sum(y == 1)),
        "negative_count": int(np.sum(y == 0)),
        "parent_group_count": len(groups),
        "parent_support": per_group,
        "calibration_passed": passed,
        "contract": dict(CALIBRATION_CONTRACT),
        "threshold_is_safety_certificate": False,
    }


def fit_and_calibrate(*, train_root: Path, calibration_root: Path, output_dir: Path) -> dict[str, Any]:
    train_manifest, train_rows = _load_role(Path(train_root), "train")
    calibration_manifest, calibration_rows = _load_role(Path(calibration_root), "calibration")
    for field in ("iteration", "policy_actor_sha256", "policy_payload_sha256", "source_tube_manifest_sha256", "plan_sha256"):
        if train_manifest.get(field) != calibration_manifest.get(field):
            raise ValueError(f"TRAIN/calibration {field} mismatch")
    iteration = int(train_manifest["iteration"])
    if iteration < 1:
        raise ValueError("automatic fixed-architecture continuation fitting is for k>=1")

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"iterative continuation output already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    model = {
        **MODEL_BASE,
        "phase_specific_seeds": {
            "upstream": 850001 + 100 * iteration,
            "downstream": 850002 + 100 * iteration,
        },
    }

    phase_results = {}
    for phase in ("upstream", "downstream"):
        phase_train, train_counts = _rows_for_phase(train_rows, phase)
        if train_counts["positive_count"] < 20 or train_counts["negative_count"] < 20 or train_counts["parent_group_count"] < 3:
            raise ValueError(f"{phase} TRAIN support insufficient: {train_counts}")
        raw_manifest = _fit_phase(phase_train, phase=phase, model_cfg=model, output=output)
        manifest_path = output / phase / "manifest.json"
        manifest = _read(manifest_path)
        manifest.update(
            {
                "field_name": f"C_{'up' if phase == 'upstream' else 'down'}^{iteration}",
                "iteration": iteration,
                "policy_name": f"pi_{iteration}",
                "policy_actor_sha256": train_manifest["policy_actor_sha256"],
                "policy_payload_sha256": train_manifest["policy_payload_sha256"],
                "source_train_role_manifest_sha256": train_manifest["role_manifest_sha256"],
                "train_counts": train_counts,
            }
        )
        manifest.pop("manifest_sha256", None)
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        _write(manifest_path, manifest)

        phase_calibration, calibration_counts = _rows_for_phase(calibration_rows, phase)
        scores = _score(output / phase / "field.npz", phase_calibration)
        calibration = _calibrate(phase, phase_calibration, scores)
        calibration.update(
            {
                "iteration": iteration,
                "field_name": manifest["field_name"],
                "field_manifest_sha256": manifest["manifest_sha256"],
                "field_file_sha256": manifest["field_file_sha256"],
                "calibration_role_manifest_sha256": calibration_manifest["role_manifest_sha256"],
                "calibration_counts": calibration_counts,
                "model_parameters_refit_on_calibration": False,
                "calibration_rows_may_enter_tube": False,
                "test_data_used": False,
                "final_evaluation_data_used": False,
            }
        )
        calibration["calibration_sha256"] = canonical_sha256(calibration)
        _write(output / phase / "calibration.json", calibration)
        if not calibration["calibration_passed"]:
            raise ValueError(f"{phase} continuation calibration did not pass")
        phase_results[phase] = {
            "field_manifest_sha256": manifest["manifest_sha256"],
            "field_file_sha256": manifest["field_file_sha256"],
            "train_counts": train_counts,
            "calibration_sha256": calibration["calibration_sha256"],
            "acceptance_threshold_exclusive": calibration["acceptance_threshold_exclusive"],
            "calibration_metrics": calibration["metrics"],
            "calibration_positive_recall": calibration["positive_recall_at_threshold"],
        }

    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed_calibrated",
        "iteration": iteration,
        "policy_name": f"pi_{iteration}",
        "policy_actor_sha256": train_manifest["policy_actor_sha256"],
        "policy_payload_sha256": train_manifest["policy_payload_sha256"],
        "source_tube_manifest_sha256": train_manifest["source_tube_manifest_sha256"],
        "architecture": "76->8_tanh->1",
        "model_family": "tiny_mlp_tanh",
        "parameter_count_per_phase": 625,
        "architecture_selection_repeated": False,
        "architecture_frozen_from_bootstrap_iteration": True,
        "model_contract": model,
        "train_role_manifest_sha256": train_manifest["role_manifest_sha256"],
        "calibration_role_manifest_sha256": calibration_manifest["role_manifest_sha256"],
        "phases": phase_results,
        "fields_calibrated": True,
        "training_transitions": 0,
        "environment_interactions": 0,
        "validation_rows_used_for_weight_fit": 0,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "next_tube_construction_authorized": True,
        "claim_boundary": {
            "policy_conditioned_empirical_continuation_fields": True,
            "certified_probability_claim": False,
            "certified_safe_set_claim": False,
            "jce_jel_claim": False,
        },
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    _write(output / "summary.json", summary)
    return summary
