from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jit_dvgc import family_landing_predictor as predictor
from jit_dvgc.family_landing_predictor import assess_phase_support


def test_predictor_fits_supported_upstream_and_marks_downstream_single_class() -> None:
    support = assess_phase_support(
        {
            "train": {
                "upstream": {"positive_count": 714, "negative_count": 28, "parent_group_count": 27},
                "downstream": {"positive_count": 516, "negative_count": 0, "parent_group_count": 15},
            },
            "calibration": {
                "upstream": {"positive_count": 227, "negative_count": 4, "parent_group_count": 9},
                "downstream": {"positive_count": 177, "negative_count": 0, "parent_group_count": 5},
            },
            "acceptance": {
                "upstream": {"positive_count": 216, "negative_count": 9, "parent_group_count": 9},
                "downstream": {"positive_count": 171, "negative_count": 0, "parent_group_count": 5},
            },
        }
    )

    assert support["upstream"]["fit_authorized"] is True
    assert support["downstream"] == {
        "fit_authorized": False,
        "status": "not_fitted_single_class_all_positive",
        "reason": "TRAIN has no observed policy-family landing failures",
    }


def test_lock_forward_scores_reads_catalog_snapshots_without_labels(
    tmp_path: Path, monkeypatch
) -> None:
    acquisition = tmp_path / "acquisition"
    acquisition.mkdir()
    catalog = {
        "schema": "jit_unified_boundary_catalog_v1",
        "status": "completed",
        "protocol_sha256": "protocol",
        "entries": [
            {
                "candidate_id": "up-1",
                "phase": "upstream",
                "state_sha256": "state-up",
                "parent_group_id": "group-up",
                "source_bank": "boundary_bank",
                "snapshot": "snapshots/up.npz",
            },
            {
                "candidate_id": "down-1",
                "phase": "downstream",
                "state_sha256": "state-down",
                "parent_group_id": "group-down",
                "source_bank": "boundary_bank",
                "snapshot": "snapshots/down.npz",
            },
        ],
    }
    catalog_path = acquisition / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    field = tmp_path / "field.npz"
    np.savez(field, placeholder=np.asarray([1]))
    manifest = tmp_path / "manifest.json"
    field_manifest = {"field_file_sha256": predictor.file_sha256(field)}
    field_manifest["manifest_sha256"] = predictor.canonical_sha256(field_manifest)
    manifest.write_text(json.dumps(field_manifest), encoding="utf-8")
    calibration = tmp_path / "calibration.json"
    calibration_payload = {
        "field_manifest_sha256": field_manifest["manifest_sha256"],
        "acceptance_threshold_exclusive": 0.4,
    }
    calibration_payload["calibration_sha256"] = predictor.canonical_sha256(
        calibration_payload
    )
    calibration.write_text(
        json.dumps(calibration_payload),
        encoding="utf-8",
    )

    class Snapshot:
        observation = np.arange(76, dtype=np.float32)

    loaded = []

    def fake_load(path: Path):
        loaded.append(path)
        return Snapshot()

    monkeypatch.setattr(predictor, "load_unified_envelope_snapshot", fake_load)
    monkeypatch.setattr(
        predictor.iterative,
        "_score",
        lambda _field, rows: np.asarray([0.75 for _ in rows]),
    )
    output = tmp_path / "scores.json"
    report = predictor.lock_forward_predictor_scores(
        catalog_path=catalog_path,
        role="acceptance",
        field_path=field,
        field_manifest_path=manifest,
        calibration_path=calibration,
        output_path=output,
    )

    assert report["upstream_candidate_count"] == 1
    assert report["downstream_candidate_count"] == 1
    assert report["outcome_labels_read"] is False
    assert report["entries"] == [
        {
            "candidate_id": "up-1",
            "state_sha256": "state-up",
            "parent_group_id": "group-up",
            "score": 0.75,
            "predicted_positive": True,
        }
    ]
    assert loaded == [acquisition / "boundary_bank" / "snapshots/up.npz"]


def test_evaluate_locked_forward_scores_joins_only_after_labels(
    tmp_path: Path, monkeypatch
) -> None:
    score_path = tmp_path / "scores.json"
    score_path.write_text(
        json.dumps(
            {
                "schema": "jit_policy_family_landing_forward_scores_v1",
                "status": "locked_before_outcome_analysis",
                "role": "acceptance",
                "acceptance_threshold_exclusive": 0.4,
                "outcome_labels_read": False,
                "entries": [
                    {
                        "candidate_id": "a",
                        "state_sha256": "sa",
                        "parent_group_id": "ga",
                        "score": 0.8,
                        "predicted_positive": True,
                    },
                    {
                        "candidate_id": "b",
                        "state_sha256": "sb",
                        "parent_group_id": "gb",
                        "score": 0.2,
                        "predicted_positive": False,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    rows = (
        {"candidate_id": "a", "state_sha256": "sa", "phase": "upstream", "label": 1},
        {"candidate_id": "b", "state_sha256": "sb", "phase": "upstream", "label": 0},
        {"candidate_id": "c", "state_sha256": "sc", "phase": "downstream", "label": 1},
    )
    monkeypatch.setattr(
        predictor.iterative,
        "_load_role",
        lambda _root, _role: ({"role_manifest_sha256": "role-manifest"}, rows),
    )
    output = tmp_path / "audit.json"
    report = predictor.evaluate_locked_forward_scores(
        scores_path=score_path,
        role_root=tmp_path / "role",
        role="acceptance",
        output_path=output,
    )

    assert report["fresh_labeled_upstream_count"] == 2
    assert report["metrics"]["roc_auc"] == 1.0
    assert report["metrics"]["average_precision"] == 1.0
    assert report["positive_recall_at_locked_threshold"] == 1.0
    assert report["accepted_negative_count_at_locked_threshold"] == 0
    assert report["false_positive_rate_at_locked_threshold"] == 0.0
