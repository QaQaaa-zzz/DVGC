"""Strict integrity audit for the completed upstream parent-diversity source."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import file_sha256
from .upstream_checkpoint_train_evidence import (
    load_upstream_checkpoint_train_freeze_config,
)


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _array(path: Path) -> list[dict[str, Any]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"JSON array required: {path}")
    return [dict(row) for row in value]


def audit_parent_diversity_source_integrity(config_path: Path) -> dict[str, Any]:
    config = load_upstream_checkpoint_train_freeze_config(Path(config_path))
    protocol = config["protocol"]
    root = Path(str(protocol["parent_diversity_root"]))
    summary_path = root / "summary.json"
    catalog_path = root / "candidate_catalog.json"
    summary = _object(summary_path)
    catalog = _object(catalog_path)

    expected_acq = str(protocol["parent_diversity_acquisition_protocol_sha256"])
    expected_scientific = str(protocol["parent_diversity_scientific_protocol_sha256"])
    actor = str(protocol["policy_actor_sha256"])
    payload = str(protocol["policy_payload_sha256"])
    expected = protocol["expected_expansion"]

    if summary.get("status") != "completed":
        raise ValueError("parent-diversity summary is not completed")
    if summary.get("scientific_protocol_sha256") != expected_scientific:
        raise ValueError("parent-diversity scientific protocol drift")
    if summary.get("acquisition_protocol_sha256") != expected_acq:
        raise ValueError("parent-diversity acquisition protocol drift")
    if catalog.get("schema") != "jit_unified_boundary_catalog_v1" or catalog.get("status") != "completed":
        raise ValueError("parent-diversity candidate catalog invalid")
    if catalog.get("split") != "train" or catalog.get("protocol_sha256") != expected_acq:
        raise ValueError("parent-diversity candidate catalog protocol/split drift")
    if catalog.get("policy_actor_sha256") != actor or catalog.get("policy_payload_sha256") != payload:
        raise ValueError("parent-diversity candidate catalog policy drift")
    if int(catalog.get("candidate_count", -1)) != int(expected["candidate_count"]):
        raise ValueError("parent-diversity candidate catalog count drift")

    labels_dir = Path(str(summary.get("labels_dir", "")))
    label_summary_path = labels_dir / "summary.json"
    labels_path = labels_dir / "labels.json"
    label_summary = _object(label_summary_path)
    labels = _array(labels_path)

    if label_summary.get("schema") != "jit_unified_continuation_labels_v1" or label_summary.get("status") != "completed":
        raise ValueError("parent-diversity label summary invalid")
    if label_summary.get("split") != "train":
        raise ValueError("parent-diversity labels are not TRAIN")
    if label_summary.get("policy_actor_sha256") != actor or label_summary.get("policy_payload_sha256") != payload:
        raise ValueError("parent-diversity label policy drift")
    if label_summary.get("candidate_catalog_protocol_sha256") != expected_acq:
        raise ValueError("parent-diversity label/acquisition protocol drift")
    if label_summary.get("candidate_catalog_file_sha256") != file_sha256(catalog_path):
        raise ValueError("parent-diversity label summary candidate catalog SHA drift")
    if int(label_summary.get("candidate_count", -1)) != int(expected["candidate_count"]):
        raise ValueError("parent-diversity label candidate count drift")
    if int(label_summary.get("label_count", -1)) != int(expected["candidate_count"]):
        raise ValueError("parent-diversity label count drift")
    if int(label_summary.get("positive_count", -1)) != int(expected["positive_count"]):
        raise ValueError("parent-diversity label positive count drift")
    if int(label_summary.get("negative_count", -1)) != int(expected["negative_count"]):
        raise ValueError("parent-diversity label negative count drift")
    for key in ("validation_data_used", "test_data_used", "final_evaluation_data_used"):
        if label_summary.get(key) is not False:
            raise ValueError(f"parent-diversity label summary {key} drift")
    if int(label_summary.get("training_transitions", -1)) != 0:
        raise ValueError("parent-diversity labels unexpectedly trained")

    candidates = [dict(row) for row in catalog.get("entries", [])]
    if len(candidates) != len(labels) or len(labels) != int(expected["candidate_count"]):
        raise ValueError("parent-diversity catalog/label row count mismatch")
    by_state: dict[str, dict[str, Any]] = {}
    for row in candidates:
        state = str(row.get("state_sha256", ""))
        if state in by_state:
            raise ValueError("parent-diversity catalog repeats state")
        by_state[state] = row
    seen_labels: set[str] = set()
    for label in labels:
        state = str(label.get("state_sha256", ""))
        if state in seen_labels or state not in by_state:
            raise ValueError("parent-diversity label state identity mismatch")
        seen_labels.add(state)
        candidate = by_state[state]
        if label.get("candidate_id") != candidate.get("candidate_id"):
            raise ValueError("parent-diversity candidate_id drift")
        if label.get("parent_group_id") != candidate.get("parent_group_id"):
            raise ValueError("parent-diversity parent group drift")
        if label.get("acquisition_protocol_sha256") != expected_acq:
            raise ValueError("parent-diversity row acquisition protocol drift")
        label_obs = np.asarray(label.get("actor_observation"), dtype=np.float32)
        candidate_obs = np.asarray(candidate.get("actor_observation"), dtype=np.float32)
        if label_obs.shape != (76,) or candidate_obs.shape != (76,) or not np.allclose(
            label_obs, candidate_obs, rtol=0.0, atol=1.0e-6
        ):
            raise ValueError("parent-diversity catalog/label actor observation drift")
    if set(by_state) != seen_labels:
        raise ValueError("parent-diversity catalog/label state set mismatch")

    return {
        "status": "source_integrity_ready",
        "summary_file_sha256": file_sha256(summary_path),
        "candidate_catalog_file_sha256": file_sha256(catalog_path),
        "label_summary_file_sha256": file_sha256(label_summary_path),
        "labels_file_sha256": file_sha256(labels_path),
        "candidate_count": len(candidates),
        "label_count": len(labels),
        "positive_count": int(label_summary["positive_count"]),
        "negative_count": int(label_summary["negative_count"]),
        "scientific_protocol_sha256": expected_scientific,
        "acquisition_protocol_sha256": expected_acq,
        "validation_data_used": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "training_transitions": 0,
    }
