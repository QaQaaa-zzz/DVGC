"""Freeze completed pi_k-conditioned transition-band TRAIN evidence.

The output is an immutable training-data input for continuation-field fitting.
It does not train a model, construct a Tube, or consume validation/TEST data.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np

from .config import file_sha256


FREEZE_CONFIG_SCHEMA = "jit_iteration_train_evidence_freeze_config_v1"
FREEZE_PROTOCOL_SCHEMA = "jit_iteration_train_evidence_freeze_protocol_v1"
FROZEN_EVIDENCE_SCHEMA = "jit_frozen_iteration_train_evidence_v1"

_SOURCE_CLAIMS = {
    "training_only_search": True,
    "upstream_transition_band_frozen": True,
    "continuation_field_trained": False,
    "tube_1_constructed": False,
    "pi_1_trained": False,
    "jce_jel_claim": False,
    "certified_safe_set_claim": False,
}

_FROZEN_CLAIMS = {
    "expansion_train_evidence_only": True,
    "continuation_field_trained": False,
    "tube_1_constructed": False,
    "pi_1_trained": False,
    "jce_jel_claim": False,
    "certified_safe_set_claim": False,
}


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _require_sha256(value: Any, *, field: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} must be SHA-256")
    return text


def load_iteration_train_evidence_config(path: Path) -> dict[str, Any]:
    config = _read_object(Path(path))
    if config.get("schema") != FREEZE_CONFIG_SCHEMA:
        raise ValueError("unsupported iteration TRAIN evidence freeze config schema")
    protocol = config.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("iteration TRAIN evidence freeze requires a protocol")
    if protocol.get("schema") != FREEZE_PROTOCOL_SCHEMA:
        raise ValueError("iteration TRAIN evidence freeze protocol schema drift")
    if protocol.get("status") != "predeclared":
        raise ValueError("iteration TRAIN evidence freeze protocol must be predeclared")
    if int(protocol.get("iteration", -1)) < 0:
        raise ValueError("iteration TRAIN evidence freeze iteration must be nonnegative")
    for field in (
        "policy_actor_sha256",
        "policy_payload_sha256",
        "xml_sha256",
        "source_tube_manifest_sha256",
        "source_terminal_protocol_sha256",
        "source_summary_file_sha256",
        "source_labels_file_sha256",
    ):
        _require_sha256(protocol.get(field), field=field)
    if protocol.get("claim_boundary") != _FROZEN_CLAIMS:
        raise ValueError("iteration TRAIN evidence freeze claim boundary drift")
    expected_readiness = protocol.get("expected_readiness")
    if not isinstance(expected_readiness, Mapping) or set(expected_readiness) != {
        "upstream",
        "downstream",
    }:
        raise ValueError("iteration TRAIN evidence freeze requires both phase readiness records")
    criteria = protocol.get("readiness_criteria")
    required_criteria = {
        "minimum_positive_candidates",
        "minimum_negative_candidates",
        "minimum_parent_groups_with_positive",
        "minimum_parent_groups_with_negative",
    }
    if not isinstance(criteria, Mapping) or set(criteria) != required_criteria:
        raise ValueError("iteration TRAIN evidence freeze readiness criteria drift")
    if any(int(criteria[key]) <= 0 for key in required_criteria):
        raise ValueError("iteration TRAIN evidence freeze readiness minima must be positive")
    if int(protocol.get("expected_accumulated_unique_label_count", 0)) <= 0:
        raise ValueError("iteration TRAIN evidence freeze expected label count must be positive")
    if not str(protocol.get("source_root", "")) or not str(config.get("output_dir", "")):
        raise ValueError("iteration TRAIN evidence freeze paths are required")
    if canonical_sha256(protocol) != config.get("expected_protocol_sha256"):
        raise ValueError("iteration TRAIN evidence freeze protocol SHA-256 drift")
    return config


def _phase_counts(
    rows: list[dict[str, Any]], criteria: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    counts: dict[str, dict[str, Any]] = {}
    for phase in ("upstream", "downstream"):
        phase_rows = [row for row in rows if row["phase"] == phase]
        positive_groups = {
            str(row["parent_group_id"]) for row in phase_rows if int(row["label"]) == 1
        }
        negative_groups = {
            str(row["parent_group_id"]) for row in phase_rows if int(row["label"]) == 0
        }
        positive = sum(int(row["label"]) for row in phase_rows)
        negative = len(phase_rows) - positive
        counts[phase] = {
            "candidate_count": len(phase_rows),
            "positive_count": positive,
            "negative_count": negative,
            "positive_parent_group_count": len(positive_groups),
            "negative_parent_group_count": len(negative_groups),
            "ready": bool(
                positive >= int(criteria["minimum_positive_candidates"])
                and negative >= int(criteria["minimum_negative_candidates"])
                and len(positive_groups)
                >= int(criteria["minimum_parent_groups_with_positive"])
                and len(negative_groups)
                >= int(criteria["minimum_parent_groups_with_negative"])
            ),
        }
    return counts


def _validate_rows(
    payload: Mapping[str, Any], *, protocol: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], int]:
    source = payload.get("entries")
    if not isinstance(source, list) or not source:
        raise ValueError("iteration TRAIN evidence source has no entries")
    rows = [dict(row) for row in source]
    seen_states: set[str] = set()
    observation_size: int | None = None
    phase_index = {"upstream": 0, "downstream": 1}
    for row in rows:
        if row.get("split") != "train":
            raise ValueError("iteration evidence freeze accepts TRAIN-only labels")
        phase = str(row.get("phase", ""))
        if phase not in phase_index or int(row.get("phase_index", -1)) != phase_index[phase]:
            raise ValueError("iteration TRAIN evidence phase identity drift")
        if int(row.get("label", -1)) not in (0, 1):
            raise ValueError("iteration TRAIN evidence label must be binary")
        if int(row.get("policy_iteration", -1)) != int(protocol["iteration"]):
            raise ValueError("iteration TRAIN evidence policy iteration drift")
        if row.get("policy_actor_sha256") != protocol["policy_actor_sha256"]:
            raise ValueError("iteration TRAIN evidence actor identity drift")
        if row.get("policy_payload_sha256") != protocol["policy_payload_sha256"]:
            raise ValueError("iteration TRAIN evidence payload identity drift")
        state_sha = _require_sha256(row.get("state_sha256"), field="state_sha256")
        if state_sha in seen_states:
            raise ValueError("iteration TRAIN evidence contains duplicate physical state")
        seen_states.add(state_sha)
        if not str(row.get("parent_group_id", "")):
            raise ValueError("iteration TRAIN evidence parent group is missing")
        _require_sha256(row.get("label_protocol_sha256"), field="label_protocol_sha256")
        _require_sha256(
            row.get("acquisition_protocol_sha256"), field="acquisition_protocol_sha256"
        )
        try:
            observation = np.asarray(row.get("actor_observation"), dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("iteration TRAIN evidence requires a finite actor observation") from exc
        if observation.ndim != 1 or not observation.size or not np.isfinite(observation).all():
            raise ValueError("iteration TRAIN evidence requires a finite actor observation")
        if observation_size is None:
            observation_size = int(observation.size)
        elif int(observation.size) != observation_size:
            raise ValueError("iteration TRAIN evidence actor observation size drift")
    counts = _phase_counts(rows, protocol["readiness_criteria"])
    return rows, counts, int(observation_size or 0)


def _atomic_copy(source: Path, destination: Path) -> None:
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def freeze_iteration_train_evidence(config_path: Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_iteration_train_evidence_config(config_path)
    protocol = dict(config["protocol"])
    source_root = Path(str(protocol["source_root"]))
    summary_path = source_root / "summary.json"
    labels_path = source_root / "accumulated_train_labels.json"
    if file_sha256(summary_path) != protocol["source_summary_file_sha256"]:
        raise ValueError("iteration TRAIN evidence source summary SHA-256 drift")
    if file_sha256(labels_path) != protocol["source_labels_file_sha256"]:
        raise ValueError("iteration TRAIN evidence source labels SHA-256 drift")
    summary = _read_object(summary_path)
    if summary.get("schema") != "jit_downstream_transition_refinement_summary_v1":
        raise ValueError("iteration TRAIN evidence source summary schema drift")
    if summary.get("status") != "transition_band_ready":
        raise ValueError("iteration TRAIN evidence source is not transition-band ready")
    if int(summary.get("iteration", -1)) != int(protocol["iteration"]):
        raise ValueError("iteration TRAIN evidence source iteration drift")
    for summary_field, protocol_field in (
        ("policy_actor_sha256", "policy_actor_sha256"),
        ("policy_payload_sha256", "policy_payload_sha256"),
        ("source_tube_manifest_sha256", "source_tube_manifest_sha256"),
        ("protocol_sha256", "source_terminal_protocol_sha256"),
    ):
        if summary.get(summary_field) != protocol[protocol_field]:
            raise ValueError(f"iteration TRAIN evidence source {summary_field} drift")
    if int(summary.get("accumulated_unique_label_count", -1)) != int(
        protocol["expected_accumulated_unique_label_count"]
    ):
        raise ValueError("iteration TRAIN evidence source count drift")
    if summary.get("readiness") != protocol["expected_readiness"]:
        raise ValueError("iteration TRAIN evidence source readiness drift")
    if summary.get("claim_boundary") != _SOURCE_CLAIMS:
        raise ValueError("iteration TRAIN evidence source claim boundary drift")
    for key in (
        "expert_switching_used",
        "validation_data_used",
        "test_data_used",
        "final_evaluation_data_used",
    ):
        if summary.get(key) is not False:
            raise ValueError(f"iteration TRAIN evidence source {key} drift")
    if int(summary.get("training_transitions", -1)) != 0:
        raise ValueError("iteration TRAIN evidence source unexpectedly trained")

    labels_payload = _read_object(labels_path)
    rows, counts, observation_size = _validate_rows(labels_payload, protocol=protocol)
    if len(rows) != int(protocol["expected_accumulated_unique_label_count"]):
        raise ValueError("iteration TRAIN evidence label count drift")
    if counts != protocol["expected_readiness"] or not all(
        bool(record["ready"]) for record in counts.values()
    ):
        raise ValueError("iteration TRAIN evidence recomputed readiness drift")

    output = Path(str(config["output_dir"]))
    output.mkdir(parents=True, exist_ok=False)
    copied_labels = output / "train_labels.json"
    _atomic_copy(labels_path, copied_labels)
    parent_groups = sorted({str(row["parent_group_id"]) for row in rows})
    by_phase_groups = defaultdict(set)
    for row in rows:
        by_phase_groups[str(row["phase"])].add(str(row["parent_group_id"]))
    manifest = {
        "schema": FROZEN_EVIDENCE_SCHEMA,
        "status": "frozen",
        "artifact_role": "frozen_pi_k_conditioned_transition_band_train_evidence",
        "split": "train",
        "iteration": int(protocol["iteration"]),
        "policy_name": str(protocol["policy_name"]),
        "policy_actor_sha256": str(protocol["policy_actor_sha256"]),
        "policy_payload_sha256": str(protocol["policy_payload_sha256"]),
        "xml_sha256": str(protocol["xml_sha256"]),
        "source_tube_manifest_sha256": str(protocol["source_tube_manifest_sha256"]),
        "source_terminal_protocol_sha256": str(
            protocol["source_terminal_protocol_sha256"]
        ),
        "freeze_protocol_sha256": canonical_sha256(protocol),
        "freeze_config_file_sha256": file_sha256(config_path),
        "source_root": str(source_root),
        "source_summary_file_sha256": str(protocol["source_summary_file_sha256"]),
        "source_labels_file_sha256": str(protocol["source_labels_file_sha256"]),
        "labels_file": "train_labels.json",
        "labels_sha256": file_sha256(copied_labels),
        "label_count": len(rows),
        "observation_size": observation_size,
        "phase_counts": counts,
        "train_parent_group_ids": parent_groups,
        "train_parent_group_ids_by_phase": {
            phase: sorted(by_phase_groups[phase]) for phase in ("upstream", "downstream")
        },
        "validation_parent_group_policy": (
            "reject every TRAIN parent_group_id and every exact or near-duplicate physical state"
        ),
        "validation_protocol_required_before_calibration": True,
        "environment_interactions": 0,
        "training_transitions": 0,
        "expert_switching_used": False,
        "validation_data_used": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "claim_boundary": dict(protocol["claim_boundary"]),
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_frozen_iteration_train_evidence(
    root: Path,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    root = Path(root)
    manifest = _read_object(root / "manifest.json")
    if manifest.get("schema") != FROZEN_EVIDENCE_SCHEMA or manifest.get("status") != "frozen":
        raise ValueError("invalid frozen iteration TRAIN evidence manifest")
    declared = str(manifest.get("manifest_sha256", ""))
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if not declared or canonical_sha256(payload) != declared:
        raise ValueError("frozen iteration TRAIN evidence manifest SHA-256 mismatch")
    labels_path = root / str(manifest.get("labels_file", ""))
    if file_sha256(labels_path) != manifest.get("labels_sha256"):
        raise ValueError("frozen iteration TRAIN evidence labels SHA-256 mismatch")
    labels = _read_object(labels_path)
    rows = labels.get("entries")
    if not isinstance(rows, list) or len(rows) != int(manifest.get("label_count", -1)):
        raise ValueError("frozen iteration TRAIN evidence label count mismatch")
    return manifest, tuple(dict(row) for row in rows)
