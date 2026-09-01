"""Immutable protocol lock for the first usable V_up reachable-boundary dataset."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .continuation_labels import DEFAULT_TEST_SEEDS, DEFAULT_VALIDATION_SEEDS
from .upstream_boundary import canonical_sha256, file_sha256, validate_durations, validate_strengths

BOUNDARY_LOCK_SCHEMA = "jit_upstream_boundary_lock_v1"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _verify_protocol_hash(protocol: Mapping[str, Any]) -> str:
    declared = str(protocol.get("protocol_sha256", ""))
    if not declared:
        raise ValueError("TRAIN boundary protocol is missing protocol_sha256")
    payload = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    actual = canonical_sha256(payload)
    if actual != declared:
        raise ValueError("TRAIN boundary protocol content does not match protocol_sha256")
    return declared


def build_boundary_lock(train_protocol_path: Path, train_analysis_path: Path) -> dict[str, Any]:
    """Freeze the TRAIN-discovered boundary recipe before any validation interaction."""
    protocol = _load_json(train_protocol_path)
    analysis = _load_json(train_analysis_path)

    if protocol.get("schema") != "jit_upstream_boundary_protocol_v1":
        raise ValueError("unsupported TRAIN boundary protocol schema")
    if protocol.get("split") != "train" or protocol.get("target") != "V_up":
        raise ValueError("boundary lock requires TRAIN V_up protocol")
    if analysis.get("schema") != "jit_upstream_boundary_analysis_v1":
        raise ValueError("unsupported TRAIN boundary analysis schema")
    if analysis.get("split") != "train" or analysis.get("target") != "V_up":
        raise ValueError("boundary lock requires TRAIN V_up analysis")
    if not bool(analysis.get("boundary_evidence", False)):
        raise ValueError("cannot lock without TRAIN boundary evidence")
    if not bool(analysis.get("dataset_lock_ready", False)):
        raise ValueError("cannot lock before TRAIN dataset_lock_ready")

    protocol_sha = _verify_protocol_hash(protocol)
    action_names = tuple(str(x) for x in protocol.get("selected_action_names", ()))
    signs = tuple(int(x) for x in protocol.get("selected_signs", ()))
    if not action_names or len(set(action_names)) != len(action_names):
        raise ValueError("locked action names must be non-empty and unique")
    if not signs or len(set(signs)) != len(signs) or any(x not in (-1, 1) for x in signs):
        raise ValueError("locked signs must be unique +/-1 values")
    strengths = validate_strengths(protocol.get("strengths", ()))
    durations = validate_durations(protocol.get("durations", ()))

    required_identity = (
        "frozen_pi_up_actor_sha256",
        "frozen_pi_up_payload_sha256",
        "frozen_pi_up_config_sha256",
        "xml_sha256",
        "nominal_catalog_sha256",
        "nominal_labels_sha256",
        "source_training_transitions",
        "negative_role",
        "failure_reason",
    )
    missing = [key for key in required_identity if key not in protocol]
    if missing:
        raise ValueError(f"TRAIN boundary protocol missing identity fields: {missing}")

    lock = {
        "schema": BOUNDARY_LOCK_SCHEMA,
        "status": "locked",
        "target": "V_up",
        "train_protocol_sha256": protocol_sha,
        "train_protocol_file_sha256": file_sha256(train_protocol_path),
        "train_analysis_file_sha256": file_sha256(train_analysis_path),
        "frozen_pi_up_actor_sha256": str(protocol["frozen_pi_up_actor_sha256"]),
        "frozen_pi_up_payload_sha256": str(protocol["frozen_pi_up_payload_sha256"]),
        "frozen_pi_up_config_sha256": str(protocol["frozen_pi_up_config_sha256"]),
        "xml_sha256": str(protocol["xml_sha256"]),
        "nominal_catalog_sha256": str(protocol["nominal_catalog_sha256"]),
        "nominal_labels_sha256": str(protocol["nominal_labels_sha256"]),
        "source_training_transitions": int(protocol["source_training_transitions"]),
        "negative_role": str(protocol["negative_role"]),
        "failure_reason": str(protocol["failure_reason"]),
        "direction_family": str(protocol.get("direction_family", "action_basis_subset")),
        "selected_action_names": list(action_names),
        "selected_signs": list(signs),
        "strengths": list(strengths),
        "durations": list(durations),
        "near_duplicate_tolerances": dict(protocol.get("near_duplicate_tolerances", {})),
        "state_generation": str(protocol.get("state_generation", "")),
        "validation_seeds": list(DEFAULT_VALIDATION_SEEDS),
        "test_seeds": list(DEFAULT_TEST_SEEDS),
        "validation_policy": "apply the locked TRAIN recipe once; do not retune boundary collection from validation outcomes",
        "test_policy": "do not inspect or tune on TEST before V_up model selection is complete",
        "training_transitions": 0,
    }
    lock["lock_sha256"] = canonical_sha256(lock)
    return lock


def write_boundary_lock(train_protocol_path: Path, train_analysis_path: Path, output_path: Path) -> dict[str, Any]:
    lock = build_boundary_lock(train_protocol_path, train_analysis_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return lock


def load_boundary_lock(path: Path) -> dict[str, Any]:
    lock = _load_json(path)
    if lock.get("schema") != BOUNDARY_LOCK_SCHEMA or lock.get("status") != "locked":
        raise ValueError("invalid V_up boundary lock")
    declared = str(lock.get("lock_sha256", ""))
    payload = {key: value for key, value in lock.items() if key != "lock_sha256"}
    if not declared or canonical_sha256(payload) != declared:
        raise ValueError("V_up boundary lock hash mismatch")
    return lock
