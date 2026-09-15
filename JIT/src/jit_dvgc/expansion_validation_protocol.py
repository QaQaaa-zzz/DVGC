"""Audit a predeclared group-disjoint expansion-validation protocol.

This module performs no environment steps and does not inspect validation
outcomes. It freezes and checks the inputs/recipe that a later runtime must use.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import file_sha256
from .handoff_snapshot import load_snapshot
from .iteration_train_evidence import load_frozen_iteration_train_evidence
from .unified_policy_freeze import load_frozen_unified_manifest
from .upstream_boundary import physical_state_sha256


CONFIG_SCHEMA = "jit_expansion_validation_protocol_config_v1"
PROTOCOL_SCHEMA = "jit_expansion_validation_protocol_v1"

_CLAIMS = {
    "expansion_validation_only": True,
    "continuation_field_trained": False,
    "tube_1_constructed": False,
    "pi_1_trained": False,
    "jce_jel_claim": False,
    "certified_safe_set_claim": False,
}

_DATA_POLICY = {
    "validation_outcomes_may_calibrate_c0": True,
    "validation_rows_may_enter_train_or_tube": False,
    "test_outcomes_inspected": False,
    "final_evaluation_data_used": False,
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


def _rows(value: Any, *, context: str) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        value = value.get("entries", value.get("labels"))
    if not isinstance(value, list):
        raise ValueError(f"row array required: {context}")
    return [dict(row) for row in value]


def _validate_sha(value: Any, *, field: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} must be SHA-256")
    return text


def audit_group_disjointness(
    train_rows: Sequence[Mapping[str, Any]],
    validation_anchors: Sequence[Mapping[str, Any]],
    *,
    observation_atol: float,
) -> dict[str, int | float]:
    """Reject parent, exact-state, or observation-level TRAIN leakage."""
    tolerance = float(observation_atol)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("near-duplicate observation tolerance must be positive")
    train_groups = {str(row.get("parent_group_id", "")) for row in train_rows}
    train_states = {str(row.get("state_sha256", "")) for row in train_rows}
    validation_groups = [str(row.get("parent_group_id", "")) for row in validation_anchors]
    validation_states = [str(row.get("state_sha256", "")) for row in validation_anchors]
    if any(group in train_groups for group in validation_groups):
        raise ValueError("expansion validation contains a TRAIN parent group")
    if any(state in train_states for state in validation_states):
        raise ValueError("expansion validation contains a TRAIN physical state")
    if len(validation_states) != len(set(validation_states)):
        raise ValueError("expansion validation repeats an exact physical state")

    train_observations = np.asarray(
        [row.get("actor_observation") for row in train_rows], dtype=np.float64
    )
    if train_observations.ndim != 2 or not np.isfinite(train_observations).all():
        raise ValueError("frozen TRAIN observations are invalid")
    validation_observations = np.asarray(
        [row.get("actor_observation") for row in validation_anchors], dtype=np.float64
    )
    if (
        validation_observations.ndim != 2
        or validation_observations.shape[1] != train_observations.shape[1]
        or not np.isfinite(validation_observations).all()
    ):
        raise ValueError("validation anchor observations are invalid")
    for observation in validation_observations:
        if np.any(np.all(np.abs(train_observations - observation) <= tolerance, axis=1)):
            raise ValueError("expansion validation contains a near-duplicate TRAIN observation")
    for index, observation in enumerate(validation_observations):
        others = validation_observations[index + 1 :]
        if len(others) and np.any(np.all(np.abs(others - observation) <= tolerance, axis=1)):
            raise ValueError("expansion validation repeats a near-duplicate anchor observation")
    return {
        "train_parent_overlap_count": 0,
        "exact_state_overlap_count": 0,
        "near_duplicate_overlap_count": 0,
        "observation_near_duplicate_atol": tolerance,
    }


def load_expansion_validation_protocol_config(path: Path) -> dict[str, Any]:
    config = _read_object(Path(path))
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unsupported expansion validation protocol config schema")
    protocol = config.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("expansion validation config requires a protocol")
    if protocol.get("schema") != PROTOCOL_SCHEMA or protocol.get("status") != "predeclared":
        raise ValueError("expansion validation protocol identity drift")
    if int(protocol.get("iteration", -1)) != 0 or protocol.get("policy_name") != "pi_0":
        raise ValueError("expansion validation protocol policy identity drift")
    if int(protocol.get("validation_seed", -1)) != 1_000_006:
        raise ValueError("expansion validation must use the locked validation seed")
    if protocol.get("claim_boundary") != _CLAIMS:
        raise ValueError("expansion validation claim boundary drift")
    if protocol.get("data_policy") != _DATA_POLICY:
        raise ValueError("expansion validation data policy drift")
    if not str(config.get("output_dir", "")):
        raise ValueError("expansion validation output directory is required")
    for field in (
        "frozen_policy_file_sha256",
        "policy_actor_sha256",
        "policy_payload_sha256",
        "xml_sha256",
        "frozen_train_manifest_sha256",
    ):
        _validate_sha(protocol.get(field), field=field)
    sources = protocol.get("sources")
    panels = protocol.get("panels")
    if not isinstance(sources, Mapping) or set(sources) != {"upstream", "downstream"}:
        raise ValueError("expansion validation source phases drift")
    if not isinstance(panels, Mapping) or set(panels) != {"upstream", "downstream"}:
        raise ValueError("expansion validation panel phases drift")
    for phase in ("upstream", "downstream"):
        source = sources[phase]
        if not isinstance(source, Mapping) or not isinstance(source.get("anchors"), list):
            raise ValueError(f"expansion validation {phase} anchors are required")
        if not source["anchors"]:
            raise ValueError(f"expansion validation {phase} anchors are empty")
        _validate_sha(source.get("catalog_file_sha256"), field=f"{phase} catalog")
        _validate_sha(source.get("labels_file_sha256"), field=f"{phase} labels")
        if panels[phase].get("terminal_clipping") is not True:
            raise ValueError("expansion validation terminal clipping must remain enabled")
        for anchor in source["anchors"]:
            _validate_sha(anchor.get("state_sha256"), field=f"{phase} anchor state")
    near_duplicate = protocol.get("near_duplicate_audit")
    if not isinstance(near_duplicate, Mapping) or near_duplicate.get(
        "exact_state_sha256"
    ) is not True or near_duplicate.get("all_features_must_be_within_tolerance") is not True:
        raise ValueError("expansion validation near-duplicate contract drift")
    if canonical_sha256(protocol) != config.get("expected_protocol_sha256"):
        raise ValueError("expansion validation protocol SHA-256 drift")
    return config


def _match_anchor(
    anchor: Mapping[str, Any],
    *,
    phase: str,
    catalog_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    catalog_path: Path,
    validation_seed: int,
) -> dict[str, Any]:
    identity_fields = (
        "source_bank",
        "snapshot",
        "parent_group_id",
        "role",
        "tick",
        "state_sha256",
    )
    matches = [
        row
        for row in catalog_rows
        if all(str(row.get(key)) == str(anchor.get(key)) for key in identity_fields)
        and int(row.get("seed", -1)) == validation_seed
    ]
    if len(matches) != 1:
        raise ValueError("expansion validation anchor catalog identity is not unique")
    label_matches = [
        row
        for row in label_rows
        if all(str(row.get(key)) == str(anchor.get(key)) for key in identity_fields)
        and int(row.get("seed", -1)) == validation_seed
    ]
    if len(label_matches) != 1:
        raise ValueError("expansion validation anchor label identity is not unique")
    # Deliberately inspect only split and identity, never validation outcomes.
    if label_matches[0].get("split") != "validation":
        raise ValueError("expansion validation anchor is not in the validation split")
    snapshot_path = catalog_path.parent / str(anchor["source_bank"]) / str(anchor["snapshot"])
    snapshot = load_snapshot(snapshot_path)
    if physical_state_sha256(snapshot) != anchor["state_sha256"]:
        raise ValueError("expansion validation anchor physical-state identity drift")
    expected_snapshot_parent = str(
        anchor.get("snapshot_parent_trajectory", anchor["parent_group_id"])
    )
    if snapshot.parent_trajectory != expected_snapshot_parent:
        raise ValueError("expansion validation anchor parent trajectory drift")
    observation = np.asarray(snapshot.observation, dtype=np.float64)
    if observation.ndim != 1 or not observation.size or not np.isfinite(observation).all():
        raise ValueError("expansion validation anchor observation is invalid")
    return {
        **{key: anchor[key] for key in identity_fields},
        "snapshot_parent_trajectory": expected_snapshot_parent,
        "phase": phase,
        "seed": validation_seed,
        "snapshot_path": str(snapshot_path),
        "actor_observation": observation,
    }


def _panel_budget(panel: Mapping[str, Any], anchor_count: int) -> dict[str, int]:
    action_names = tuple(str(value) for value in panel.get("action_names", ()))
    signs = tuple(int(value) for value in panel.get("signs", ()))
    strengths = tuple(float(value) for value in panel.get("strengths", ()))
    durations = tuple(int(value) for value in panel.get("durations", ()))
    if not action_names or len(action_names) != len(set(action_names)):
        raise ValueError("expansion validation action names must be nonempty and unique")
    if not signs or len(signs) != len(set(signs)) or any(value not in (-1, 1) for value in signs):
        raise ValueError("expansion validation signs must be unique +/-1")
    if not strengths or any(not np.isfinite(value) or value <= 0.0 for value in strengths):
        raise ValueError("expansion validation strengths must be positive")
    if not durations or any(value <= 0 for value in durations):
        raise ValueError("expansion validation durations must be positive")
    directions = len(action_names) * len(signs)
    attempts = int(anchor_count) * directions * len(strengths) * len(durations)
    acquisition = int(anchor_count) * directions * len(strengths) * sum(durations)
    max_ticks = int(panel.get("max_label_ticks", 0))
    if max_ticks <= 0:
        raise ValueError("expansion validation max label ticks must be positive")
    return {
        "attempt_count": attempts,
        "maximum_acquisition_environment_interactions": acquisition,
        "maximum_labeling_environment_interactions": attempts * max_ticks,
    }


def audit_expansion_validation_protocol(config_path: Path) -> dict[str, Any]:
    config = load_expansion_validation_protocol_config(config_path)
    protocol = config["protocol"]
    frozen_policy_path = Path(str(protocol["frozen_policy"]))
    frozen_policy = load_frozen_unified_manifest(frozen_policy_path)
    policy = frozen_policy["policy"]
    if file_sha256(frozen_policy_path) != protocol["frozen_policy_file_sha256"]:
        raise ValueError("expansion validation frozen policy file SHA-256 drift")
    for field, value in (
        ("iteration", policy["iteration"]),
        ("policy_name", policy["name"]),
        ("policy_actor_sha256", policy["actor_sha256"]),
        ("policy_payload_sha256", policy["payload_sha256"]),
        ("xml_sha256", policy["xml_sha256"]),
    ):
        if protocol.get(field) != value:
            raise ValueError(f"expansion validation frozen policy {field} drift")

    frozen_train_root = Path(str(protocol["frozen_train_evidence"]))
    train_manifest, train_rows = load_frozen_iteration_train_evidence(frozen_train_root)
    if train_manifest["manifest_sha256"] != protocol["frozen_train_manifest_sha256"]:
        raise ValueError("expansion validation frozen TRAIN manifest drift")
    if train_manifest["policy_actor_sha256"] != policy["actor_sha256"]:
        raise ValueError("expansion validation TRAIN/policy actor drift")
    if train_manifest["policy_payload_sha256"] != policy["payload_sha256"]:
        raise ValueError("expansion validation TRAIN/policy payload drift")

    validation_seed = int(protocol["validation_seed"])
    anchors: list[dict[str, Any]] = []
    parent_counts: dict[str, int] = {}
    budgets = {
        "attempt_count": 0,
        "maximum_acquisition_environment_interactions": 0,
        "maximum_labeling_environment_interactions": 0,
    }
    for phase in ("upstream", "downstream"):
        source = protocol["sources"][phase]
        catalog_path = Path(str(source["catalog_path"]))
        labels_path = Path(str(source["labels_path"]))
        if file_sha256(catalog_path) != source["catalog_file_sha256"]:
            raise ValueError(f"expansion validation {phase} catalog SHA-256 drift")
        if file_sha256(labels_path) != source["labels_file_sha256"]:
            raise ValueError(f"expansion validation {phase} labels SHA-256 drift")
        catalog_rows = _rows(_read_object(catalog_path), context=str(catalog_path))
        labels_value = json.loads(labels_path.read_text(encoding="utf-8"))
        label_rows = _rows(labels_value, context=str(labels_path))
        phase_anchors = [
            _match_anchor(
                anchor,
                phase=phase,
                catalog_rows=catalog_rows,
                label_rows=label_rows,
                catalog_path=catalog_path,
                validation_seed=validation_seed,
            )
            for anchor in source["anchors"]
        ]
        groups = [str(anchor["parent_group_id"]) for anchor in phase_anchors]
        if len(groups) != len(set(groups)):
            raise ValueError(f"expansion validation {phase} anchors are not parent-group unique")
        parent_counts[phase] = len(groups)
        anchors.extend(phase_anchors)
        phase_budget = _panel_budget(protocol["panels"][phase], len(phase_anchors))
        for key, value in phase_budget.items():
            budgets[key] += int(value)

    leakage = audit_group_disjointness(
        train_rows,
        anchors,
        observation_atol=float(protocol["near_duplicate_audit"]["actor_observation_atol"]),
    )
    for key, value in budgets.items():
        if int(protocol["interaction_budget"].get(key, -1)) != value:
            raise ValueError(f"expansion validation declared {key} drift")
    if protocol["interaction_budget"].get("training_transitions") != 0:
        raise ValueError("expansion validation protocol unexpectedly permits training")

    return {
        "schema": "jit_expansion_validation_protocol_audit_v1",
        "status": "protocol_ready",
        "iteration": 0,
        "policy_name": "pi_0",
        "protocol_sha256": canonical_sha256(protocol),
        "frozen_train_manifest_sha256": train_manifest["manifest_sha256"],
        "validation_anchor_count": len(anchors),
        "validation_parent_group_count_by_phase": parent_counts,
        **budgets,
        **leakage,
        "training_transitions": 0,
        "expert_switching_used": False,
        "validation_outcomes_inspected": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "claim_boundary": dict(protocol["claim_boundary"]),
        "next_gate": (
            "implement and execute this exact validation runtime before fitting or calibrating C_up^0/C_down^0"
        ),
    }
