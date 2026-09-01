"""Freeze completed group-disjoint pi_k validation evidence.

This is a post-run administrative freeze. It copies the already completed
validation catalog/labels into a self-hashing artifact, re-validates TRAIN
separation, and performs no environment interaction or training.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np

from .config import file_sha256
from .iteration_train_evidence import canonical_sha256, load_frozen_iteration_train_evidence


CONFIG_SCHEMA = "jit_iteration_validation_evidence_freeze_config_v1"
PROTOCOL_SCHEMA = "jit_iteration_validation_evidence_freeze_protocol_v1"
FROZEN_SCHEMA = "jit_frozen_iteration_validation_evidence_v1"


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _read_array(path: Path) -> list[dict[str, Any]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"JSON array required: {path}")
    return [dict(row) for row in value]


def _sha(value: Any, *, field: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{field} must be SHA-256")
    return text


def _audit_train_separation(
    train_rows: tuple[dict[str, Any], ...],
    validation_rows: list[dict[str, Any]],
    *,
    observation_atol: float,
) -> dict[str, Any]:
    """Recheck only validation-vs-TRAIN leakage locked by the runtime.

    Validation descendants are allowed to be close to other validation
    descendants; the formal runtime did not predeclare a validation-vs-
    validation near-duplicate exclusion.
    """
    tolerance = float(observation_atol)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("validation freeze near-duplicate tolerance invalid")
    train_groups = {str(row["parent_group_id"]) for row in train_rows}
    train_states = {str(row["state_sha256"]) for row in train_rows}
    train_obs = np.asarray(
        [row["actor_observation"] for row in train_rows], dtype=np.float64
    )
    if train_obs.ndim != 2 or not np.isfinite(train_obs).all():
        raise ValueError("frozen TRAIN observations invalid during validation freeze")
    for row in validation_rows:
        if str(row["parent_group_id"]) in train_groups:
            raise ValueError("expansion validation contains a TRAIN parent group")
        if str(row["state_sha256"]) in train_states:
            raise ValueError("expansion validation contains a TRAIN physical state")
        obs = np.asarray(row["actor_observation"], dtype=np.float64).reshape(-1)
        if obs.shape != (train_obs.shape[1],) or not np.isfinite(obs).all():
            raise ValueError("validation observation invalid during TRAIN separation audit")
        if np.any(np.all(np.abs(train_obs - obs) <= tolerance, axis=1)):
            raise ValueError("expansion validation contains a near-duplicate TRAIN observation")
    return {
        "train_parent_overlap_count": 0,
        "exact_state_overlap_count": 0,
        "near_duplicate_overlap_count": 0,
        "observation_near_duplicate_atol": tolerance,
    }


def load_iteration_validation_evidence_config(path: Path) -> dict[str, Any]:
    config = _read_object(Path(path))
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unsupported validation evidence freeze config")
    protocol = config.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("validation evidence freeze protocol is required")
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("validation evidence freeze protocol schema drift")
    if protocol.get("status") != "postrun_freeze":
        raise ValueError("validation evidence freeze must be postrun_freeze")
    if int(protocol.get("iteration", -1)) != 0 or protocol.get("policy_name") != "pi_0":
        raise ValueError("validation evidence freeze policy identity drift")
    for field in (
        "policy_actor_sha256",
        "policy_payload_sha256",
        "scientific_protocol_sha256",
        "runtime_protocol_sha256",
        "source_repository_head",
        "frozen_train_manifest_sha256",
    ):
        _sha(protocol.get(field), field=field)
    expected = protocol.get("expected")
    required_expected = {
        "attempt_count",
        "candidate_count",
        "label_count",
        "positive_count",
        "negative_count",
        "phase_candidate_counts",
        "phase_positive_counts",
        "candidate_exclusion_counts",
        "outcome_counts",
        "terminal_probe_outcomes",
    }
    if not isinstance(expected, Mapping) or set(expected) != required_expected:
        raise ValueError("validation evidence freeze expected summary contract drift")
    if set(expected["phase_candidate_counts"]) != {"upstream", "downstream"}:
        raise ValueError("validation evidence freeze phase candidate contract drift")
    if set(expected["phase_positive_counts"]) != {"upstream", "downstream"}:
        raise ValueError("validation evidence freeze phase positive contract drift")
    if not str(protocol.get("source_root", "")) or not str(config.get("output_dir", "")):
        raise ValueError("validation evidence freeze paths are required")
    if canonical_sha256(protocol) != config.get("expected_protocol_sha256"):
        raise ValueError("validation evidence freeze protocol SHA drift")
    return config


def _validate_summary(summary: Mapping[str, Any], protocol: Mapping[str, Any]) -> None:
    if summary.get("schema") != "jit_expansion_validation_runtime_summary_v1":
        raise ValueError("validation source summary schema drift")
    if summary.get("status") != "completed" or summary.get("split") != "validation":
        raise ValueError("validation source is not a completed validation artifact")
    if int(summary.get("iteration", -1)) != int(protocol["iteration"]):
        raise ValueError("validation source iteration drift")
    for key in ("policy_actor_sha256", "policy_payload_sha256"):
        if summary.get(key) != protocol[key]:
            raise ValueError(f"validation source {key} drift")
    if summary.get("scientific_protocol_sha256") != protocol["scientific_protocol_sha256"]:
        raise ValueError("validation source scientific protocol drift")
    if summary.get("runtime_protocol_sha256") != protocol["runtime_protocol_sha256"]:
        raise ValueError("validation source runtime protocol drift")
    expected = protocol["expected"]
    for field in (
        "attempt_count",
        "candidate_count",
        "label_count",
        "positive_count",
        "negative_count",
        "phase_candidate_counts",
        "phase_positive_counts",
        "candidate_exclusion_counts",
        "outcome_counts",
        "terminal_probe_outcomes",
    ):
        if summary.get(field) != expected[field]:
            raise ValueError(f"validation source {field} drift")
    if int(summary.get("training_transitions", -1)) != 0:
        raise ValueError("validation source unexpectedly trained")
    if summary.get("expert_switching_used") is not False:
        raise ValueError("validation source used expert switching")
    if summary.get("validation_data_used") is not True:
        raise ValueError("validation source did not mark validation usage")
    if summary.get("test_data_used") is not False or summary.get("final_evaluation_data_used") is not False:
        raise ValueError("validation source touched TEST/final data")
    if summary.get("validation_rows_may_enter_train_or_tube") is not False:
        raise ValueError("validation source permits leakage into TRAIN/Tube")
    for field in ("continuation_field_trained", "tube_1_constructed", "pi_1_trained"):
        if summary.get(field) is not False:
            raise ValueError(f"validation source {field} claim drift")


def _validate_rows(
    *,
    catalog: Mapping[str, Any],
    labels: list[dict[str, Any]],
    protocol: Mapping[str, Any],
    train_rows: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    entries_value = catalog.get("entries")
    if not isinstance(entries_value, list):
        raise ValueError("validation candidate catalog entries are missing")
    entries = [dict(row) for row in entries_value]
    expected = protocol["expected"]
    if len(entries) != int(expected["candidate_count"]) or len(labels) != int(expected["label_count"]):
        raise ValueError("validation row count drift")
    by_state: dict[str, dict[str, Any]] = {}
    observation_size: int | None = None
    validation_rows_for_leakage: list[dict[str, Any]] = []
    for row in entries:
        if row.get("split") != "validation":
            raise ValueError("frozen validation catalog contains non-validation row")
        state_sha = _sha(row.get("state_sha256"), field="validation state_sha256")
        if state_sha in by_state:
            raise ValueError("frozen validation catalog repeats physical state")
        phase = str(row.get("phase", ""))
        if phase not in ("upstream", "downstream"):
            raise ValueError("validation catalog phase drift")
        if not str(row.get("parent_group_id", "")):
            raise ValueError("validation catalog parent group missing")
        obs = np.asarray(row.get("actor_observation"), dtype=np.float64).reshape(-1)
        if obs.size == 0 or not np.isfinite(obs).all():
            raise ValueError("validation catalog actor observation invalid")
        if observation_size is None:
            observation_size = int(obs.size)
        elif int(obs.size) != observation_size:
            raise ValueError("validation catalog observation width drift")
        by_state[state_sha] = row
        validation_rows_for_leakage.append(
            {
                "parent_group_id": str(row["parent_group_id"]),
                "state_sha256": state_sha,
                "actor_observation": obs,
            }
        )

    label_states: set[str] = set()
    phase_counts: Counter[str] = Counter()
    phase_positive: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    parent_stats: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in labels:
        if row.get("split") != "validation":
            raise ValueError("frozen validation labels contain non-validation row")
        state_sha = _sha(row.get("state_sha256"), field="validation label state_sha256")
        if state_sha in label_states:
            raise ValueError("frozen validation labels repeat physical state")
        label_states.add(state_sha)
        candidate = by_state.get(state_sha)
        if candidate is None:
            raise ValueError("validation label has no matching candidate")
        if row.get("candidate_id") != candidate.get("candidate_id"):
            raise ValueError("validation candidate/label identity drift")
        if row.get("parent_group_id") != candidate.get("parent_group_id"):
            raise ValueError("validation candidate/label parent drift")
        if row.get("phase") != candidate.get("phase"):
            raise ValueError("validation candidate/label phase drift")
        label = int(row.get("label", -1))
        if label not in (0, 1):
            raise ValueError("validation label must be binary")
        obs = np.asarray(row.get("actor_observation"), dtype=np.float64).reshape(-1)
        candidate_obs = np.asarray(candidate["actor_observation"], dtype=np.float64).reshape(-1)
        if obs.shape != candidate_obs.shape or not np.array_equal(obs, candidate_obs):
            raise ValueError("validation candidate/label observation drift")
        if row.get("policy_actor_sha256") != protocol["policy_actor_sha256"]:
            raise ValueError("validation label actor identity drift")
        if row.get("policy_payload_sha256") != protocol["policy_payload_sha256"]:
            raise ValueError("validation label payload identity drift")
        if row.get("scientific_protocol_sha256") != protocol["scientific_protocol_sha256"]:
            raise ValueError("validation label scientific protocol drift")
        if row.get("runtime_protocol_sha256") != protocol["runtime_protocol_sha256"]:
            raise ValueError("validation label runtime protocol drift")
        phase = str(row["phase"])
        phase_counts[phase] += 1
        phase_positive[phase] += label
        outcomes[str(row.get("outcome_class", ""))] += 1
        stats = parent_stats[(phase, str(row["parent_group_id"]))]
        stats["count"] += 1
        stats["positive"] += label

    if label_states != set(by_state):
        raise ValueError("validation labels do not close candidate catalog")
    if dict(phase_counts) != expected["phase_candidate_counts"]:
        raise ValueError("validation recomputed phase candidate counts drift")
    if dict(phase_positive) != expected["phase_positive_counts"]:
        raise ValueError("validation recomputed phase positive counts drift")
    if dict(outcomes) != expected["outcome_counts"]:
        raise ValueError("validation recomputed outcome counts drift")
    positive = sum(int(row["label"]) for row in labels)
    if positive != int(expected["positive_count"]):
        raise ValueError("validation recomputed positive count drift")
    if len(labels) - positive != int(expected["negative_count"]):
        raise ValueError("validation recomputed negative count drift")

    leakage = _audit_train_separation(
        train_rows,
        validation_rows_for_leakage,
        observation_atol=float(protocol["near_duplicate_audit"]["actor_observation_atol"]),
    )
    by_phase_parent = {
        phase: sorted({group for p, group in parent_stats if p == phase})
        for phase in ("upstream", "downstream")
    }
    return {
        "observation_size": int(observation_size or 0),
        "validation_parent_group_ids_by_phase": by_phase_parent,
        **leakage,
    }


def freeze_iteration_validation_evidence(config_path: Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_iteration_validation_evidence_config(config_path)
    protocol = dict(config["protocol"])
    source = Path(str(protocol["source_root"]))
    summary_path = source / "summary.json"
    catalog_path = source / "candidate_catalog.json"
    labels_path = source / "labels.json"
    runtime_protocol_path = source / "runtime_protocol.json"
    protocol_audit_path = source / "protocol_audit.json"
    for path in (summary_path, catalog_path, labels_path, runtime_protocol_path, protocol_audit_path):
        if not path.is_file():
            raise FileNotFoundError(f"validation source artifact missing: {path}")

    summary = _read_object(summary_path)
    _validate_summary(summary, protocol)
    runtime_protocol = _read_object(runtime_protocol_path)
    if runtime_protocol.get("repository_head") != protocol["source_repository_head"]:
        raise ValueError("validation source repository HEAD drift")
    if runtime_protocol.get("scientific_protocol_sha256") != protocol["scientific_protocol_sha256"]:
        raise ValueError("validation runtime/scientific protocol drift")
    if runtime_protocol.get("protocol_sha256") != protocol["runtime_protocol_sha256"]:
        raise ValueError("validation runtime protocol SHA drift")

    train_manifest, train_rows = load_frozen_iteration_train_evidence(
        Path(str(protocol["frozen_train_evidence"]))
    )
    if train_manifest["manifest_sha256"] != protocol["frozen_train_manifest_sha256"]:
        raise ValueError("validation freeze frozen TRAIN manifest drift")
    if train_manifest["policy_actor_sha256"] != protocol["policy_actor_sha256"]:
        raise ValueError("validation freeze TRAIN/actor drift")
    if train_manifest["policy_payload_sha256"] != protocol["policy_payload_sha256"]:
        raise ValueError("validation freeze TRAIN/payload drift")

    catalog = _read_object(catalog_path)
    labels = _read_array(labels_path)
    row_audit = _validate_rows(
        catalog=catalog,
        labels=labels,
        protocol=protocol,
        train_rows=train_rows,
    )

    output = Path(str(config["output_dir"]))
    output.mkdir(parents=True, exist_ok=False)
    copied: dict[str, str] = {}
    for name in (
        "summary.json",
        "candidate_catalog.json",
        "labels.json",
        "runtime_protocol.json",
        "protocol_audit.json",
    ):
        src = source / name
        dst = output / name
        shutil.copyfile(src, dst)
        copied[name] = file_sha256(dst)

    manifest = {
        "schema": FROZEN_SCHEMA,
        "status": "frozen",
        "artifact_role": "frozen_group_disjoint_pi_k_expansion_validation_evidence",
        "split": "validation",
        "iteration": int(protocol["iteration"]),
        "policy_name": str(protocol["policy_name"]),
        "policy_actor_sha256": str(protocol["policy_actor_sha256"]),
        "policy_payload_sha256": str(protocol["policy_payload_sha256"]),
        "scientific_protocol_sha256": str(protocol["scientific_protocol_sha256"]),
        "runtime_protocol_sha256": str(protocol["runtime_protocol_sha256"]),
        "source_repository_head": str(protocol["source_repository_head"]),
        "source_root": str(source),
        "freeze_protocol_sha256": canonical_sha256(protocol),
        "freeze_config_file_sha256": file_sha256(config_path),
        "frozen_train_manifest_sha256": str(protocol["frozen_train_manifest_sha256"]),
        "attempt_count": int(protocol["expected"]["attempt_count"]),
        "candidate_count": int(protocol["expected"]["candidate_count"]),
        "label_count": int(protocol["expected"]["label_count"]),
        "positive_count": int(protocol["expected"]["positive_count"]),
        "negative_count": int(protocol["expected"]["negative_count"]),
        "phase_candidate_counts": dict(protocol["expected"]["phase_candidate_counts"]),
        "phase_positive_counts": dict(protocol["expected"]["phase_positive_counts"]),
        "observation_size": int(row_audit["observation_size"]),
        "validation_parent_group_ids_by_phase": row_audit["validation_parent_group_ids_by_phase"],
        "train_parent_overlap_count": int(row_audit["train_parent_overlap_count"]),
        "exact_state_overlap_count": int(row_audit["exact_state_overlap_count"]),
        "near_duplicate_overlap_count": int(row_audit["near_duplicate_overlap_count"]),
        "observation_near_duplicate_atol": float(row_audit["observation_near_duplicate_atol"]),
        "files": copied,
        "environment_interactions": 0,
        "training_transitions": 0,
        "expert_switching_used": False,
        "validation_outcomes_already_consumed": True,
        "validation_rows_may_enter_train_or_tube": False,
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


def load_frozen_iteration_validation_evidence(
    root: Path,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    root = Path(root)
    manifest = _read_object(root / "manifest.json")
    if manifest.get("schema") != FROZEN_SCHEMA or manifest.get("status") != "frozen":
        raise ValueError("invalid frozen iteration validation evidence manifest")
    declared = str(manifest.get("manifest_sha256", ""))
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if not declared or canonical_sha256(payload) != declared:
        raise ValueError("frozen validation evidence manifest SHA mismatch")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("frozen validation evidence file map missing")
    for name, expected_sha in files.items():
        if file_sha256(root / str(name)) != expected_sha:
            raise ValueError(f"frozen validation evidence file SHA mismatch: {name}")
    catalog = _read_object(root / "candidate_catalog.json")
    labels = _read_array(root / "labels.json")
    entries = catalog.get("entries")
    if not isinstance(entries, list):
        raise ValueError("frozen validation candidate entries missing")
    if len(entries) != int(manifest.get("candidate_count", -1)):
        raise ValueError("frozen validation candidate count mismatch")
    if len(labels) != int(manifest.get("label_count", -1)):
        raise ValueError("frozen validation label count mismatch")
    return manifest, tuple(dict(row) for row in entries), tuple(labels)
