"""Freeze merged upstream TRAIN evidence across three checkpoint domains.

This stage combines the already-frozen Iteration-0 upstream TRAIN rows from
transition_4988928 with the completed TRAIN-only parent-diversity labels from
transition_7987200 and transition_9977856.  It performs no environment step,
model fit, validation read, Tube construction, or PPO update.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .config import file_sha256
from .iteration_train_evidence import (
    canonical_sha256,
    load_frozen_iteration_train_evidence,
)

CONFIG_SCHEMA = "jit_upstream_checkpoint_train_evidence_freeze_config_v1"
PROTOCOL_SCHEMA = "jit_upstream_checkpoint_train_evidence_freeze_protocol_v1"
MANIFEST_SCHEMA = "jit_frozen_upstream_checkpoint_train_evidence_v1"
_GROUP_RE = re.compile(r"^(transition_\d+)__(\d+)$")


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


def checkpoint_domain(parent_group_id: str) -> str:
    match = _GROUP_RE.match(str(parent_group_id))
    if match is None:
        raise ValueError(f"invalid checkpoint parent group: {parent_group_id}")
    return match.group(1)


def load_upstream_checkpoint_train_freeze_config(path: Path) -> dict[str, Any]:
    config = _read_object(Path(path))
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unsupported upstream checkpoint TRAIN freeze config")
    protocol = config.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("upstream checkpoint TRAIN freeze protocol is required")
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("upstream checkpoint TRAIN freeze protocol schema drift")
    if protocol.get("status") != "predeclared_after_parent_domain_expansion_before_checkpoint_cv":
        raise ValueError("upstream checkpoint TRAIN freeze status drift")
    if int(protocol.get("iteration", -1)) != 0 or protocol.get("policy_name") != "pi_0":
        raise ValueError("upstream checkpoint TRAIN freeze policy drift")
    for field in (
        "policy_actor_sha256",
        "policy_payload_sha256",
        "base_frozen_train_manifest_sha256",
        "parent_diversity_scientific_protocol_sha256",
        "parent_diversity_acquisition_protocol_sha256",
    ):
        _sha(protocol.get(field), field=field)
    expected = protocol.get("expected_combined")
    if expected != {
        "candidate_count": 1051,
        "positive_count": 963,
        "negative_count": 88,
        "parent_group_count": 15,
        "checkpoint_domain_count": 3,
    }:
        raise ValueError("upstream checkpoint TRAIN expected combined counts drift")
    required_domains = protocol.get("required_domain_stats")
    if not isinstance(required_domains, Mapping) or set(required_domains) != {
        "transition_4988928",
        "transition_7987200",
        "transition_9977856",
    }:
        raise ValueError("upstream checkpoint TRAIN required domain contract drift")
    if protocol.get("data_policy") != {
        "train_rows_only": True,
        "consumed_validation_rows_read": False,
        "consumed_validation_predictions_read": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
    }:
        raise ValueError("upstream checkpoint TRAIN data policy drift")
    if protocol.get("claim_boundary") != {
        "checkpoint_domain_train_evidence_only": True,
        "continuation_field_reselected": False,
        "fresh_validation_bank_predeclared": False,
        "tube_1_constructed": False,
        "pi_1_trained": False,
        "jce_jel_claim": False,
        "certified_safe_set_claim": False,
    }:
        raise ValueError("upstream checkpoint TRAIN claim boundary drift")
    for field in ("base_frozen_train_evidence", "parent_diversity_root"):
        if not str(protocol.get(field, "")):
            raise ValueError(f"upstream checkpoint TRAIN {field} missing")
    if not str(config.get("output_dir", "")):
        raise ValueError("upstream checkpoint TRAIN output_dir missing")
    if canonical_sha256(protocol) != config.get("expected_protocol_sha256"):
        raise ValueError("upstream checkpoint TRAIN freeze protocol SHA drift")
    return config


def _validate_row(row: Mapping[str, Any], *, actor: str, payload: str) -> dict[str, Any]:
    value = dict(row)
    if value.get("split") != "train" or value.get("phase") != "upstream":
        raise ValueError("upstream checkpoint TRAIN accepts upstream TRAIN rows only")
    if int(value.get("phase_index", -1)) != 0:
        raise ValueError("upstream checkpoint TRAIN phase index drift")
    if value.get("policy_actor_sha256") != actor:
        raise ValueError("upstream checkpoint TRAIN actor drift")
    if value.get("policy_payload_sha256") != payload:
        raise ValueError("upstream checkpoint TRAIN payload drift")
    _sha(value.get("state_sha256"), field="upstream checkpoint TRAIN state_sha256")
    label = int(value.get("label", -1))
    if label not in (0, 1):
        raise ValueError("upstream checkpoint TRAIN label must be binary")
    group = str(value.get("parent_group_id", ""))
    domain = checkpoint_domain(group)
    observation = np.asarray(value.get("actor_observation"), dtype=np.float32).reshape(-1)
    if observation.shape != (76,) or not np.isfinite(observation).all():
        raise ValueError("upstream checkpoint TRAIN observation must be finite 76-D")
    value["parent_domain_id"] = domain
    return value


def _stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    domains = sorted({str(row["parent_domain_id"]) for row in rows})
    for domain in domains:
        subset = [row for row in rows if row["parent_domain_id"] == domain]
        positive = sum(int(row["label"]) for row in subset)
        groups = sorted({str(row["parent_group_id"]) for row in subset})
        result[domain] = {
            "candidate_count": len(subset),
            "positive_count": positive,
            "negative_count": len(subset) - positive,
            "parent_group_count": len(groups),
        }
    return result


def _validate_group_support(rows: Sequence[Mapping[str, Any]]) -> None:
    by_group: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        by_group[str(row["parent_group_id"])].add(int(row["label"]))
    bad = sorted(group for group, labels in by_group.items() if labels != {0, 1})
    if bad:
        raise ValueError(f"upstream checkpoint TRAIN groups without both labels: {bad}")


def _source_payload(config_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    config = load_upstream_checkpoint_train_freeze_config(config_path)
    protocol = config["protocol"]
    actor = str(protocol["policy_actor_sha256"])
    payload = str(protocol["policy_payload_sha256"])

    base_manifest, base_rows_all = load_frozen_iteration_train_evidence(
        Path(str(protocol["base_frozen_train_evidence"]))
    )
    if base_manifest["manifest_sha256"] != protocol["base_frozen_train_manifest_sha256"]:
        raise ValueError("base frozen TRAIN manifest drift")
    if base_manifest["policy_actor_sha256"] != actor or base_manifest["policy_payload_sha256"] != payload:
        raise ValueError("base frozen TRAIN policy identity drift")
    base_rows = [
        _validate_row(row, actor=actor, payload=payload)
        for row in base_rows_all
        if row.get("phase") == "upstream"
    ]
    base_expected = protocol["expected_base_upstream"]
    base_positive = sum(int(row["label"]) for row in base_rows)
    base_groups = {str(row["parent_group_id"]) for row in base_rows}
    if (
        len(base_rows) != int(base_expected["candidate_count"])
        or base_positive != int(base_expected["positive_count"])
        or len(base_rows) - base_positive != int(base_expected["negative_count"])
        or len(base_groups) != int(base_expected["parent_group_count"])
        or {row["parent_domain_id"] for row in base_rows} != {base_expected["checkpoint_domain"]}
    ):
        raise ValueError("base upstream TRAIN semantic counts drift")

    expansion_root = Path(str(protocol["parent_diversity_root"]))
    expansion_summary_path = expansion_root / "summary.json"
    expansion_summary = _read_object(expansion_summary_path)
    if expansion_summary.get("schema") != "jit_upstream_parent_diversity_train_summary_v1":
        raise ValueError("parent-diversity summary schema drift")
    if expansion_summary.get("status") != "completed":
        raise ValueError("parent-diversity source is not completed")
    if expansion_summary.get("scientific_protocol_sha256") != protocol["parent_diversity_scientific_protocol_sha256"]:
        raise ValueError("parent-diversity scientific protocol drift")
    if expansion_summary.get("acquisition_protocol_sha256") != protocol["parent_diversity_acquisition_protocol_sha256"]:
        raise ValueError("parent-diversity acquisition protocol drift")
    if expansion_summary.get("policy_actor_sha256") != actor or expansion_summary.get("policy_payload_sha256") != payload:
        raise ValueError("parent-diversity policy identity drift")
    for key in (
        "consumed_validation_rows_read",
        "consumed_validation_predictions_read",
        "validation_data_used",
        "test_data_used",
        "final_evaluation_data_used",
    ):
        if expansion_summary.get(key) is not False:
            raise ValueError(f"parent-diversity source {key} drift")
    if int(expansion_summary.get("training_transitions", -1)) != 0:
        raise ValueError("parent-diversity source unexpectedly trained")

    labels_dir = Path(str(expansion_summary.get("labels_dir", "")))
    if not labels_dir.is_dir():
        raise ValueError("parent-diversity completed labels directory missing")
    label_summary = _read_object(labels_dir / "summary.json")
    if label_summary.get("status") != "completed":
        raise ValueError("parent-diversity labels are not completed")
    expansion_rows = [
        _validate_row(row, actor=actor, payload=payload)
        for row in _read_array(labels_dir / "labels.json")
    ]
    exp_expected = protocol["expected_expansion"]
    exp_positive = sum(int(row["label"]) for row in expansion_rows)
    exp_groups = {str(row["parent_group_id"]) for row in expansion_rows}
    exp_domains = sorted({row["parent_domain_id"] for row in expansion_rows})
    if (
        len(expansion_rows) != int(exp_expected["candidate_count"])
        or exp_positive != int(exp_expected["positive_count"])
        or len(expansion_rows) - exp_positive != int(exp_expected["negative_count"])
        or len(exp_groups) != int(exp_expected["parent_group_count"])
        or exp_domains != list(exp_expected["checkpoint_domains"])
    ):
        raise ValueError("parent-diversity label semantic counts drift")

    combined = [
        {**row, "checkpoint_train_source": "frozen_iter0_base"} for row in base_rows
    ] + [
        {**row, "checkpoint_train_source": "parent_diversity_expansion"}
        for row in expansion_rows
    ]
    seen: set[str] = set()
    for row in combined:
        state = str(row["state_sha256"])
        if state in seen:
            raise ValueError("combined upstream checkpoint TRAIN repeats physical state")
        seen.add(state)
    _validate_group_support(combined)
    positive = sum(int(row["label"]) for row in combined)
    expected = protocol["expected_combined"]
    groups = {str(row["parent_group_id"]) for row in combined}
    domains = {str(row["parent_domain_id"]) for row in combined}
    if (
        len(combined) != int(expected["candidate_count"])
        or positive != int(expected["positive_count"])
        or len(combined) - positive != int(expected["negative_count"])
        or len(groups) != int(expected["parent_group_count"])
        or len(domains) != int(expected["checkpoint_domain_count"])
    ):
        raise ValueError("combined upstream checkpoint TRAIN counts drift")
    domain_stats = _stats(combined)
    if domain_stats != protocol["required_domain_stats"]:
        raise ValueError("combined upstream checkpoint TRAIN domain stats drift")
    source_meta = {
        "base_manifest_sha256": base_manifest["manifest_sha256"],
        "parent_diversity_summary_file_sha256": file_sha256(expansion_summary_path),
        "parent_diversity_label_summary_file_sha256": file_sha256(labels_dir / "summary.json"),
        "parent_diversity_labels_file_sha256": file_sha256(labels_dir / "labels.json"),
        "parent_diversity_labels_dir": str(labels_dir),
    }
    return config, combined, source_meta


def audit_upstream_checkpoint_train_evidence(config_path: Path) -> dict[str, Any]:
    config, rows, source_meta = _source_payload(Path(config_path))
    protocol = config["protocol"]
    return {
        "schema": "jit_upstream_checkpoint_train_evidence_audit_v1",
        "status": "ready",
        "freeze_protocol_sha256": str(config["expected_protocol_sha256"]),
        "candidate_count": len(rows),
        "positive_count": sum(int(row["label"]) for row in rows),
        "negative_count": sum(1 - int(row["label"]) for row in rows),
        "parent_group_count": len({str(row["parent_group_id"]) for row in rows}),
        "checkpoint_domains": sorted({str(row["parent_domain_id"]) for row in rows}),
        "domain_stats": _stats(rows),
        "source_files": source_meta,
        "consumed_validation_rows_read": False,
        "consumed_validation_predictions_read": False,
        "environment_interactions": 0,
        "training_transitions": 0,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "claim_boundary": dict(protocol["claim_boundary"]),
    }


def freeze_upstream_checkpoint_train_evidence(config_path: Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config, rows, source_meta = _source_payload(config_path)
    protocol = config["protocol"]
    output = Path(str(config["output_dir"]))
    output.mkdir(parents=True, exist_ok=False)
    labels_path = output / "upstream_train_labels.json"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(rows, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, labels_path)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "frozen",
        "artifact_role": "frozen_pi0_upstream_train_evidence_across_checkpoint_domains",
        "split": "train",
        "phase": "upstream",
        "iteration": 0,
        "policy_name": "pi_0",
        "policy_actor_sha256": str(protocol["policy_actor_sha256"]),
        "policy_payload_sha256": str(protocol["policy_payload_sha256"]),
        "freeze_protocol_sha256": str(config["expected_protocol_sha256"]),
        "freeze_config_file_sha256": file_sha256(config_path),
        "labels_file": "upstream_train_labels.json",
        "labels_sha256": file_sha256(labels_path),
        "candidate_count": len(rows),
        "positive_count": sum(int(row["label"]) for row in rows),
        "negative_count": sum(1 - int(row["label"]) for row in rows),
        "parent_group_count": len({str(row["parent_group_id"]) for row in rows}),
        "checkpoint_domains": sorted({str(row["parent_domain_id"]) for row in rows}),
        "domain_stats": _stats(rows),
        "observation_size": 76,
        "source_files": source_meta,
        "environment_interactions": 0,
        "training_transitions": 0,
        "consumed_validation_rows_read": False,
        "consumed_validation_predictions_read": False,
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


def load_frozen_upstream_checkpoint_train_evidence(
    root: Path,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    root = Path(root)
    manifest = _read_object(root / "manifest.json")
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("status") != "frozen":
        raise ValueError("invalid frozen upstream checkpoint TRAIN evidence manifest")
    declared = str(manifest.get("manifest_sha256", ""))
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if canonical_sha256(payload) != declared:
        raise ValueError("frozen upstream checkpoint TRAIN manifest self-hash drift")
    labels_path = root / str(manifest.get("labels_file", ""))
    if file_sha256(labels_path) != manifest.get("labels_sha256"):
        raise ValueError("frozen upstream checkpoint TRAIN labels SHA drift")
    rows = _read_array(labels_path)
    if len(rows) != int(manifest.get("candidate_count", -1)):
        raise ValueError("frozen upstream checkpoint TRAIN label count drift")
    return manifest, tuple(rows)
