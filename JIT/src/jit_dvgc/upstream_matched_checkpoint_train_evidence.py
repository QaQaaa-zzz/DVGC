"""Freeze a fair three-checkpoint upstream TRAIN panel without new interactions.

This capability selects the already-reconstructed transition_4988928 matched
panel and combines it with the two newer checkpoint domains that were collected
under that exact locked panel. It never reads consumed validation rows or
predictions, steps an environment, trains a model, constructs a Tube, or runs
PPO.
"""
from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .config import file_sha256
from .iteration_train_evidence import canonical_sha256
from .upstream_checkpoint_train_evidence import (
    MANIFEST_SCHEMA,
    checkpoint_domain,
    load_frozen_upstream_checkpoint_train_evidence,
)
from .upstream_matched_panel_audit import audit_upstream_matched_panel

CONFIG_SCHEMA = "jit_upstream_matched_checkpoint_train_evidence_freeze_config_v1"
PROTOCOL_SCHEMA = "jit_upstream_matched_checkpoint_train_evidence_freeze_protocol_v1"


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _sha(value: Any, *, field: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{field} must be SHA-256")
    return text


def load_matched_checkpoint_train_freeze_config(path: Path) -> dict[str, Any]:
    config = _read_object(Path(path))
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unsupported matched checkpoint TRAIN freeze config")
    protocol = config.get("protocol")
    if not isinstance(protocol, Mapping) or protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("matched checkpoint TRAIN freeze protocol drift")
    if protocol.get("status") != "predeclared_after_matched_panel_audit_before_fair_checkpoint_cv":
        raise ValueError("matched checkpoint TRAIN freeze status drift")
    if int(protocol.get("iteration", -1)) != 0 or protocol.get("policy_name") != "pi_0":
        raise ValueError("matched checkpoint TRAIN freeze policy drift")
    for field in (
        "policy_actor_sha256",
        "policy_payload_sha256",
        "source_frozen_upstream_train_manifest_sha256",
        "source_frozen_upstream_train_labels_sha256",
        "matched_panel_audit_protocol_sha256",
    ):
        _sha(protocol.get(field), field=field)
    if protocol.get("required_domains") != [
        "transition_4988928",
        "transition_7987200",
        "transition_9977856",
    ]:
        raise ValueError("matched checkpoint TRAIN required domain drift")
    expected_domains = {
        "transition_4988928": {
            "candidate_count": 240,
            "positive_count": 221,
            "negative_count": 19,
            "parent_group_count": 5,
        },
        "transition_7987200": {
            "candidate_count": 240,
            "positive_count": 217,
            "negative_count": 23,
            "parent_group_count": 5,
        },
        "transition_9977856": {
            "candidate_count": 240,
            "positive_count": 201,
            "negative_count": 39,
            "parent_group_count": 5,
        },
    }
    if protocol.get("expected_domain_stats") != expected_domains:
        raise ValueError("matched checkpoint TRAIN domain stats drift")
    if protocol.get("expected_combined") != {
        "candidate_count": 720,
        "positive_count": 639,
        "negative_count": 81,
        "parent_group_count": 15,
        "checkpoint_domain_count": 3,
    }:
        raise ValueError("matched checkpoint TRAIN combined counts drift")
    if protocol.get("data_policy") != {
        "train_rows_only": True,
        "consumed_validation_rows_read": False,
        "consumed_validation_predictions_read": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "environment_interactions": 0,
        "training_transitions": 0,
    }:
        raise ValueError("matched checkpoint TRAIN data policy drift")
    if protocol.get("claim_boundary") != {
        "matched_checkpoint_train_evidence_only": True,
        "continuation_field_reselected": False,
        "fresh_validation_bank_predeclared": False,
        "tube_1_constructed": False,
        "pi_1_trained": False,
        "jce_jel_claim": False,
        "certified_safe_set_claim": False,
    }:
        raise ValueError("matched checkpoint TRAIN claim boundary drift")
    for field in (
        "source_frozen_upstream_train_evidence",
        "matched_panel_audit_config",
    ):
        if not str(protocol.get(field, "")):
            raise ValueError(f"matched checkpoint TRAIN {field} missing")
    if not str(config.get("output_dir", "")):
        raise ValueError("matched checkpoint TRAIN output_dir missing")
    if canonical_sha256(protocol) != config.get("expected_protocol_sha256"):
        raise ValueError("matched checkpoint TRAIN protocol SHA drift")
    return config


def _stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for domain in sorted({checkpoint_domain(str(row["parent_group_id"])) for row in rows}):
        subset = [row for row in rows if checkpoint_domain(str(row["parent_group_id"])) == domain]
        positive = sum(int(row["label"]) for row in subset)
        result[domain] = {
            "candidate_count": len(subset),
            "positive_count": positive,
            "negative_count": len(subset) - positive,
            "parent_group_count": len({str(row["parent_group_id"]) for row in subset}),
        }
    return result


def _validate_rows(rows: Sequence[Mapping[str, Any]], *, actor: str, payload: str) -> None:
    seen: set[str] = set()
    by_group: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        if row.get("split") != "train" or row.get("phase") != "upstream":
            raise ValueError("matched checkpoint TRAIN accepts upstream TRAIN rows only")
        if row.get("policy_actor_sha256") != actor or row.get("policy_payload_sha256") != payload:
            raise ValueError("matched checkpoint TRAIN policy identity drift")
        state = _sha(row.get("state_sha256"), field="matched checkpoint TRAIN state_sha256")
        if state in seen:
            raise ValueError("matched checkpoint TRAIN repeats physical state")
        seen.add(state)
        label = int(row.get("label", -1))
        if label not in (0, 1):
            raise ValueError("matched checkpoint TRAIN label must be binary")
        observation = np.asarray(row.get("actor_observation"), dtype=np.float32).reshape(-1)
        if observation.shape != (76,) or not np.isfinite(observation).all():
            raise ValueError("matched checkpoint TRAIN observation must be finite 76-D")
        by_group[str(row.get("parent_group_id", ""))].add(label)
    bad = sorted(group for group, labels in by_group.items() if labels != {0, 1})
    if bad:
        raise ValueError(f"matched checkpoint TRAIN groups without both labels: {bad}")


def _source_payload(config_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    config = load_matched_checkpoint_train_freeze_config(config_path)
    protocol = config["protocol"]
    source_root = Path(str(protocol["source_frozen_upstream_train_evidence"]))
    manifest, raw_rows = load_frozen_upstream_checkpoint_train_evidence(source_root)
    if manifest.get("manifest_sha256") != protocol["source_frozen_upstream_train_manifest_sha256"]:
        raise ValueError("matched checkpoint TRAIN source manifest drift")
    if manifest.get("labels_sha256") != protocol["source_frozen_upstream_train_labels_sha256"]:
        raise ValueError("matched checkpoint TRAIN source labels drift")
    if manifest.get("policy_actor_sha256") != protocol["policy_actor_sha256"]:
        raise ValueError("matched checkpoint TRAIN source actor drift")
    if manifest.get("policy_payload_sha256") != protocol["policy_payload_sha256"]:
        raise ValueError("matched checkpoint TRAIN source payload drift")
    for key in (
        "consumed_validation_rows_read",
        "consumed_validation_predictions_read",
        "test_data_used",
        "final_evaluation_data_used",
    ):
        if manifest.get(key) is not False:
            raise ValueError(f"matched checkpoint TRAIN source {key} drift")
    if int(manifest.get("training_transitions", -1)) != 0:
        raise ValueError("matched checkpoint TRAIN source unexpectedly trained")

    audit_config = Path(str(protocol["matched_panel_audit_config"]))
    audit = audit_upstream_matched_panel(audit_config)
    if audit.get("protocol_sha256") != protocol["matched_panel_audit_protocol_sha256"]:
        raise ValueError("matched checkpoint TRAIN audit protocol drift")
    if audit.get("matched_panel_freeze_authorized") is not True:
        raise ValueError("matched checkpoint TRAIN audit did not authorize freeze")
    if audit.get("matched_panel_complete") is not True:
        raise ValueError("matched checkpoint TRAIN audit panel incomplete")
    if audit.get("matched_panel_both_labels_in_every_parent") is not True:
        raise ValueError("matched checkpoint TRAIN audit label support incomplete")
    if int(audit.get("missing_cell_count", -1)) != 0 or int(audit.get("extra_cell_count", -1)) != 0:
        raise ValueError("matched checkpoint TRAIN audit cell mismatch")
    if int(audit.get("duplicate_cell_count", -1)) != 0:
        raise ValueError("matched checkpoint TRAIN audit duplicate cells")

    target_states = set(str(value) for value in audit["matched_state_sha256"])
    rows: list[dict[str, Any]] = []
    for source in raw_rows:
        row = dict(source)
        domain = checkpoint_domain(str(row["parent_group_id"]))
        if domain == "transition_4988928" and str(row["state_sha256"]) not in target_states:
            continue
        if domain not in protocol["required_domains"]:
            continue
        row["parent_domain_id"] = domain
        row["matched_checkpoint_panel"] = True
        rows.append(row)

    actor = str(protocol["policy_actor_sha256"])
    payload = str(protocol["policy_payload_sha256"])
    _validate_rows(rows, actor=actor, payload=payload)
    stats = _stats(rows)
    if stats != protocol["expected_domain_stats"]:
        raise ValueError("matched checkpoint TRAIN selected domain stats drift")
    expected = protocol["expected_combined"]
    positive = sum(int(row["label"]) for row in rows)
    if (
        len(rows) != int(expected["candidate_count"])
        or positive != int(expected["positive_count"])
        or len(rows) - positive != int(expected["negative_count"])
        or len({str(row["parent_group_id"]) for row in rows}) != int(expected["parent_group_count"])
        or len(stats) != int(expected["checkpoint_domain_count"])
    ):
        raise ValueError("matched checkpoint TRAIN selected counts drift")
    rows.sort(key=lambda row: (str(row["parent_domain_id"]), str(row["parent_group_id"]), str(row["state_sha256"])))
    source_meta = {
        "source_manifest_sha256": manifest["manifest_sha256"],
        "source_labels_sha256": manifest["labels_sha256"],
        "matched_panel_audit_protocol_sha256": audit["protocol_sha256"],
        "matched_panel_candidate_count": int(audit["matched_panel_candidate_count"]),
        "matched_panel_positive_count": int(audit["matched_panel_positive_count"]),
        "matched_panel_negative_count": int(audit["matched_panel_negative_count"]),
        "matched_panel_catalog_scan": audit["catalog_scan"],
    }
    return config, rows, source_meta


def audit_matched_checkpoint_train_evidence(config_path: Path) -> dict[str, Any]:
    config, rows, source_meta = _source_payload(Path(config_path))
    protocol = config["protocol"]
    return {
        "schema": "jit_upstream_matched_checkpoint_train_evidence_audit_v1",
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


def freeze_matched_checkpoint_train_evidence(config_path: Path) -> dict[str, Any]:
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
        "artifact_role": "frozen_pi0_upstream_train_evidence_matched_across_checkpoint_domains",
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
        "matched_panel": True,
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
