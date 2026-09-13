from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


ACTOR = "a" * 64
PAYLOAD = "b" * 64
XML = "c" * 64
TUBE = "d" * 64
PROTOCOL = "e" * 64


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _row(index: int, phase: str, label: int, group: str) -> dict:
    return {
        "candidate_id": f"candidate_{index}",
        "state_sha256": f"{index:064x}",
        "phase": phase,
        "phase_index": 0 if phase == "upstream" else 1,
        "label": label,
        "parent_group_id": group,
        "split": "train",
        "actor_observation": [float(index), 1.0],
        "policy_iteration": 0,
        "policy_actor_sha256": ACTOR,
        "policy_payload_sha256": PAYLOAD,
        "label_protocol_sha256": f"{index + 20:064x}",
        "acquisition_protocol_sha256": f"{index + 40:064x}",
    }


def _fixture(tmp_path: Path):
    rows = [
        _row(1, "upstream", 1, "up_a"),
        _row(2, "upstream", 0, "up_b"),
        _row(3, "downstream", 1, "down_a"),
        _row(4, "downstream", 0, "down_b"),
    ]
    source = tmp_path / "source"
    labels = source / "accumulated_train_labels.json"
    summary = source / "summary.json"
    _write(labels, {"entries": rows})
    readiness = {
        "upstream": {
            "candidate_count": 2,
            "positive_count": 1,
            "negative_count": 1,
            "positive_parent_group_count": 1,
            "negative_parent_group_count": 1,
            "ready": True,
        },
        "downstream": {
            "candidate_count": 2,
            "positive_count": 1,
            "negative_count": 1,
            "positive_parent_group_count": 1,
            "negative_parent_group_count": 1,
            "ready": True,
        },
    }
    _write(
        summary,
        {
            "schema": "jit_downstream_transition_refinement_summary_v1",
            "status": "transition_band_ready",
            "iteration": 0,
            "policy_actor_sha256": ACTOR,
            "policy_payload_sha256": PAYLOAD,
            "source_tube_manifest_sha256": TUBE,
            "protocol_sha256": PROTOCOL,
            "accumulated_unique_label_count": 4,
            "readiness": readiness,
            "training_transitions": 0,
            "expert_switching_used": False,
            "validation_data_used": False,
            "test_data_used": False,
            "final_evaluation_data_used": False,
            "claim_boundary": {
                "training_only_search": True,
                "upstream_transition_band_frozen": True,
                "continuation_field_trained": False,
                "tube_1_constructed": False,
                "pi_1_trained": False,
                "jce_jel_claim": False,
                "certified_safe_set_claim": False,
            },
        },
    )
    return source, labels, summary, readiness


def _config(tmp_path: Path, source: Path, labels: Path, summary: Path, readiness):
    from jit_dvgc.config import file_sha256
    from jit_dvgc.iteration_train_evidence import canonical_sha256

    protocol = {
        "schema": "jit_iteration_train_evidence_freeze_protocol_v1",
        "status": "predeclared",
        "iteration": 0,
        "policy_name": "pi_0",
        "policy_actor_sha256": ACTOR,
        "policy_payload_sha256": PAYLOAD,
        "xml_sha256": XML,
        "source_tube_manifest_sha256": TUBE,
        "source_terminal_protocol_sha256": PROTOCOL,
        "source_root": str(source),
        "source_summary_file_sha256": file_sha256(summary),
        "source_labels_file_sha256": file_sha256(labels),
        "expected_accumulated_unique_label_count": 4,
        "expected_readiness": readiness,
        "readiness_criteria": {
            "minimum_positive_candidates": 1,
            "minimum_negative_candidates": 1,
            "minimum_parent_groups_with_positive": 1,
            "minimum_parent_groups_with_negative": 1,
        },
        "claim_boundary": {
            "expansion_train_evidence_only": True,
            "continuation_field_trained": False,
            "tube_1_constructed": False,
            "pi_1_trained": False,
            "jce_jel_claim": False,
            "certified_safe_set_claim": False,
        },
    }
    config = {
        "schema": "jit_iteration_train_evidence_freeze_config_v1",
        "output_dir": str(tmp_path / "frozen"),
        "expected_protocol_sha256": canonical_sha256(protocol),
        "protocol": protocol,
    }
    path = tmp_path / "config.json"
    _write(path, config)
    return path


def test_freeze_copies_and_hash_locks_ready_train_evidence(tmp_path):
    from jit_dvgc.iteration_train_evidence import (
        freeze_iteration_train_evidence,
        load_frozen_iteration_train_evidence,
    )

    source, labels, summary, readiness = _fixture(tmp_path)
    config = _config(tmp_path, source, labels, summary, readiness)
    manifest = freeze_iteration_train_evidence(config)
    loaded, rows = load_frozen_iteration_train_evidence(tmp_path / "frozen")
    assert loaded == manifest
    assert len(rows) == 4
    assert loaded["train_parent_group_ids"] == ["down_a", "down_b", "up_a", "up_b"]
    assert loaded["phase_counts"]["upstream"]["positive_count"] == 1
    assert loaded["phase_counts"]["downstream"]["negative_count"] == 1
    assert loaded["environment_interactions"] == 0
    assert loaded["training_transitions"] == 0


def test_freeze_rejects_nontrain_or_duplicate_physical_state(tmp_path):
    from jit_dvgc.iteration_train_evidence import freeze_iteration_train_evidence

    source, labels, summary, readiness = _fixture(tmp_path)
    payload = json.loads(labels.read_text())
    payload["entries"][0]["split"] = "validation"
    _write(labels, payload)
    config = _config(tmp_path, source, labels, summary, readiness)
    with pytest.raises(ValueError, match="TRAIN-only"):
        freeze_iteration_train_evidence(config)

    payload["entries"][0]["split"] = "train"
    payload["entries"][1]["state_sha256"] = payload["entries"][0]["state_sha256"]
    _write(labels, payload)
    config = _config(tmp_path, source, labels, summary, readiness)
    with pytest.raises(ValueError, match="duplicate physical state"):
        freeze_iteration_train_evidence(config)


def test_freeze_rejects_nonfinite_observation(tmp_path):
    from jit_dvgc.iteration_train_evidence import freeze_iteration_train_evidence

    source, labels, summary, readiness = _fixture(tmp_path)
    payload = json.loads(labels.read_text())
    payload["entries"][0]["actor_observation"] = [1.0, "nan"]
    _write(labels, payload)
    config = _config(tmp_path, source, labels, summary, readiness)
    with pytest.raises(ValueError, match="finite actor observation"):
        freeze_iteration_train_evidence(config)


def test_loader_detects_copied_label_tampering(tmp_path):
    from jit_dvgc.iteration_train_evidence import (
        freeze_iteration_train_evidence,
        load_frozen_iteration_train_evidence,
    )

    source, labels, summary, readiness = _fixture(tmp_path)
    config = _config(tmp_path, source, labels, summary, readiness)
    freeze_iteration_train_evidence(config)
    frozen_labels = tmp_path / "frozen" / "train_labels.json"
    frozen_labels.write_text(frozen_labels.read_text() + " ")
    with pytest.raises(ValueError, match="labels SHA-256"):
        load_frozen_iteration_train_evidence(tmp_path / "frozen")


def test_real_freeze_config_is_bound_to_completed_3190_label_run(jit_root):
    from jit_dvgc.iteration_train_evidence import load_iteration_train_evidence_config

    config = load_iteration_train_evidence_config(
        jit_root / "configs/envelope_iter0_train_evidence_freeze.json"
    )
    protocol = config["protocol"]
    assert protocol["expected_accumulated_unique_label_count"] == 3190
    assert protocol["expected_readiness"]["upstream"]["ready"] is True
    assert protocol["expected_readiness"]["downstream"]["ready"] is True
    assert protocol["expected_readiness"]["downstream"]["negative_count"] == 30
    assert protocol["claim_boundary"]["continuation_field_trained"] is False
