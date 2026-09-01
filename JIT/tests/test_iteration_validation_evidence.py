from __future__ import annotations

import json

import pytest


def test_real_validation_freeze_config_locks_completed_run(jit_root):
    from jit_dvgc.iteration_validation_evidence import (
        load_iteration_validation_evidence_config,
    )

    config = load_iteration_validation_evidence_config(
        jit_root / "configs/envelope_iter0_validation_evidence_freeze.json"
    )
    p = config["protocol"]
    assert p["scientific_protocol_sha256"] == (
        "9ec0a1e8c314cc5710688a3537fbd339f520d4db3c4268d20715bcde938586b0"
    )
    assert p["runtime_protocol_sha256"] == (
        "9cda2c58c5918db4c5791ce9d18e7689d49a50336638c2caac43ddb02e88eda3"
    )
    assert p["expected"]["candidate_count"] == 147
    assert p["expected"]["positive_count"] == 117
    assert p["expected"]["negative_count"] == 30
    assert p["expected"]["phase_candidate_counts"] == {
        "upstream": 131,
        "downstream": 16,
    }


def test_validation_row_audit_rejects_train_parent_overlap():
    from jit_dvgc.iteration_validation_evidence import _validate_rows

    actor = "a" * 64
    payload = "b" * 64
    sci = "c" * 64
    runtime = "d" * 64
    protocol = {
        "iteration": 0,
        "policy_actor_sha256": actor,
        "policy_payload_sha256": payload,
        "scientific_protocol_sha256": sci,
        "runtime_protocol_sha256": runtime,
        "expected": {
            "candidate_count": 2,
            "label_count": 2,
            "positive_count": 1,
            "negative_count": 1,
            "phase_candidate_counts": {"upstream": 1, "downstream": 1},
            "phase_positive_counts": {"upstream": 1, "downstream": 0},
            "outcome_counts": {"success": 1, "physical_failure": 1},
        },
        "near_duplicate_audit": {"actor_observation_atol": 0.01},
    }
    entries = []
    labels = []
    for idx, (phase, label, group) in enumerate(
        [("upstream", 1, "train_group"), ("downstream", 0, "valid_group")]
    ):
        state = f"{idx + 10:064x}"
        obs = [float(idx + 10), 0.0]
        entry = {
            "candidate_id": f"c{idx}",
            "split": "validation",
            "phase": phase,
            "state_sha256": state,
            "parent_group_id": group,
            "actor_observation": obs,
        }
        entries.append(entry)
        labels.append(
            {
                **entry,
                "label": label,
                "outcome_class": "success" if label else "physical_failure",
                "policy_actor_sha256": actor,
                "policy_payload_sha256": payload,
                "scientific_protocol_sha256": sci,
                "runtime_protocol_sha256": runtime,
            }
        )
    train_rows = (
        {
            "parent_group_id": "train_group",
            "state_sha256": f"{99:064x}",
            "actor_observation": [99.0, 0.0],
        },
    )
    with pytest.raises(ValueError, match="TRAIN parent group"):
        _validate_rows(
            catalog={"entries": entries},
            labels=labels,
            protocol=protocol,
            train_rows=train_rows,
        )


def test_frozen_validation_loader_detects_file_tampering(tmp_path):
    from jit_dvgc.config import file_sha256
    from jit_dvgc.iteration_train_evidence import canonical_sha256
    from jit_dvgc.iteration_validation_evidence import (
        FROZEN_SCHEMA,
        load_frozen_iteration_validation_evidence,
    )

    root = tmp_path / "frozen"
    root.mkdir()
    for name, payload in (
        ("candidate_catalog.json", {"entries": []}),
        ("labels.json", []),
        ("summary.json", {}),
        ("runtime_protocol.json", {}),
        ("protocol_audit.json", {}),
    ):
        (root / name).write_text(json.dumps(payload) + "\n")
    files = {
        name: file_sha256(root / name)
        for name in (
            "candidate_catalog.json",
            "labels.json",
            "summary.json",
            "runtime_protocol.json",
            "protocol_audit.json",
        )
    }
    manifest = {
        "schema": FROZEN_SCHEMA,
        "status": "frozen",
        "candidate_count": 0,
        "label_count": 0,
        "files": files,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest) + "\n")
    (root / "labels.json").write_text("[] \n")
    with pytest.raises(ValueError, match="file SHA mismatch"):
        load_frozen_iteration_validation_evidence(root)
