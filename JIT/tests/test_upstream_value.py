from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _label(index: int, split: str, success: int, *, actor: str, boundary_protocol="", lock_sha=""):
    row = {
        "candidate_id": f"candidate-{split}-{index}",
        "state_sha256": f"state-{split}-{index}",
        "parent_group_id": f"parent-{split}-{index // 2}",
        "seed": {"train": 1000001, "validation": 1000006, "test": 1000007}[split],
        "role": "ascending_entry",
        "split": split,
        "branch_count": 1,
        "success_count": int(success),
        "actor_observation": [float(index), float(success), 1.0],
        "expert_actor_sha256": actor,
    }
    if boundary_protocol:
        row["boundary_protocol_sha256"] = boundary_protocol
    if lock_sha:
        row["lock_sha256"] = lock_sha
    return row


def _lock(tmp_path: Path):
    from jit_dvgc.upstream_boundary import canonical_sha256

    lock = {
        "schema": "jit_upstream_boundary_lock_v1",
        "status": "locked",
        "target": "V_up",
        "train_protocol_sha256": "train-protocol",
        "frozen_pi_up_actor_sha256": "a" * 64,
        "validation_seeds": [1000006],
        "test_seeds": [1000007, 1000008],
    }
    lock["lock_sha256"] = canonical_sha256(lock)
    path = tmp_path / "lock.json"
    _write(path, lock)
    return path, lock


def test_binary_metrics_reports_ranking_and_calibration():
    from jit_dvgc.upstream_value import binary_metrics

    metrics = binary_metrics(np.asarray([0, 0, 1, 1]), np.asarray([0.1, 0.2, 0.8, 0.9]))
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["accuracy_at_0_5"] == pytest.approx(1.0)
    assert metrics["positive_negative_score_gap"] > 0.0
    assert metrics["brier"] < 0.05


def test_value_dataset_uses_train_and_validation_but_never_test(tmp_path):
    from jit_dvgc.upstream_value import build_upstream_value_datasets

    lock_path, lock = _lock(tmp_path)
    actor = lock["frozen_pi_up_actor_sha256"]
    nominal = tmp_path / "nominal.json"
    boundary_train = tmp_path / "boundary_train.json"
    boundary_validation = tmp_path / "boundary_validation.json"
    _write(
        nominal,
        [
            _label(0, "train", 0, actor=actor),
            _label(1, "train", 1, actor=actor),
            _label(2, "validation", 0, actor=actor),
            _label(3, "validation", 1, actor=actor),
            _label(4, "test", 0, actor=actor),
            _label(5, "test", 1, actor=actor),
        ],
    )
    _write(
        boundary_train,
        [
            _label(10, "train", 0, actor=actor, boundary_protocol=lock["train_protocol_sha256"]),
            _label(11, "train", 1, actor=actor, boundary_protocol=lock["train_protocol_sha256"]),
        ],
    )
    _write(
        boundary_validation,
        [
            _label(20, "validation", 0, actor=actor, lock_sha=lock["lock_sha256"]),
            _label(21, "validation", 1, actor=actor, lock_sha=lock["lock_sha256"]),
        ],
    )

    train, validation, provenance = build_upstream_value_datasets(
        nominal, boundary_train, boundary_validation, lock_path
    )
    assert train.count == 4
    assert validation.count == 4
    assert provenance["test_data_used"] is False
    assert {row["split"] for row in train.metadata} == {"train"}
    assert {row["split"] for row in validation.metadata} == {"validation"}
    assert not any(row["seed"] == 1000007 for row in (*train.metadata, *validation.metadata))


def test_value_dataset_rejects_wrong_boundary_protocol(tmp_path):
    from jit_dvgc.upstream_value import build_upstream_value_datasets

    lock_path, lock = _lock(tmp_path)
    actor = lock["frozen_pi_up_actor_sha256"]
    nominal = tmp_path / "nominal.json"
    boundary_train = tmp_path / "boundary_train.json"
    boundary_validation = tmp_path / "boundary_validation.json"
    _write(nominal, [_label(0, "train", 0, actor=actor), _label(1, "train", 1, actor=actor), _label(2, "validation", 0, actor=actor), _label(3, "validation", 1, actor=actor)])
    _write(boundary_train, [_label(10, "train", 0, actor=actor, boundary_protocol="wrong"), _label(11, "train", 1, actor=actor, boundary_protocol="wrong")])
    _write(boundary_validation, [_label(20, "validation", 0, actor=actor, lock_sha=lock["lock_sha256"]), _label(21, "validation", 1, actor=actor, lock_sha=lock["lock_sha256"])])
    with pytest.raises(ValueError, match="locked TRAIN protocol"):
        build_upstream_value_datasets(nominal, boundary_train, boundary_validation, lock_path)
