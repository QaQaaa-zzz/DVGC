from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _row(index: int, split: str, success: int, *, actor: str) -> dict:
    seed = {"train": 1000001, "validation": 1000006, "test": 1000007}[split]
    base = -1.0 if success == 0 else 1.0
    return {
        "candidate_id": f"candidate-{split}-{index}",
        "source_bank": "handoff_bank",
        "state_sha256": f"state-{split}-{index}",
        "parent_group_id": f"parent-{split}-{index}",
        "seed": seed,
        "role": "descent_entry",
        "split": split,
        "branch_count": 1,
        "success_count": int(success),
        "actor_observation": [base, base * 0.5, float(index) * 0.01],
        "expert_actor_sha256": actor,
        "protocol_sha256": "down-protocol",
    }


def _fake_frozen(actor: str) -> dict:
    return {
        "experts": {
            "pi_down_star": {
                "actor_sha256": actor,
                "payload_sha256": "p" * 64,
                "config_sha256": "c" * 64,
                "xml_sha256": "x" * 64,
            }
        }
    }


def test_downstream_dataset_uses_train_validation_and_never_test(monkeypatch, tmp_path):
    import jit_dvgc.downstream_value as module

    actor = "d" * 64
    labels = tmp_path / "labels.json"
    _write(
        labels,
        [
            _row(0, "train", 0, actor=actor),
            _row(1, "train", 1, actor=actor),
            _row(2, "validation", 0, actor=actor),
            _row(3, "validation", 1, actor=actor),
            _row(4, "test", 0, actor=actor),
            _row(5, "test", 1, actor=actor),
        ],
    )
    monkeypatch.setattr(module, "load_frozen_manifest", lambda _path: _fake_frozen(actor))
    train, validation, provenance = module.build_downstream_value_datasets(
        labels, tmp_path / "frozen.json"
    )
    assert train.count == 2
    assert validation.count == 2
    assert provenance["declared_test_count"] == 2
    assert provenance["test_data_used"] is False
    assert {row["split"] for row in train.metadata} == {"train"}
    assert {row["split"] for row in validation.metadata} == {"validation"}
    assert not any(row["seed"] == 1000007 for row in (*train.metadata, *validation.metadata))


def test_downstream_first_pass_trains_and_writes_no_test_predictions(monkeypatch, tmp_path):
    import jit_dvgc.downstream_value as module

    actor = "d" * 64
    labels = tmp_path / "labels.json"
    rows = []
    for split, count in (("train", 12), ("validation", 6), ("test", 6)):
        for index in range(count):
            rows.append(_row(index, split, index % 2, actor=actor))
    _write(labels, rows)
    monkeypatch.setattr(module, "load_frozen_manifest", lambda _path: _fake_frozen(actor))
    output = tmp_path / "model"
    report = module.train_downstream_value_model(
        labels,
        tmp_path / "frozen.json",
        output,
        hidden_sizes=(8, 8),
        steps=100,
        learning_rate=1e-2,
        weight_decay=0.0,
        seed=7,
    )
    assert report["manifest"]["target"] == "V_down"
    assert report["manifest"]["test_data_used"] is False
    assert report["manifest"]["environment_interactions"] == 0
    predictions = json.loads((output / "validation_predictions.json").read_text())
    assert len(predictions) == 6
    assert {row["split"] for row in predictions} == {"validation"}
    assert report["metrics"]["validation"]["roc_auc"] >= 0.9
