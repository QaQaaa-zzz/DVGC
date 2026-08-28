from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _snapshot(value=0.0, tick=34):
    return SimpleNamespace(
        qpos=np.asarray([0.0, 0.0, 0.14 + value, 0.0, 0.035], dtype=np.float32),
        qvel=np.asarray([0.73, 0.0, 0.15, 0.0, 0.91 + value], dtype=np.float32),
        observation=np.asarray([value, 1.0, 2.0], dtype=np.float32),
        tick=tick,
    )


def _make_nominal(tmp_path: Path):
    from jit_dvgc.upstream_boundary import file_sha256, row_identity

    entries = []
    labels = []
    for seed, split in [(1000001, "train"), (1000006, "validation"), (1000007, "test")]:
        row = {
            "source_bank": "source_4988928",
            "parent_group_id": f"transition_4988928__{seed}",
            "seed": seed,
            "role": "ascending_entry",
            "tick": 34,
            "state_sha256": f"state-{seed}",
            "snapshot": f"snapshots/state-{seed}",
            "source_training_transitions": 4988928,
            "source_checkpoint": "checkpoint",
            "source_actor_sha256": "a" * 64,
        }
        entries.append(row)
        labels.append(
            {
                **{key: row[key] for key in ("source_bank", "parent_group_id", "seed", "role", "tick", "state_sha256", "snapshot")},
                "split": split,
                "branch_count": 1,
                "success_count": 0,
                "branches": [{"reason": "pitch_limit"}],
            }
        )
        assert row_identity(row) == row_identity(labels[-1])
    catalog = tmp_path / "nominal" / "catalog.json"
    label_path = tmp_path / "nominal" / "labels.json"
    _write(catalog, {"entries": entries})
    _write(label_path, labels)
    return catalog, label_path, file_sha256(catalog), file_sha256(label_path)


def _make_train_protocol(tmp_path: Path, catalog_sha: str, labels_sha: str):
    from jit_dvgc.upstream_boundary import canonical_sha256

    protocol = {
        "schema": "jit_upstream_boundary_protocol_v1",
        "target": "V_up",
        "split": "train",
        "protocol_seed": 820403,
        "frozen_pi_up_actor_sha256": "b" * 64,
        "frozen_pi_up_payload_sha256": "c" * 64,
        "frozen_pi_up_config_sha256": "d" * 64,
        "xml_sha256": "e" * 64,
        "nominal_catalog_sha256": catalog_sha,
        "nominal_labels_sha256": labels_sha,
        "source_training_transitions": 4988928,
        "negative_role": "ascending_entry",
        "failure_reason": "pitch_limit",
        "direction_family": "action_basis_subset",
        "selected_action_names": ["hip"],
        "selected_signs": [1],
        "strengths": [0.05, 0.1],
        "durations": [1],
        "near_duplicate_tolerances": {"qpos_atol": 5e-4, "qvel_atol": 2e-3, "observation_atol": 1e-2},
        "state_generation": "real restore + bounded action perturbation + env.step",
        "training_transitions": 0,
    }
    protocol["protocol_sha256"] = canonical_sha256(protocol)
    path = tmp_path / "train" / "protocol.json"
    _write(path, protocol)
    return path


def _make_analysis(tmp_path: Path, *, ready=True):
    path = tmp_path / "train" / "analysis.json"
    _write(
        path,
        {
            "schema": "jit_upstream_boundary_analysis_v1",
            "target": "V_up",
            "split": "train",
            "boundary_evidence": True,
            "dataset_lock_ready": bool(ready),
        },
    )
    return path


def test_lock_requires_ready_train_analysis(tmp_path):
    from jit_dvgc.upstream_boundary_lock import build_boundary_lock

    catalog, labels, catalog_sha, labels_sha = _make_nominal(tmp_path)
    protocol = _make_train_protocol(tmp_path, catalog_sha, labels_sha)
    analysis = _make_analysis(tmp_path, ready=False)
    with pytest.raises(ValueError, match="dataset_lock_ready"):
        build_boundary_lock(protocol, analysis)


def test_lock_freezes_train_recipe_and_split_policy(tmp_path):
    from jit_dvgc.upstream_boundary_lock import write_boundary_lock, load_boundary_lock

    _, _, catalog_sha, labels_sha = _make_nominal(tmp_path)
    protocol = _make_train_protocol(tmp_path, catalog_sha, labels_sha)
    analysis = _make_analysis(tmp_path, ready=True)
    path = tmp_path / "lock.json"
    lock = write_boundary_lock(protocol, analysis, path)
    loaded = load_boundary_lock(path)
    assert loaded == lock
    assert lock["selected_action_names"] == ["hip"]
    assert lock["selected_signs"] == [1]
    assert lock["validation_seeds"] == [1000006]
    assert lock["test_seeds"] == [1000007, 1000008]
    assert lock["training_transitions"] == 0


def test_validation_selector_uses_validation_seed_only(monkeypatch, tmp_path):
    import jit_dvgc.upstream_boundary_validation as module
    from jit_dvgc.upstream_boundary_lock import write_boundary_lock

    catalog, labels, catalog_sha, labels_sha = _make_nominal(tmp_path)
    protocol = _make_train_protocol(tmp_path, catalog_sha, labels_sha)
    analysis = _make_analysis(tmp_path)
    lock_path = tmp_path / "lock.json"
    write_boundary_lock(protocol, analysis, lock_path)
    monkeypatch.setattr(module, "load_snapshot", lambda _path: _snapshot())

    anchors, audit = module.select_locked_validation_anchors(catalog, labels, lock_path)
    assert len(anchors) == 1
    assert anchors[0].row["seed"] == 1000006
    assert anchors[0].label["split"] == "validation"
    assert audit["validation_seeds"] == [1000006]
    assert audit["test_seed_interaction_count"] == 0


def test_validation_selector_rejects_nominal_identity_change(monkeypatch, tmp_path):
    import jit_dvgc.upstream_boundary_validation as module
    from jit_dvgc.upstream_boundary_lock import write_boundary_lock

    catalog, labels, catalog_sha, labels_sha = _make_nominal(tmp_path)
    protocol = _make_train_protocol(tmp_path, catalog_sha, labels_sha)
    analysis = _make_analysis(tmp_path)
    lock_path = tmp_path / "lock.json"
    write_boundary_lock(protocol, analysis, lock_path)
    payload = json.loads(labels.read_text())
    payload[0]["tick"] = 999
    _write(labels, payload)
    monkeypatch.setattr(module, "load_snapshot", lambda _path: _snapshot())
    with pytest.raises(ValueError, match=r"differ(?:s)? from the locked TRAIN identity"):
        module.select_locked_validation_anchors(catalog, labels, lock_path)


def test_locked_validation_collection_preserves_validation_split(monkeypatch, tmp_path):
    import jit_dvgc.upstream_boundary_validation as module
    from jit_dvgc.upstream_boundary import AuditedAnchor
    from jit_dvgc.upstream_boundary_lock import write_boundary_lock

    catalog, labels, catalog_sha, labels_sha = _make_nominal(tmp_path)
    protocol = _make_train_protocol(tmp_path, catalog_sha, labels_sha)
    analysis = _make_analysis(tmp_path)
    lock_path = tmp_path / "lock.json"
    lock = write_boundary_lock(protocol, analysis, lock_path)
    monkeypatch.setattr(module, "save_snapshot", lambda *_args, **_kwargs: None)

    row = json.loads(catalog.read_text())["entries"][1]
    label = json.loads(labels.read_text())[1]
    anchor = AuditedAnchor(row, label, _snapshot(), "failure_anchor")

    class Events:
        apex_seen = False

    class State:
        def __init__(self, tick=34):
            self.tick = tick
            self.info = {"events": Events(), "terminated": False, "truncated": False, "timeout": False}

    counter = {"n": 0}
    def capture(state, _anchor):
        counter["n"] += 1
        return _snapshot(counter["n"] * 0.001, tick=state.tick)

    report = module.collect_locked_validation_candidates(
        [anchor],
        tmp_path / "validation",
        lock_path=lock_path,
        restore=lambda _snapshot: State(),
        policy_action=lambda _state, _variant, _step: np.zeros(4, dtype=np.float32),
        step=lambda state, _action: State(state.tick + 1),
        capture=capture,
    )
    assert report["split"] == "validation"
    assert report["attempted_candidate_count"] == 2
    assert report["candidate_count"] == 2
    assert report["test_interaction_count"] == 0
    assert all(row["anchor_split"] == "validation" for row in report["entries"])
    assert all(row["lock_sha256"] == lock["lock_sha256"] for row in report["entries"])
