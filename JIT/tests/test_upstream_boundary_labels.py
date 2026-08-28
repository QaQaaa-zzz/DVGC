from __future__ import annotations

import json

import pytest


def _row(*, seed=1000001, split="train"):
    return {
        "source_bank": "boundary_bank",
        "parent_group_id": f"transition_4988928__{seed}",
        "seed": seed,
        "role": "ascending_entry",
        "tick": 35,
        "snapshot": "snapshots/candidate_000000",
        "state_sha256": "a" * 64,
        "anchor_split": split,
    }


def test_boundary_label_catalog_accepts_train_subset(tmp_path):
    from jit_dvgc.upstream_boundary_labels import _load_train_catalog

    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"schema": "jit_upstream_boundary_catalog_v1", "entries": [_row()]}))
    rows = _load_train_catalog(catalog, (1000001, 1000002))
    assert [row["seed"] for row in rows] == [1000001]


def test_boundary_label_catalog_rejects_validation_or_test_rows(tmp_path):
    from jit_dvgc.upstream_boundary_labels import _load_train_catalog

    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"schema": "jit_upstream_boundary_catalog_v1", "entries": [_row(seed=1000006, split="validation")]}))
    with pytest.raises(ValueError, match="TRAIN rows only"):
        _load_train_catalog(catalog, (1000001, 1000002))


def test_boundary_label_catalog_rejects_parent_seed_collision(tmp_path):
    from jit_dvgc.upstream_boundary_labels import _load_train_catalog

    first = _row(seed=1000001)
    second = _row(seed=1000002)
    second["parent_group_id"] = first["parent_group_id"]
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"schema": "jit_upstream_boundary_catalog_v1", "entries": [first, second]}))
    with pytest.raises(ValueError, match="multiple numeric seeds"):
        _load_train_catalog(catalog, (1000001, 1000002))
