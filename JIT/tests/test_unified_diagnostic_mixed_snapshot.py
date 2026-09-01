from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from jit_dvgc import unified_diagnostic as diagnostic
from jit_dvgc import unified_envelope_snapshot


def _identity(path, schema: str) -> None:
    path.mkdir(parents=True)
    (path / "identity.json").write_text(
        json.dumps({"schema": schema}) + "\n",
        encoding="utf-8",
    )


def test_tube_points_accepts_legacy_and_unified_snapshot_schemas(
    tmp_path, monkeypatch
):
    legacy = tmp_path / "legacy"
    unified = tmp_path / "unified"
    _identity(legacy, "handoff_snapshot_v1")
    _identity(unified, "jit_unified_envelope_snapshot_v1")

    legacy_snapshot = SimpleNamespace(
        qpos=np.asarray([1.25, 99.0, 2.5], dtype=np.float32)
    )
    unified_snapshot = SimpleNamespace(
        qpos=np.asarray([3.75, 99.0, 4.5], dtype=np.float32)
    )

    def load_legacy(path):
        assert path == legacy
        return legacy_snapshot

    def load_unified(path):
        assert path == unified
        return unified_snapshot

    monkeypatch.setattr(diagnostic, "load_snapshot", load_legacy)
    monkeypatch.setattr(
        unified_envelope_snapshot,
        "load_unified_envelope_snapshot",
        load_unified,
    )

    artifact = SimpleNamespace(
        entries=(
            {
                "phase": "upstream",
                "snapshot": str(legacy),
                "sampling_weight": 0.25,
            },
            {
                "phase": "downstream",
                "snapshot": str(unified),
                "sampling_weight": 0.75,
            },
        )
    )

    assert diagnostic._tube_points(artifact) == (
        {
            "phase": "upstream",
            "x": 1.25,
            "z": 2.5,
            "sampling_weight": 0.25,
        },
        {
            "phase": "downstream",
            "x": 3.75,
            "z": 4.5,
            "sampling_weight": 0.75,
        },
    )


def test_tube_points_rejects_unknown_snapshot_schema(tmp_path):
    unknown = tmp_path / "unknown"
    _identity(unknown, "unknown_snapshot_v1")
    artifact = SimpleNamespace(
        entries=(
            {
                "phase": "upstream",
                "snapshot": str(unknown),
                "sampling_weight": 1.0,
            },
        )
    )

    with pytest.raises(ValueError, match="unsupported Soft Tube snapshot schema"):
        diagnostic._tube_points(artifact)
