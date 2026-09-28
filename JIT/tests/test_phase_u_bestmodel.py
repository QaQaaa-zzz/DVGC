from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from jit_dvgc.phase_u_bestmodel import update_best_model


def _panel(root: Path, kind: str, step: int, returns: tuple[float, float]) -> None:
    directory = root / kind / f"transition_{step}"
    directory.mkdir(parents=True)
    artifacts = []
    for seed, value in zip((11, 12), returns, strict=True):
        path = directory / f"seed_{seed}.npz"
        np.savez_compressed(path, reward=np.array([0.0, value / 2, value / 2]))
        artifacts.append({"seed": seed, "npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (directory / "summary.json").write_text(json.dumps({
        "absolute_transition": step, "held_out_seeds": [11, 12],
        "trace_artifacts": artifacts,
    }))


def _checkpoint(root: Path, step: int) -> None:
    path = root / "checkpoints" / f"transition_{step}"
    path.mkdir(parents=True)
    payload = path / "payload.pkl"
    payload.write_bytes(f"model {step}".encode())
    (path / "identity.json").write_text(json.dumps({
        "training_transitions": step,
        "payload_sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
    }))


def test_best_model_uses_actual_fixed_episode_returns_and_updates_link(tmp_path):
    for step, natural, airborne in (
        (100, (1.0, 3.0), (5.0, 7.0)),
        (200, (2.0, 4.0), (6.0, 8.0)),
    ):
        _checkpoint(tmp_path, step)
        _panel(tmp_path, "evaluations", step, natural)
        _panel(tmp_path, "diagnostics/airborne_rsi", step, airborne)
    report = update_best_model(tmp_path)
    assert report["training_transitions"] == 200
    assert report["mean_episode_return"] == 5.0
    assert (tmp_path / "bestmodel" / "model" / "payload.pkl").read_bytes() == b"model 200"
    assert len(report["candidates"]) == 2
    assert [row["training_transitions"] for row in report["candidates"]] == [100, 200]


def test_best_model_lists_candidates_in_numeric_transition_order(tmp_path):
    for step in (1000, 200):
        _checkpoint(tmp_path, step)
        _panel(tmp_path, "evaluations", step, (1.0, 3.0))
        _panel(tmp_path, "diagnostics/airborne_rsi", step, (5.0, 7.0))
    report = update_best_model(tmp_path)
    assert [row["training_transitions"] for row in report["candidates"]] == [200, 1000]


def test_best_model_ignores_incomplete_panel(tmp_path):
    _checkpoint(tmp_path, 100)
    _panel(tmp_path, "evaluations", 100, (1.0, 3.0))
    assert update_best_model(tmp_path) is None
    assert not (tmp_path / "bestmodel").exists()


def test_best_model_rejects_unpaired_seeds(tmp_path):
    _checkpoint(tmp_path, 100)
    _panel(tmp_path, "evaluations", 100, (1.0, 3.0))
    _panel(tmp_path, "diagnostics/airborne_rsi", 100, (5.0, 7.0))
    summary = tmp_path / "diagnostics/airborne_rsi/transition_100/summary.json"
    payload = json.loads(summary.read_text())
    payload["held_out_seeds"] = [12, 11]
    summary.write_text(json.dumps(payload))
    import pytest
    with pytest.raises(ValueError, match="paired panel seeds differ"):
        update_best_model(tmp_path)
