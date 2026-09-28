from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from jit_dvgc import phase_u_offline_evaluation as offline


def test_offline_panel_restores_named_checkpoint_and_runs_only_requested_panel(
    jit_root, tmp_path, monkeypatch
):
    source = json.loads((jit_root / "configs/phase_u_continuation_10m.json").read_text())
    step = source["formal"]["checkpoint_transitions"][1]
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(source))
    checkpoint = tmp_path / "checkpoints" / f"transition_{step}"
    checkpoint.mkdir(parents=True)
    calls = []
    env = SimpleNamespace(_bundle=SimpleNamespace(xml_sha256=source["model"]["xml_sha256"]))
    monkeypatch.setattr(offline, "TwoPhaseBikeEnv", lambda config: env)
    monkeypatch.setattr(offline, "load_checkpoint", lambda path, expected: calls.append(("load", path, expected.config_sha256)) or "payload")
    monkeypatch.setattr(offline, "make_checkpoint_policy", lambda environment, payload, deterministic: ("policy", payload, deterministic))
    monkeypatch.setattr(offline, "_evaluate_airborne_rsi_panel", lambda environment, run, config, current, make, params: calls.append(("evaluate", current, make(params, deterministic=True))) or SimpleNamespace(environment_transitions=123))
    monkeypatch.setattr(offline, "_evaluate_fixed_panel", lambda *args: pytest.fail("wrong panel"))

    count = offline.evaluate_saved_phase_u_panel(config_path, tmp_path, step, "airborne_rsi", backend_name=lambda: "gpu")
    assert count == 123
    assert calls[0][0:2] == ("load", checkpoint)
    assert calls[1] == ("evaluate", step, ("policy", "payload", True))


def test_offline_panel_rejects_step_without_saved_checkpoint(jit_root, tmp_path):
    with pytest.raises(ValueError, match="checkpoint schedule"):
        offline.evaluate_saved_phase_u_panel(
            jit_root / "configs/phase_u_continuation_10m.json", tmp_path, 123,
            "natural", backend_name=lambda: "gpu",
        )
