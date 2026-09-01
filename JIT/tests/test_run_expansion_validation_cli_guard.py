from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_cli_module():
    path = Path("JIT/cli/run_expansion_validation.py")
    spec = importlib.util.spec_from_file_location("jit_run_expansion_validation_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_machine_json_writer_is_not_stdout_dependent(tmp_path: Path) -> None:
    cli = _load_cli_module()
    target = tmp_path / "audit.json"
    payload = {"status": "fresh_validation_preflight_ready", "attempt_count": 304}
    cli._write_machine_json(target, payload)
    assert json.loads(target.read_text(encoding="utf-8")) == payload


def test_consumed_identity_audit_runs_before_any_fresh_label(monkeypatch, tmp_path: Path) -> None:
    cli = _load_cli_module()
    events: list[str] = []

    def original_label_candidates(*args, **kwargs):
        events.append("label")
        return {"status": "labels_completed"}

    def fake_execute(config, *, resume):
        events.append("execute")
        cli.fresh_shared._label_candidates(output=tmp_path)
        events.append("execute_return")
        return {"status": "completed"}

    def fake_audit(config, output):
        events.append("identity_audit")
        assert Path(output) == tmp_path
        return {
            "status": "independent",
            "exact_state_overlap_count": 0,
            "near_duplicate_observation_overlap_count": 0,
            "consumed_validation_outcomes_read": False,
            "consumed_validation_predictions_read": False,
        }

    monkeypatch.setattr(cli.fresh_shared, "_label_candidates", original_label_candidates)
    monkeypatch.setattr(cli.fresh_shared, "execute_fresh_shared_validation", fake_execute)
    monkeypatch.setattr(cli, "audit_fresh_candidate_vs_consumed_validation", fake_audit)

    report, audit = cli._execute_fresh_with_identity_guard(
        tmp_path / "config.json", resume=False
    )

    assert report == {"status": "completed"}
    assert audit["status"] == "independent"
    assert events == ["execute", "identity_audit", "label", "execute_return"]
    assert cli.fresh_shared._label_candidates is original_label_candidates
