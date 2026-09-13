from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from jit_dvgc.workflow.iteration_loop import load_workflow_config


def _load_cli():
    path = Path(__file__).resolve().parents[1] / "cli" / "sanitize_v3b_workflow.py"
    spec = importlib.util.spec_from_file_location("sanitize_v3b_workflow", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stage(name: str) -> dict:
    return {
        "name": name,
        "command": ["python", "x.py"],
        "cwd": ".",
        "requires": [],
        "completion": {
            "path": f"{name}.json",
            "kind": "json",
            "assertions": [],
            "exports": {},
        },
    }


def test_sanitizer_removes_only_known_bug_fields_and_uses_fresh_state_dir(tmp_path: Path) -> None:
    cli = _load_cli()
    payload = {
        "schema": "jit_iteration_workflow_v1",
        "workflow_name": "v3b",
        "state_dir": "old_state",
        "variables": {},
        "environment": {},
        "stages": [_stage("frontier_train")],
        "calibration_repair_plan": "repair.json",
        "calibration_repair_plan_sha256": "a" * 64,
        "source_workflow": "workflow.json",
    }
    fresh_state = tmp_path / "fresh_state"
    sanitized = cli.sanitize_workflow(payload, state_dir=fresh_state)
    assert set(sanitized) == cli.ALLOWED_TOP_LEVEL
    assert sanitized["state_dir"] == str(fresh_state)
    out = tmp_path / "workflow.json"
    out.write_text(json.dumps(sanitized), encoding="utf-8")
    config = load_workflow_config(out)
    assert config.state_dir == str(fresh_state)


def test_sanitizer_rejects_unrecognized_extra_field(tmp_path: Path) -> None:
    cli = _load_cli()
    payload = {
        "schema": "jit_iteration_workflow_v1",
        "workflow_name": "v3b",
        "state_dir": "old_state",
        "variables": {},
        "environment": {},
        "stages": [_stage("frontier_train")],
        "unexpected": 1,
    }
    try:
        cli.sanitize_workflow(payload, state_dir=tmp_path / "state")
    except ValueError as exc:
        assert "unexpected workflow fields" in str(exc)
    else:
        raise AssertionError("unexpected extra field must be rejected")
