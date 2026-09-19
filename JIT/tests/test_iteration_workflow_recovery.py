from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from jit_dvgc.iterative_frontier_protocol import _completed_labeling
from jit_dvgc.workflow.iteration_loop import WorkflowError, run_workflow


def test_workflow_failure_surfaces_child_stderr_and_keeps_log(tmp_path: Path) -> None:
    config_path = tmp_path / "workflow.json"
    config = {
        "schema": "jit_iteration_workflow_v1",
        "workflow_name": "stderr-surface-contract",
        "state_dir": str(tmp_path / "state"),
        "variables": {},
        "environment": {},
        "stages": [
            {
                "name": "failing_stage",
                "command": [
                    sys.executable,
                    "-c",
                    "import sys; print('frontier-inner-error', file=sys.stderr); raise SystemExit(7)",
                ],
                "cwd": str(tmp_path),
                "requires": [],
                "completion": {
                    "path": str(tmp_path / "never.json"),
                    "kind": "json",
                    "assertions": [],
                    "exports": {},
                },
            }
        ],
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(WorkflowError, match="frontier-inner-error"):
        run_workflow(config_path, execute=True)

    stderr_path = tmp_path / "state" / "logs" / "failing_stage.stderr.log"
    assert stderr_path.is_file()
    assert "frontier-inner-error" in stderr_path.read_text(encoding="utf-8")
    state = json.loads((tmp_path / "state" / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["failed_stage"] == "failing_stage"
    assert "frontier-inner-error" in state["last_error"]


def test_completed_labeling_reuses_completed_artifact(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    rows = [{"candidate_id": "a"}, {"candidate_id": "b"}]
    (labels_dir / "labels.json").write_text(json.dumps(rows), encoding="utf-8")
    summary = {
        "status": "completed",
        "candidate_count": 2,
        "environment_interactions": 123,
    }
    (labels_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    assert _completed_labeling(labels_dir) == summary


def test_previous_label_engineering_error_is_reported_not_hidden(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    summary = {
        "status": "engineering_error",
        "error": "RuntimeError: simulated GPU allocation failure",
        "completed_candidate_count": 317,
        "environment_interactions": 81234,
    }
    (labels_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(RuntimeError) as raised:
        _completed_labeling(labels_dir)

    message = str(raised.value)
    assert "simulated GPU allocation failure" in message
    assert "completed_candidate_count=317" in message
    assert "environment_interactions=81234" in message
