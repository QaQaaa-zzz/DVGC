from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

import jit_dvgc.iterative_tube as iterative_tube
from jit_dvgc.iterative_frontier_protocol import _frontier_pool
from jit_dvgc.soft_tube import SoftTubeArtifact
from jit_dvgc.workflow.iteration_loop import WorkflowError, run_workflow


def _source_tube(entries: tuple[dict, ...], *, core_retained_count: int) -> SoftTubeArtifact:
    return SoftTubeArtifact(
        root=Path("unused"),
        manifest={
            "iteration": 1,
            "manifest_sha256": "a" * 64,
            "core_retained_count": core_retained_count,
        },
        entries=entries,
        diagnostics={},
    )


def test_frontier_pool_uses_only_newest_expansion_shell() -> None:
    entries = (
        {"phase": "upstream", "state_sha256": "core-up", "parent_group_id": "g0"},
        {"phase": "downstream", "state_sha256": "core-down", "parent_group_id": "g1"},
        {"phase": "upstream", "state_sha256": "new-up-0", "parent_group_id": "g2"},
        {"phase": "downstream", "state_sha256": "new-down-0", "parent_group_id": "g3"},
        {"phase": "upstream", "state_sha256": "new-up-1", "parent_group_id": "g4"},
        {"phase": "downstream", "state_sha256": "new-down-1", "parent_group_id": "g5"},
    )
    pool = _frontier_pool(_source_tube(entries, core_retained_count=2))

    assert [row[2]["state_sha256"] for row in pool["upstream"]] == ["new-up-0", "new-up-1"]
    assert [row[2]["state_sha256"] for row in pool["downstream"]] == ["new-down-0", "new-down-1"]
    assert all(
        row[2]["state_sha256"] not in {"core-up", "core-down"}
        for rows in pool.values()
        for row in rows
    )


def test_frontier_pool_refuses_iteration_without_two_phase_new_shell() -> None:
    entries = (
        {"phase": "upstream", "state_sha256": "core-up", "parent_group_id": "g0"},
        {"phase": "downstream", "state_sha256": "core-down", "parent_group_id": "g1"},
        {"phase": "upstream", "state_sha256": "new-up", "parent_group_id": "g2"},
    )
    with pytest.raises(ValueError, match="newest Tube shell has no downstream support"):
        _frontier_pool(_source_tube(entries, core_retained_count=2))


def test_iterative_tube_retains_entire_source_tube_and_adds_train_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_entries = (
        {
            "phase": "upstream",
            "phase_index": 0,
            "state_sha256": "core-up",
            "parent_group_id": "core-group-up",
            "value_score": 0.8,
            "sampling_weight": 0.81,
        },
        {
            "phase": "downstream",
            "phase_index": 1,
            "state_sha256": "core-down",
            "parent_group_id": "core-group-down",
            "value_score": 0.9,
            "sampling_weight": 0.91,
        },
    )
    source = _source_tube(source_entries, core_retained_count=1)
    train_manifest = {
        "iteration": 1,
        "source_tube_manifest_sha256": source.manifest["manifest_sha256"],
        "role_manifest_sha256": "b" * 64,
        "source_acquisition_catalog": str(tmp_path / "acquisition" / "catalog.json"),
    }
    train_rows = (
        {
            "candidate_id": "train-up",
            "phase": "upstream",
            "phase_index": 0,
            "label": 1,
            "state_sha256": "exp-up",
            "parent_group_id": "frontier-up",
            "parent_state_sha256": "parent-up",
            "source_bank": "bank",
            "snapshot": "snap-up",
            "policy_actor_sha256": "c" * 64,
            "policy_payload_sha256": "d" * 64,
        },
        {
            "candidate_id": "train-down",
            "phase": "downstream",
            "phase_index": 1,
            "label": 1,
            "state_sha256": "exp-down",
            "parent_group_id": "frontier-down",
            "parent_state_sha256": "parent-down",
            "source_bank": "bank",
            "snapshot": "snap-down",
            "policy_actor_sha256": "c" * 64,
            "policy_payload_sha256": "d" * 64,
        },
    )
    fields_summary = {
        "iteration": 1,
        "source_tube_manifest_sha256": source.manifest["manifest_sha256"],
        "train_role_manifest_sha256": train_manifest["role_manifest_sha256"],
        "calibration_role_manifest_sha256": "e" * 64,
        "summary_sha256": "f" * 64,
    }
    fields = {
        phase: {
            "field_path": tmp_path / f"{phase}.npz",
            "manifest": {
                "manifest_sha256": ("1" if phase == "upstream" else "2") * 64,
                "field_file_sha256": ("3" if phase == "upstream" else "4") * 64,
            },
            "calibration": {"acceptance_threshold_exclusive": 0.5},
        }
        for phase in ("upstream", "downstream")
    }

    snapshot_root = tmp_path / "acquisition" / "bank"
    (snapshot_root / "snap-up").mkdir(parents=True)
    (snapshot_root / "snap-down").mkdir(parents=True)

    monkeypatch.setattr(iterative_tube, "load_soft_tube", lambda _path: source)
    monkeypatch.setattr(
        iterative_tube,
        "_load_train_role",
        lambda _root: (train_manifest, train_rows),
    )
    monkeypatch.setattr(
        iterative_tube,
        "_load_fields",
        lambda _root: (fields_summary, fields),
    )
    monkeypatch.setattr(
        iterative_tube,
        "_score",
        lambda _path, rows: np.full(len(rows), 0.9),
    )

    def fake_snapshot(path: Path):
        is_up = Path(path).name == "snap-up"
        return SimpleNamespace(
            active_phase=0 if is_up else 1,
            state_sha256="exp-up" if is_up else "exp-down",
        )

    monkeypatch.setattr(iterative_tube, "load_unified_envelope_snapshot", fake_snapshot)
    monkeypatch.setattr(
        iterative_tube,
        "physical_state_sha256",
        lambda snapshot: snapshot.state_sha256,
    )

    output = tmp_path / "tube2"
    result = iterative_tube.build_iterative_tube(
        source_tube=tmp_path / "source-tube",
        train_root=tmp_path / "train",
        fields_root=tmp_path / "fields",
        output_dir=output,
    )

    entries = json.loads((output / "entries.json").read_text(encoding="utf-8"))
    manifest = result["manifest"]
    assert entries[: len(source_entries)] == list(source_entries)
    assert {row["state_sha256"] for row in entries[len(source_entries) :]} == {
        "exp-up",
        "exp-down",
    }
    assert all(row["split"] == "train" for row in entries[len(source_entries) :])
    assert manifest["core_retained_count"] == len(source_entries)
    assert manifest["source_tube_entry_count"] == len(source_entries)
    assert manifest["expansion_count"] == 2
    assert manifest["calibration_rows_embedded"] is False
    assert manifest["acceptance_rows_embedded"] is False
    assert manifest["test_data_used"] is False


def test_workflow_stops_before_next_stage_when_scientific_gate_fails(tmp_path: Path) -> None:
    gate = tmp_path / "gate.json"
    should_not_run = tmp_path / "should_not_run.txt"
    config_path = tmp_path / "workflow.json"
    gate_json = json.dumps({"status": "completed", "iteration_accepted": False}) + "\n"
    gate_json_template = gate_json.replace("{", "{{").replace("}", "}}")
    config = {
        "schema": "jit_iteration_workflow_v1",
        "workflow_name": "gate-stop-contract",
        "state_dir": str(tmp_path / "state"),
        "variables": {},
        "environment": {},
        "stages": [
            {
                "name": "scientific_gate",
                "command": [
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; "
                        f"Path({str(gate)!r}).write_text({gate_json_template!r}, encoding='utf-8')"
                    ),
                ],
                "cwd": str(tmp_path),
                "requires": [],
                "completion": {
                    "path": str(gate),
                    "kind": "json",
                    "assertions": [
                        {"pointer": "/status", "op": "eq", "value": "completed"},
                        {"pointer": "/iteration_accepted", "op": "eq", "value": True},
                    ],
                    "exports": {},
                },
            },
            {
                "name": "forbidden_after_failed_gate",
                "command": [
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; "
                        f"Path({str(should_not_run)!r}).write_text('ran', encoding='utf-8')"
                    ),
                ],
                "cwd": str(tmp_path),
                "requires": [],
                "completion": {
                    "path": str(should_not_run),
                    "kind": "file",
                    "assertions": [],
                    "exports": {},
                },
            },
        ],
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(WorkflowError, match="artifact assertion failed"):
        run_workflow(config_path, execute=True)

    assert not should_not_run.exists()
    state = json.loads((tmp_path / "state" / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["failed_stage"] == "scientific_gate"
    assert state["completed_stages"] == []
