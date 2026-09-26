from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from jit_dvgc.iterative_frontier_protocol import canonical_sha256


def _load_cli():
    path = Path(__file__).resolve().parents[1] / "cli" / "calibration_hard_negative_repair.py"
    spec = importlib.util.spec_from_file_location("calibration_hard_negative_repair", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v3b_panel_is_fixed_sparse_two_axis_historical_family() -> None:
    cli = _load_cli()
    assert cli.UPSTREAM_PANEL == {
        "action_names": ["steer", "rear_wheel_drive", "hip", "knee"],
        "signs": [-1, 1],
        "strengths": [0.15, 0.30, 0.50],
        "durations": [2, 4, 8],
        "active_action_dimensions": 2,
    }
    assert cli.MIN_SUPPLEMENT_NEGATIVES == 5
    assert cli.MIN_SUPPLEMENT_NEGATIVE_PARENT_GROUPS == 2


def test_phase_counts_tracks_negative_parent_support() -> None:
    cli = _load_cli()
    rows = [
        {"phase": "upstream", "label": 1, "parent_group_id": "a"},
        {"phase": "upstream", "label": 0, "parent_group_id": "a"},
        {"phase": "upstream", "label": 0, "parent_group_id": "b"},
        {"phase": "downstream", "label": 0, "parent_group_id": "d"},
    ]
    assert cli._phase_counts(rows, "upstream") == {
        "candidate_count": 3,
        "positive_count": 1,
        "negative_count": 2,
        "parent_group_count": 2,
        "negative_parent_group_count": 2,
    }


def test_prepare_preserves_roles_and_rewrites_only_calibration_consumer_path(tmp_path: Path) -> None:
    cli = _load_cli()
    source_plan = tmp_path / "frontier_plan.json"
    failed_root = tmp_path / "frontier_calibration"
    failed_root.mkdir()
    source_workflow = tmp_path / "workflow.json"
    repair_plan = tmp_path / "calibration_repair_plan.json"
    repaired_root = tmp_path / "frontier_calibration_v3b_repaired"
    workflow_out = tmp_path / "workflow_v3b.json"

    actor = "a" * 64
    payload = "b" * 64
    anchors = []
    for index in range(3):
        anchors.append(
            {
                "role": "calibration",
                "phase": "upstream",
                "phase_index": 0,
                "entry_index": index,
                "global_index": index,
                "state_sha256": f"u{index}".ljust(64, "0"),
                "parent_group_id": f"ug{index}",
                "value_score": 0.935 + index * 0.001,
                "sampling_weight": 1.0,
            }
        )
    anchors.append(
        {
            "role": "calibration",
            "phase": "downstream",
            "phase_index": 1,
            "entry_index": 0,
            "global_index": 100,
            "state_sha256": "d" * 64,
            "parent_group_id": "dg0",
            "value_score": 0.94,
            "sampling_weight": 1.0,
        }
    )
    plan = {
        "schema": "jit_iterative_frontier_plan_v1",
        "status": "predeclared_before_frontier_outcomes",
        "iteration": 1,
        "policy_name": "pi_1",
        "selected_policy": "selected.json",
        "selected_policy_sha256": "c" * 64,
        "policy_actor_sha256": actor,
        "policy_payload_sha256": payload,
        "source_tube": "tube1",
        "source_tube_manifest_sha256": "d" * 64,
        "source_tube_entry_count": 3119,
        "source_tube_core_retained_count": 222,
        "frontier_definition": "v3",
        "role_pattern": ["train", "train", "train", "calibration", "acceptance"],
        "role_semantics": {},
        "role_parent_group_counts": {},
        "fixed_probe_panel": {"max_label_ticks": 400},
        "seeds": {},
        "anchors": anchors,
        "protocol_revision": {"name": "phase_specific_two_axis_v3"},
        "training_transitions": 0,
        "expert_switching_used": False,
        "validation_data_used": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "claim_boundary": {},
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    source_plan.write_text(json.dumps(plan), encoding="utf-8")

    rows = []
    for index in range(3):
        rows.append(
            {
                "candidate_id": f"u{index}",
                "phase": "upstream",
                "label": 1,
                "parent_group_id": f"ug{index}",
                "policy_actor_sha256": actor,
                "policy_payload_sha256": payload,
                "split": "calibration",
                "logical_role": "calibration",
            }
        )
    rows.extend(
        [
            {
                "candidate_id": "dpos",
                "phase": "downstream",
                "label": 1,
                "parent_group_id": "dg0",
                "policy_actor_sha256": actor,
                "policy_payload_sha256": payload,
                "split": "calibration",
                "logical_role": "calibration",
            },
            {
                "candidate_id": "dneg",
                "phase": "downstream",
                "label": 0,
                "parent_group_id": "dg0",
                "policy_actor_sha256": actor,
                "policy_payload_sha256": payload,
                "split": "calibration",
                "logical_role": "calibration",
            },
        ]
    )
    logical = {
        "schema": "jit_iterative_frontier_logical_labels_v1",
        "role": "calibration",
        "entries": rows,
    }
    logical["labels_sha256"] = canonical_sha256(logical)
    (failed_root / "logical_labels.json").write_text(json.dumps(logical), encoding="utf-8")

    workflow = {
        "schema": "jit_iteration_workflow_v1",
        "workflow_name": "original",
        "state_dir": "state",
        "variables": {},
        "environment": {},
        "stages": [
            {
                "name": "frontier_train",
                "command": ["python", "train.py"],
                "requires": [],
                "completion": {"path": "train/role_manifest.json", "kind": "json", "assertions": [], "exports": {}},
            },
            {
                "name": "frontier_calibration",
                "command": ["python", "old_calibration.py"],
                "requires": [],
                "completion": {"path": str(failed_root / "role_manifest.json"), "kind": "json", "assertions": [], "exports": {}},
            },
            {
                "name": "fit_and_calibrate_Ck",
                "command": ["python", "fit.py", "--calibration-root", str(failed_root)],
                "requires": [{"path": str(failed_root / "role_manifest.json"), "kind": "file"}],
                "completion": {"path": "fields/summary.json", "kind": "json", "assertions": [], "exports": {}},
            },
        ],
    }
    source_workflow.write_text(json.dumps(workflow), encoding="utf-8")

    result = cli._prepare(
        source_plan=source_plan,
        failed_calibration_root=failed_root,
        source_workflow=source_workflow,
        repair_plan_out=repair_plan,
        repaired_calibration_root=repaired_root,
        workflow_out=workflow_out,
    )
    assert result["status"] == "prepared"
    revised = json.loads(workflow_out.read_text(encoding="utf-8"))
    names = [stage["name"] for stage in revised["stages"]]
    assert "frontier_train" in names
    assert "frontier_calibration_v3b_repair" in names
    fit = next(stage for stage in revised["stages"] if stage["name"] == "fit_and_calibrate_Ck")
    assert str(repaired_root) in fit["command"]
    assert str(failed_root) not in fit["command"]
    repair = json.loads(repair_plan.read_text(encoding="utf-8"))
    assert repair["parent_role_membership_changed"] is False
    assert repair["source_failed_upstream_counts"]["negative_count"] == 0
    assert repair["source_failed_downstream_counts"]["negative_count"] == 1
