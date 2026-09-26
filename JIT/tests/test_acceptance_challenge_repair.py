from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from jit_dvgc.iterative_frontier_protocol import canonical_sha256


def _load_cli():
    path = Path(__file__).resolve().parents[1] / "cli" / "acceptance_challenge_repair.py"
    spec = importlib.util.spec_from_file_location("acceptance_challenge_repair", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v3c_uses_historical_sparse_two_axis_family_in_both_phases() -> None:
    cli = _load_cli()
    assert cli.TWO_AXIS_PANEL == {
        "action_names": ["steer", "rear_wheel_drive", "hip", "knee"],
        "signs": [-1, 1],
        "strengths": [0.15, 0.30, 0.50],
        "durations": [2, 4, 8],
        "active_action_dimensions": 2,
    }
    assert cli.MIN_NEGATIVES_PER_PHASE == 5
    assert cli.MIN_NEGATIVE_PARENT_GROUPS == {"upstream": 2, "downstream": 1}
    assert cli.MIN_TOTAL_NEGATIVE_PARENT_GROUPS == 3


def test_v3c_gate_requires_negative_parent_diversity() -> None:
    cli = _load_cli()
    rows = []
    for phase, groups in (("upstream", ("u0", "u1")), ("downstream", ("d0",))):
        for index in range(5):
            rows.append(
                {
                    "phase": phase,
                    "label": 0,
                    "parent_group_id": groups[index % len(groups)],
                }
            )
        rows.append({"phase": phase, "label": 1, "parent_group_id": groups[0]})
    result = cli._challenge_gate(rows)
    assert result["phase_counts"]["upstream"]["negative_count"] == 5
    assert result["phase_counts"]["upstream"]["negative_parent_group_count"] == 2
    assert result["phase_counts"]["downstream"]["negative_count"] == 5
    assert result["total_negative_parent_group_count"] == 3

    bad = [dict(row) for row in rows]
    for row in bad:
        if row["phase"] == "upstream" and row["label"] == 0:
            row["parent_group_id"] = "u0"
    with pytest.raises(ValueError, match="negative parent support insufficient in upstream"):
        cli._challenge_gate(bad)


def test_prepare_emits_strict_workflow_schema_and_new_state_dir(tmp_path: Path) -> None:
    cli = _load_cli()
    source_plan = tmp_path / "frontier_plan.json"
    failed_root = tmp_path / "frontier_acceptance"
    failed_root.mkdir()
    source_workflow = tmp_path / "workflow_v3b_fixed.json"
    challenge_plan = tmp_path / "acceptance_challenge_plan.json"
    challenge_root = tmp_path / "frontier_acceptance_v3c"
    workflow_out = tmp_path / "workflow_v3c.json"
    state_dir = tmp_path / "state_v3c"

    actor = "a" * 64
    payload = "b" * 64
    anchors = []
    for index in range(3):
        anchors.append(
            {
                "role": "acceptance",
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
            "role": "acceptance",
            "phase": "downstream",
            "phase_index": 1,
            "entry_index": 0,
            "global_index": 10,
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
        "phase_probe_panels": {},
        "label_execution": {},
        "seeds": {
            "acceptance": {"acquisition": 1234, "labeling": 5678},
        },
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
                "split": "acceptance",
                "logical_role": "acceptance",
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
                "split": "acceptance",
                "logical_role": "acceptance",
            },
            {
                "candidate_id": "dneg",
                "phase": "downstream",
                "label": 0,
                "parent_group_id": "dg0",
                "policy_actor_sha256": actor,
                "policy_payload_sha256": payload,
                "split": "acceptance",
                "logical_role": "acceptance",
            },
        ]
    )
    logical = {
        "schema": "jit_iterative_frontier_logical_labels_v1",
        "role": "acceptance",
        "entries": rows,
    }
    logical["labels_sha256"] = canonical_sha256(logical)
    (failed_root / "logical_labels.json").write_text(json.dumps(logical), encoding="utf-8")

    workflow = {
        "schema": "jit_iteration_workflow_v1",
        "workflow_name": "v3b_fixed",
        "state_dir": "old_state",
        "variables": {},
        "environment": {},
        "stages": [
            {
                "name": "frontier_acceptance",
                "command": ["python", "old_acceptance.py"],
                "cwd": ".",
                "requires": [],
                "completion": {
                    "path": str(failed_root / "role_manifest.json"),
                    "kind": "json",
                    "assertions": [],
                    "exports": {},
                },
            },
            {
                "name": "lock_acceptance_baseline",
                "command": ["python", "lock.py", "--acceptance-root", str(failed_root)],
                "cwd": ".",
                "requires": [
                    {"path": str(failed_root / "role_manifest.json"), "kind": "file"}
                ],
                "completion": {
                    "path": "lock/baseline_lock.json",
                    "kind": "json",
                    "assertions": [],
                    "exports": {},
                },
            },
        ],
    }
    source_workflow.write_text(json.dumps(workflow), encoding="utf-8")

    result = cli._prepare(
        source_plan=source_plan,
        failed_acceptance_root=failed_root,
        source_workflow=source_workflow,
        challenge_plan_out=challenge_plan,
        challenge_root=challenge_root,
        workflow_out=workflow_out,
        state_dir=state_dir,
    )
    assert result["status"] == "prepared"
    revised = json.loads(workflow_out.read_text(encoding="utf-8"))
    assert set(revised) == cli.WORKFLOW_KEYS
    assert revised["state_dir"] == str(state_dir)
    names = [stage["name"] for stage in revised["stages"]]
    assert "frontier_acceptance_v3c_challenge" in names
    lock = next(stage for stage in revised["stages"] if stage["name"] == "lock_acceptance_baseline")
    assert str(challenge_root) in lock["command"]
    assert str(failed_root) not in lock["command"]
    required_paths = {str(requirement["path"]) for requirement in lock["requires"]}
    assert str(challenge_root / "role_manifest.json") in required_paths
    assert str(failed_root / "role_manifest.json") not in required_paths

    challenge = json.loads(challenge_plan.read_text(encoding="utf-8"))
    assert challenge["seeds"]["labeling"] == 5678
    assert challenge["parent_role_membership_changed"] is False
    assert challenge["claim_boundary"]["original_v3_acceptance_pass_claim"] is False
