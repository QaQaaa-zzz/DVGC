#!/usr/bin/env python3
"""Prepare/run the v3b upstream calibration hard-negative supplement.

This revision preserves the successful v3 TRAIN role and the failed v3
CALIBRATION artifact.  It does not move parent groups across logical roles.
Instead it reuses the already-predeclared upstream CALIBRATION parents and adds
one fresh sparse-two-axis acquisition/label supplement, while retaining the
completed downstream CALIBRATION evidence unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import jax

from jit_dvgc.config import file_sha256
from jit_dvgc.iterative_frontier_protocol import (
    ROLE_SCHEMA,
    _anchors_from_plan,
    _load_plan,
    _selected_policy,
    _source_tube,
    canonical_sha256,
)
from jit_dvgc.phase_specific_frontier import _runtime
from jit_dvgc.unified_boundary import collect_unified_boundary_candidates
from jit_dvgc.unified_continuation_labels import label_unified_continuations


REPAIR_SCHEMA = "jit_iterative_calibration_repair_plan_v1"
REPAIR_NAME = "upstream_sparse_two_axis_calibration_v3b"
UPSTREAM_PANEL = {
    "action_names": ["steer", "rear_wheel_drive", "hip", "knee"],
    "signs": [-1, 1],
    "strengths": [0.15, 0.30, 0.50],
    "durations": [2, 4, 8],
    "active_action_dimensions": 2,
}
ACQUISITION_SEED = 9_524_101
LABELING_SEED = 9_524_201
MIN_SUPPLEMENT_NEGATIVES = 5
MIN_SUPPLEMENT_NEGATIVE_PARENT_GROUPS = 2


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _verify_hash(payload: Mapping[str, Any], field: str) -> None:
    declared = str(payload.get(field, ""))
    base = {k: v for k, v in payload.items() if k != field}
    if len(declared) != 64 or canonical_sha256(base) != declared:
        raise ValueError(f"{field} self-hash drift")


def _failed_calibration_rows(root: Path, plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    root = Path(root)
    if (root / "role_manifest.json").exists():
        raise ValueError("repair requires the preserved failed calibration role, not a completed role")
    logical = _read(root / "logical_labels.json")
    if not isinstance(logical, dict) or logical.get("schema") != "jit_iterative_frontier_logical_labels_v1":
        raise ValueError("failed calibration logical-label artifact missing")
    if logical.get("role") != "calibration":
        raise ValueError("failed calibration logical role drift")
    _verify_hash(logical, "labels_sha256")
    rows = logical.get("entries")
    if not isinstance(rows, list) or not rows:
        raise ValueError("failed calibration logical labels empty")
    for row in rows:
        if row.get("split") != "calibration" or row.get("logical_role") != "calibration":
            raise ValueError("failed calibration row role drift")
        if row.get("policy_actor_sha256") != plan["policy_actor_sha256"]:
            raise ValueError("failed calibration actor drift")
        if row.get("policy_payload_sha256") != plan["policy_payload_sha256"]:
            raise ValueError("failed calibration payload drift")
    return [dict(row) for row in rows]


def _phase_counts(rows: list[dict[str, Any]], phase: str) -> dict[str, int]:
    selected = [row for row in rows if row.get("phase") == phase]
    positives = sum(int(row["label"]) for row in selected)
    groups = {str(row["parent_group_id"]) for row in selected}
    negative_groups = {
        str(row["parent_group_id"])
        for row in selected
        if int(row["label"]) == 0
    }
    return {
        "candidate_count": len(selected),
        "positive_count": positives,
        "negative_count": len(selected) - positives,
        "parent_group_count": len(groups),
        "negative_parent_group_count": len(negative_groups),
    }


def _prepare(
    *,
    source_plan: Path,
    failed_calibration_root: Path,
    source_workflow: Path,
    repair_plan_out: Path,
    repaired_calibration_root: Path,
    workflow_out: Path,
) -> dict[str, Any]:
    if repair_plan_out.exists() or workflow_out.exists() or repaired_calibration_root.exists():
        raise FileExistsError("v3b repair outputs must be new paths")
    plan = _load_plan(Path(source_plan))
    revision = plan.get("protocol_revision", {})
    if revision.get("name") != "phase_specific_two_axis_v3":
        raise ValueError("v3b repair requires the failed phase-specific two-axis v3 plan")
    rows = _failed_calibration_rows(Path(failed_calibration_root), plan)
    upstream = _phase_counts(rows, "upstream")
    downstream = _phase_counts(rows, "downstream")
    if upstream["candidate_count"] <= 0 or upstream["positive_count"] <= 0 or upstream["negative_count"] != 0:
        raise ValueError(f"v3b repair requires all-positive upstream calibration: {upstream}")
    if downstream["positive_count"] <= 0 or downstream["negative_count"] <= 0:
        raise ValueError(f"v3b repair requires already mixed downstream calibration: {downstream}")

    anchors = [
        dict(row)
        for row in plan.get("anchors", [])
        if row.get("role") == "calibration" and row.get("phase") == "upstream"
    ]
    if len(anchors) < 3:
        raise ValueError("v3b repair requires at least three locked upstream calibration parents")

    repair = {
        "schema": REPAIR_SCHEMA,
        "status": "predeclared_before_v3b_outcomes",
        "name": REPAIR_NAME,
        "source_plan": str(source_plan),
        "source_plan_sha256": str(plan["plan_sha256"]),
        "source_failed_calibration_root": str(failed_calibration_root),
        "source_failed_logical_labels_file_sha256": file_sha256(Path(failed_calibration_root) / "logical_labels.json"),
        "source_failed_upstream_counts": upstream,
        "source_failed_downstream_counts": downstream,
        "selected_policy_sha256": str(plan["selected_policy_sha256"]),
        "policy_actor_sha256": str(plan["policy_actor_sha256"]),
        "policy_payload_sha256": str(plan["policy_payload_sha256"]),
        "source_tube_manifest_sha256": str(plan["source_tube_manifest_sha256"]),
        "logical_role": "calibration",
        "reused_parent_groups_only": True,
        "parent_role_membership_changed": False,
        "upstream_calibration_anchors": anchors,
        "upstream_supplement_panel": dict(UPSTREAM_PANEL),
        "seeds": {
            "acquisition": ACQUISITION_SEED,
            "labeling": LABELING_SEED,
        },
        "continuation_max_label_ticks": int(plan["fixed_probe_panel"]["max_label_ticks"]),
        "supplement_acceptance_gate": {
            "minimum_negative_count": MIN_SUPPLEMENT_NEGATIVES,
            "minimum_negative_parent_group_count": MIN_SUPPLEMENT_NEGATIVE_PARENT_GROUPS,
            "combined_upstream_requires_both_labels": True,
            "downstream_source_rows_reused_without_relabeling": True,
        },
        "unchanged_contracts": [
            "pi1_identity",
            "Tube1_identity",
            "v3_TRAIN_role_and_rows",
            "v3_ACCEPTANCE_parent_assignment",
            "upstream_CALIBRATION_parent_assignment",
            "400_tick_continuation_definition",
            "real_dynamics_only",
            "TEST_final_isolation",
            "model_weights_fit_on_TRAIN_only",
        ],
        "training_transitions": 0,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "automatic_further_repair": False,
    }
    repair["repair_plan_sha256"] = canonical_sha256(repair)
    _write(repair_plan_out, repair)

    workflow = _read(source_workflow)
    if not isinstance(workflow, dict) or workflow.get("schema") != "jit_iteration_workflow_v1":
        raise ValueError("source workflow schema drift")
    stages = workflow.get("stages")
    if not isinstance(stages, list):
        raise ValueError("source workflow stages missing")
    old_root = str(failed_calibration_root)
    new_root = str(repaired_calibration_root)
    replaced = []
    found_calibration_stage = False
    for source_stage in stages:
        stage = json.loads(json.dumps(source_stage))
        if stage.get("name") == "frontier_calibration":
            found_calibration_stage = True
            interpreter = stage["command"][0]
            stage["name"] = "frontier_calibration_v3b_repair"
            stage["command"] = [
                interpreter,
                "JIT/cli/calibration_hard_negative_repair.py",
                "run",
                "--repair-plan",
                str(repair_plan_out),
                "--output-dir",
                new_root,
            ]
            stage["requires"] = [
                {"path": str(repair_plan_out), "kind": "file"},
                {"path": str(Path(failed_calibration_root) / "logical_labels.json"), "kind": "file"},
            ]
            stage["completion"] = {
                "path": str(Path(repaired_calibration_root) / "role_manifest.json"),
                "kind": "json",
                "assertions": [
                    {"pointer": "/status", "op": "eq", "value": "completed"},
                    {"pointer": "/role", "op": "eq", "value": "calibration"},
                    {"pointer": "/iteration", "op": "eq", "value": int(plan["iteration"])},
                ],
                "exports": {},
            }
        else:
            raw = json.dumps(stage)
            raw = raw.replace(old_root, new_root)
            stage = json.loads(raw)
        replaced.append(stage)
    if not found_calibration_stage:
        raise ValueError("source workflow has no frontier_calibration stage")
    revised = dict(workflow)
    revised["workflow_name"] = str(workflow.get("workflow_name", "workflow")) + "_v3b_calibration_repair"
    revised["stages"] = replaced
    revised["calibration_repair_plan"] = str(repair_plan_out)
    revised["calibration_repair_plan_sha256"] = str(repair["repair_plan_sha256"])
    revised["source_workflow"] = str(source_workflow)
    _write(workflow_out, revised)
    return {
        "status": "prepared",
        "repair_plan": str(repair_plan_out),
        "repair_plan_sha256": repair["repair_plan_sha256"],
        "repaired_calibration_root": new_root,
        "workflow": str(workflow_out),
        "source_upstream_counts": upstream,
        "source_downstream_counts": downstream,
    }


def _run(*, repair_plan_path: Path, output_dir: Path) -> dict[str, Any]:
    repair = _read(repair_plan_path)
    if not isinstance(repair, dict) or repair.get("schema") != REPAIR_SCHEMA:
        raise ValueError("invalid v3b calibration repair plan")
    if repair.get("status") != "predeclared_before_v3b_outcomes" or repair.get("name") != REPAIR_NAME:
        raise ValueError("v3b calibration repair plan status/name drift")
    _verify_hash(repair, "repair_plan_sha256")
    source_plan_path = Path(str(repair["source_plan"]))
    plan = _load_plan(source_plan_path)
    if plan["plan_sha256"] != repair["source_plan_sha256"]:
        raise ValueError("v3b source plan identity drift")
    failed_root = Path(str(repair["source_failed_calibration_root"]))
    source_rows = _failed_calibration_rows(failed_root, plan)
    if file_sha256(failed_root / "logical_labels.json") != repair["source_failed_logical_labels_file_sha256"]:
        raise ValueError("v3b failed calibration evidence drift")

    output_dir = Path(output_dir)
    manifest_path = output_dir / "role_manifest.json"
    if manifest_path.is_file():
        manifest = _read(manifest_path)
        _verify_hash(manifest, "role_manifest_sha256")
        return manifest
    output_dir.mkdir(parents=True, exist_ok=True)

    selected, record, frozen_path = _selected_policy(Path(str(plan["selected_policy"])))
    artifact = _source_tube(selected, record, Path(str(plan["source_tube"])))
    if artifact.manifest["manifest_sha256"] != repair["source_tube_manifest_sha256"]:
        raise ValueError("v3b source Tube drift")
    anchors = tuple(
        anchor
        for anchor in _anchors_from_plan(plan, artifact, "calibration")
        if anchor.phase == "upstream"
    )
    if len(anchors) != len(repair["upstream_calibration_anchors"]):
        raise ValueError("v3b locked upstream calibration anchor count drift")

    supplement_root = output_dir / "upstream_hard_negative_supplement"
    acquisition_dir = supplement_root / "acquisition"
    labels_dir = supplement_root / "labels"
    frozen_sha = file_sha256(frozen_path)
    env, policy = _runtime(record=record, artifact=artifact)
    if not (acquisition_dir / "catalog.json").is_file():
        panel = repair["upstream_supplement_panel"]
        collect_unified_boundary_candidates(
            anchors,
            acquisition_dir,
            env=env,
            policy=policy,
            policy_record=record,
            frozen_manifest_sha256=frozen_sha,
            protocol_seed=int(repair["seeds"]["acquisition"]),
            frontier_score_ceiling=1.0,
            strengths=tuple(float(x) for x in panel["strengths"]),
            durations=tuple(int(x) for x in panel["durations"]),
            action_names=tuple(str(x) for x in panel["action_names"]),
            signs=tuple(int(x) for x in panel["signs"]),
            active_action_dimensions=int(panel["active_action_dimensions"]),
        )
    if not (labels_dir / "summary.json").is_file():
        label_unified_continuations(
            acquisition_dir / "catalog.json",
            labels_dir,
            env=env,
            policy=policy,
            policy_record=record,
            frozen_manifest_sha256=frozen_sha,
            max_ticks=int(repair["continuation_max_label_ticks"]),
            protocol_seed=int(repair["seeds"]["labeling"]),
        )

    supplement_labels = _read(labels_dir / "labels.json")
    if not isinstance(supplement_labels, list) or not supplement_labels:
        raise ValueError("v3b supplement labels missing")
    source_states = {str(row["state_sha256"]) for row in source_rows}
    unique_supplement = []
    duplicate_source_states = 0
    seen = set(source_states)
    for source in supplement_labels:
        if source.get("phase") != "upstream":
            raise ValueError("v3b supplement unexpectedly contains downstream row")
        state_sha = str(source["state_sha256"])
        if state_sha in seen:
            duplicate_source_states += 1
            continue
        seen.add(state_sha)
        row = dict(source)
        row["source_candidate_id"] = str(row["candidate_id"])
        row["candidate_id"] = "v3b_upstream_calibration_" + str(row["candidate_id"])
        row["source_legacy_split"] = str(row.get("split", "train"))
        row["split"] = "calibration"
        row["logical_role"] = "calibration"
        row["calibration_repair_plan_sha256"] = str(repair["repair_plan_sha256"])
        unique_supplement.append(row)

    supplement_counts = _phase_counts(unique_supplement, "upstream")
    gate = repair["supplement_acceptance_gate"]
    if supplement_counts["negative_count"] < int(gate["minimum_negative_count"]):
        raise ValueError(f"v3b upstream supplement negative support insufficient: {supplement_counts}")
    if supplement_counts["negative_parent_group_count"] < int(gate["minimum_negative_parent_group_count"]):
        raise ValueError(f"v3b upstream supplement negative parent support insufficient: {supplement_counts}")

    combined = [dict(row) for row in source_rows] + unique_supplement
    combined_up = _phase_counts(combined, "upstream")
    combined_down = _phase_counts(combined, "downstream")
    if combined_up["positive_count"] <= 0 or combined_up["negative_count"] <= 0:
        raise ValueError(f"v3b combined upstream calibration is not two-class: {combined_up}")
    if combined_down["positive_count"] <= 0 or combined_down["negative_count"] <= 0:
        raise ValueError(f"v3b combined downstream calibration is not two-class: {combined_down}")

    logical = {
        "schema": "jit_iterative_frontier_logical_labels_v1",
        "role": "calibration",
        "entries": combined,
        "repair_provenance": {
            "source_failed_calibration": str(failed_root / "logical_labels.json"),
            "supplement_labels": str(labels_dir / "labels.json"),
            "repair_plan_sha256": str(repair["repair_plan_sha256"]),
            "duplicate_source_state_count_excluded": duplicate_source_states,
        },
    }
    logical["labels_sha256"] = canonical_sha256(logical)
    _write(output_dir / "logical_labels.json", logical)

    acquisition_summary = _read(acquisition_dir / "summary.json")
    label_summary = _read(labels_dir / "summary.json")
    manifest = {
        "schema": ROLE_SCHEMA,
        "status": "completed",
        "iteration": int(plan["iteration"]),
        "role": "calibration",
        "logical_split": "calibration",
        "legacy_low_level_split_marker": "train",
        "legacy_marker_is_not_logical_data_role": True,
        "plan": str(source_plan_path),
        "plan_sha256": str(plan["plan_sha256"]),
        "selected_policy_sha256": str(plan["selected_policy_sha256"]),
        "policy_actor_sha256": str(record["actor_sha256"]),
        "policy_payload_sha256": str(record["payload_sha256"]),
        "source_tube_manifest_sha256": str(artifact.manifest["manifest_sha256"]),
        "logical_labels": str(output_dir / "logical_labels.json"),
        "logical_labels_file_sha256": file_sha256(output_dir / "logical_labels.json"),
        "phase_counts": {
            "upstream": combined_up,
            "downstream": combined_down,
        },
        "calibration_repair": {
            "name": REPAIR_NAME,
            "repair_plan": str(repair_plan_path),
            "repair_plan_sha256": str(repair["repair_plan_sha256"]),
            "source_failed_calibration_root": str(failed_root),
            "source_failed_rows_retained": len(source_rows),
            "supplement_unique_rows_added": len(unique_supplement),
            "supplement_counts": supplement_counts,
            "supplement_acquisition_catalog": str(acquisition_dir / "catalog.json"),
            "supplement_acquisition_catalog_sha256": file_sha256(acquisition_dir / "catalog.json"),
            "supplement_label_summary": str(labels_dir / "summary.json"),
            "supplement_label_summary_sha256": file_sha256(labels_dir / "summary.json"),
            "duplicate_source_state_count_excluded": duplicate_source_states,
            "parent_role_membership_changed": False,
            "downstream_relabeling_performed": False,
        },
        "environment_interactions": int(acquisition_summary.get("environment_interactions", 0)) + int(label_summary.get("environment_interactions", 0)),
        "training_transitions": 0,
        "expert_switching_used": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "claim_boundary": {
            "role_locked_before_next_policy_training": True,
            "train_rows_may_fit_fields": False,
            "rows_may_calibrate_thresholds": True,
            "rows_may_gate_next_policy": False,
            "rows_may_enter_tube": False,
            "certified_safe_set_claim": False,
            "jce_jel_claim": False,
        },
    }
    manifest["role_manifest_sha256"] = canonical_sha256(manifest)
    _write(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)

    prepare = subs.add_parser("prepare")
    prepare.add_argument("--source-plan", type=Path, required=True)
    prepare.add_argument("--failed-calibration-root", type=Path, required=True)
    prepare.add_argument("--source-workflow", type=Path, required=True)
    prepare.add_argument("--repair-plan-out", type=Path, required=True)
    prepare.add_argument("--repaired-calibration-root", type=Path, required=True)
    prepare.add_argument("--workflow-out", type=Path, required=True)

    run = subs.add_parser("run")
    run.add_argument("--repair-plan", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "prepare":
        result = _prepare(
            source_plan=args.source_plan,
            failed_calibration_root=args.failed_calibration_root,
            source_workflow=args.source_workflow,
            repair_plan_out=args.repair_plan_out,
            repaired_calibration_root=args.repaired_calibration_root,
            workflow_out=args.workflow_out,
        )
    else:
        if jax.default_backend() != "gpu":
            raise RuntimeError("v3b calibration repair rollout requires the visible JAX GPU")
        result = _run(repair_plan_path=args.repair_plan, output_dir=args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
