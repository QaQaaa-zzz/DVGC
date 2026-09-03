#!/usr/bin/env python3
"""Prepare/run the v3c fresh acceptance challenge bank before pi2 training.

The failed v3 acceptance bank is preserved as evidence. This revision creates a
new acceptance-only bank on the already-locked ACCEPTANCE parent groups, using
the historical sparse two-axis challenge family in both phases. The source
frontier plan, role membership, frozen pi1, Tube1, continuation horizon, and
acceptance labeling seed remain unchanged. The new bank must be completed and
locked before any pi2 candidate training.
"""
from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from jit_dvgc.config import file_sha256
from jit_dvgc.iterative_frontier_protocol import (
    ROLE_SCHEMA,
    _anchors_from_plan,
    _load_plan,
    _logical_rows,
    _selected_policy,
    _source_tube,
    canonical_sha256,
)
from jit_dvgc.phase_specific_frontier import _completed_phase_acquisition, _runtime
from jit_dvgc.unified_boundary import collect_unified_boundary_candidates
from jit_dvgc.unified_continuation_labels import label_unified_continuations


CHALLENGE_SCHEMA = "jit_iterative_acceptance_challenge_plan_v1"
CHALLENGE_NAME = "fresh_sparse_two_axis_acceptance_v3c"
PHASES = ("upstream", "downstream")
TWO_AXIS_PANEL = {
    "action_names": ["steer", "rear_wheel_drive", "hip", "knee"],
    "signs": [-1, 1],
    "strengths": [0.15, 0.30, 0.50],
    "durations": [2, 4, 8],
    "active_action_dimensions": 2,
}
MIN_POSITIVES_PER_PHASE = 1
MIN_NEGATIVES_PER_PHASE = 5
MIN_NEGATIVE_PARENT_GROUPS = {"upstream": 2, "downstream": 1}
MIN_TOTAL_NEGATIVE_PARENT_GROUPS = 3
WORKFLOW_KEYS = {
    "schema",
    "workflow_name",
    "state_dir",
    "variables",
    "environment",
    "stages",
}


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
    base = {key: value for key, value in payload.items() if key != field}
    if len(declared) != 64 or canonical_sha256(base) != declared:
        raise ValueError(f"{field} self-hash drift")


def _failed_acceptance_rows(root: Path, plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    root = Path(root)
    if (root / "role_manifest.json").exists():
        raise ValueError("v3c requires the preserved failed acceptance role")
    logical = _read(root / "logical_labels.json")
    if not isinstance(logical, dict) or logical.get("schema") != "jit_iterative_frontier_logical_labels_v1":
        raise ValueError("failed acceptance logical labels missing")
    if logical.get("role") != "acceptance":
        raise ValueError("failed acceptance logical role drift")
    _verify_hash(logical, "labels_sha256")
    rows = logical.get("entries")
    if not isinstance(rows, list) or not rows:
        raise ValueError("failed acceptance rows empty")
    for row in rows:
        if row.get("split") != "acceptance" or row.get("logical_role") != "acceptance":
            raise ValueError("failed acceptance row role drift")
        if row.get("policy_actor_sha256") != plan["policy_actor_sha256"]:
            raise ValueError("failed acceptance actor drift")
        if row.get("policy_payload_sha256") != plan["policy_payload_sha256"]:
            raise ValueError("failed acceptance payload drift")
    return [dict(row) for row in rows]


def _phase_counts(rows: list[dict[str, Any]], phase: str) -> dict[str, int]:
    selected = [row for row in rows if row.get("phase") == phase]
    positives = sum(int(row["label"]) for row in selected)
    negative_groups = {
        str(row["parent_group_id"])
        for row in selected
        if int(row["label"]) == 0
    }
    all_groups = {str(row["parent_group_id"]) for row in selected}
    return {
        "candidate_count": len(selected),
        "positive_count": positives,
        "negative_count": len(selected) - positives,
        "parent_group_count": len(all_groups),
        "negative_parent_group_count": len(negative_groups),
    }


def _challenge_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_phase = {phase: _phase_counts(rows, phase) for phase in PHASES}
    negative_groups = {
        str(row["parent_group_id"])
        for row in rows
        if int(row["label"]) == 0
    }
    for phase in PHASES:
        counts = by_phase[phase]
        if counts["positive_count"] < MIN_POSITIVES_PER_PHASE:
            raise ValueError(f"v3c acceptance positive support insufficient in {phase}: {counts}")
        if counts["negative_count"] < MIN_NEGATIVES_PER_PHASE:
            raise ValueError(f"v3c acceptance negative support insufficient in {phase}: {counts}")
        if counts["negative_parent_group_count"] < MIN_NEGATIVE_PARENT_GROUPS[phase]:
            raise ValueError(
                f"v3c acceptance negative parent support insufficient in {phase}: {counts}"
            )
    if len(negative_groups) < MIN_TOTAL_NEGATIVE_PARENT_GROUPS:
        raise ValueError(
            "v3c acceptance total negative parent support insufficient: "
            f"{len(negative_groups)} < {MIN_TOTAL_NEGATIVE_PARENT_GROUPS}"
        )
    return {
        "phase_counts": by_phase,
        "total_negative_parent_group_count": len(negative_groups),
        "gate": {
            "minimum_positive_count_per_phase": MIN_POSITIVES_PER_PHASE,
            "minimum_negative_count_per_phase": MIN_NEGATIVES_PER_PHASE,
            "minimum_negative_parent_group_count": dict(MIN_NEGATIVE_PARENT_GROUPS),
            "minimum_total_negative_parent_group_count": MIN_TOTAL_NEGATIVE_PARENT_GROUPS,
        },
    }


def _prepare(
    *,
    source_plan: Path,
    failed_acceptance_root: Path,
    source_workflow: Path,
    challenge_plan_out: Path,
    challenge_root: Path,
    workflow_out: Path,
    state_dir: Path,
) -> dict[str, Any]:
    for path in (challenge_plan_out, workflow_out, challenge_root, state_dir):
        if Path(path).exists():
            raise FileExistsError(f"v3c output must be new: {path}")

    plan = _load_plan(Path(source_plan))
    if plan.get("protocol_revision", {}).get("name") != "phase_specific_two_axis_v3":
        raise ValueError("v3c requires the v3 phase-specific source plan")

    failed_rows = _failed_acceptance_rows(Path(failed_acceptance_root), plan)
    failed_counts = {phase: _phase_counts(failed_rows, phase) for phase in PHASES}
    if failed_counts["upstream"]["candidate_count"] <= 0:
        raise ValueError("v3c requires observed upstream acceptance evidence")
    if failed_counts["upstream"]["negative_count"] != 0:
        raise ValueError(
            "v3c is specifically for the upstream-all-positive acceptance failure: "
            f"{failed_counts['upstream']}"
        )

    anchors = [dict(row) for row in plan.get("anchors", []) if row.get("role") == "acceptance"]
    phase_anchor_counts = Counter(str(row["phase"]) for row in anchors)
    if phase_anchor_counts["upstream"] < 3 or phase_anchor_counts["downstream"] < 1:
        raise ValueError(
            f"v3c locked acceptance parent coverage drift: {dict(phase_anchor_counts)}"
        )

    seeds = dict(plan["seeds"]["acceptance"])
    challenge = {
        "schema": CHALLENGE_SCHEMA,
        "status": "predeclared_before_candidate_training",
        "name": CHALLENGE_NAME,
        "source_plan": str(source_plan),
        "source_plan_sha256": str(plan["plan_sha256"]),
        "source_failed_acceptance_root": str(failed_acceptance_root),
        "source_failed_logical_labels_file_sha256": file_sha256(
            Path(failed_acceptance_root) / "logical_labels.json"
        ),
        "source_failed_phase_counts": failed_counts,
        "iteration": int(plan["iteration"]),
        "selected_policy_sha256": str(plan["selected_policy_sha256"]),
        "policy_actor_sha256": str(plan["policy_actor_sha256"]),
        "policy_payload_sha256": str(plan["policy_payload_sha256"]),
        "source_tube_manifest_sha256": str(plan["source_tube_manifest_sha256"]),
        "logical_role": "acceptance",
        "locked_acceptance_anchors": anchors,
        "parent_role_membership_changed": False,
        "fresh_bank_replaces_failed_bank_for_future_gate_only": True,
        "phase_probe_panels": {phase: dict(TWO_AXIS_PANEL) for phase in PHASES},
        "seeds": {
            "acquisition": int(seeds["acquisition"]),
            "labeling": int(seeds["labeling"]),
        },
        "continuation_max_label_ticks": int(plan["fixed_probe_panel"]["max_label_ticks"]),
        "challenge_gate": {
            "minimum_positive_count_per_phase": MIN_POSITIVES_PER_PHASE,
            "minimum_negative_count_per_phase": MIN_NEGATIVES_PER_PHASE,
            "minimum_negative_parent_group_count": dict(MIN_NEGATIVE_PARENT_GROUPS),
            "minimum_total_negative_parent_group_count": MIN_TOTAL_NEGATIVE_PARENT_GROUPS,
        },
        "unchanged_contracts": [
            "pi1_identity",
            "Tube1_identity",
            "v3_TRAIN_rows",
            "v3b_repaired_CALIBRATION_rows",
            "acceptance_parent_role_membership",
            "400_tick_continuation_definition",
            "original_acceptance_labeling_seed",
            "real_dynamics_only",
            "TEST_final_isolation",
            "candidate_pi2_not_trained_before_bank_lock",
        ],
        "claim_boundary": {
            "original_v3_acceptance_pass_claim": False,
            "new_bank_may_gate_future_pi2_only": True,
            "candidate_policy_outcomes_inspected": False,
            "certified_safe_set_claim": False,
            "jce_jel_claim": False,
        },
        "automatic_further_repair": False,
        "training_transitions": 0,
        "test_data_used": False,
        "final_evaluation_data_used": False,
    }
    challenge["challenge_plan_sha256"] = canonical_sha256(challenge)
    _write(challenge_plan_out, challenge)

    workflow = _read(source_workflow)
    if not isinstance(workflow, dict) or workflow.get("schema") != "jit_iteration_workflow_v1":
        raise ValueError("source workflow schema drift")
    if set(workflow).difference(WORKFLOW_KEYS):
        raise ValueError("source workflow contains unknown top-level fields")

    old_root = str(failed_acceptance_root)
    new_root = str(challenge_root)
    revised_stages = []
    found = False
    for source_stage in workflow.get("stages", []):
        stage = json.loads(json.dumps(source_stage))
        if stage.get("name") == "frontier_acceptance":
            found = True
            interpreter = stage["command"][0]
            stage["name"] = "frontier_acceptance_v3c_challenge"
            stage["command"] = [
                interpreter,
                "JIT/cli/acceptance_challenge_repair.py",
                "run",
                "--challenge-plan",
                str(challenge_plan_out),
                "--output-dir",
                new_root,
            ]
            stage["requires"] = [
                {"path": str(challenge_plan_out), "kind": "file"},
                {
                    "path": str(Path(failed_acceptance_root) / "logical_labels.json"),
                    "kind": "file",
                },
            ]
            stage["completion"] = {
                "path": str(Path(challenge_root) / "role_manifest.json"),
                "kind": "json",
                "assertions": [
                    {"pointer": "/status", "op": "eq", "value": "completed"},
                    {"pointer": "/role", "op": "eq", "value": "acceptance"},
                    {"pointer": "/iteration", "op": "eq", "value": int(plan["iteration"])},
                ],
                "exports": {},
            }
        else:
            raw = json.dumps(stage)
            raw = raw.replace(old_root, new_root)
            stage = json.loads(raw)
        revised_stages.append(stage)
    if not found:
        raise ValueError("source workflow has no frontier_acceptance stage")

    revised = {
        "schema": workflow["schema"],
        "workflow_name": str(workflow["workflow_name"]) + "_v3c_acceptance_challenge",
        "state_dir": str(state_dir),
        "variables": dict(workflow.get("variables", {})),
        "environment": dict(workflow.get("environment", {})),
        "stages": revised_stages,
    }
    _write(workflow_out, revised)
    return {
        "status": "prepared",
        "challenge_plan": str(challenge_plan_out),
        "challenge_plan_sha256": str(challenge["challenge_plan_sha256"]),
        "challenge_root": str(challenge_root),
        "workflow": str(workflow_out),
        "state_dir": str(state_dir),
        "failed_phase_counts": failed_counts,
        "phase_anchor_counts": dict(phase_anchor_counts),
    }


def _merge_phase_acquisitions(
    *,
    challenge_plan_path: Path,
    challenge: Mapping[str, Any],
    source_plan: Mapping[str, Any],
    phase_catalogs: Mapping[str, Mapping[str, Any]],
    acquisition_dir: Path,
) -> dict[str, Any]:
    first = phase_catalogs["upstream"]
    for phase in PHASES:
        catalog = phase_catalogs[phase]
        for field in (
            "iteration",
            "policy_name",
            "policy_actor_sha256",
            "policy_payload_sha256",
            "frozen_unified_manifest_sha256",
            "source_tube_manifest_sha256",
        ):
            if catalog.get(field) != first.get(field):
                raise ValueError(f"v3c phase acquisition {field} drift")

    phase_protocols = {}
    entries: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_states: set[str] = set()
    exclusions: Counter[str] = Counter()
    phase_attempted: dict[str, int] = {}
    phase_candidates: dict[str, int] = {}
    phase_exclusions: dict[str, dict[str, int]] = {}
    attempted = interactions = maximum_interactions = anchor_count = 0

    for phase in PHASES:
        catalog = phase_catalogs[phase]
        phase_protocols[phase] = {
            "directory": f"phase_{phase}",
            "protocol_sha256": str(catalog["protocol_sha256"]),
            "panel": dict(challenge["phase_probe_panels"][phase]),
            "candidate_count": int(catalog["candidate_count"]),
            "environment_interactions": int(catalog["environment_interactions"]),
        }

    protocol = {
        "schema": "jit_unified_boundary_protocol_v1",
        "status": "predeclared",
        "purpose": "fresh_candidate_blind_acceptance_challenge_bank_v3c",
        "split": "train",
        "iteration": int(first["iteration"]),
        "policy_name": str(first["policy_name"]),
        "policy_actor_sha256": str(first["policy_actor_sha256"]),
        "policy_payload_sha256": str(first["policy_payload_sha256"]),
        "frozen_unified_manifest_sha256": str(first["frozen_unified_manifest_sha256"]),
        "source_tube_manifest_sha256": str(first["source_tube_manifest_sha256"]),
        "plan": str(challenge["source_plan"]),
        "plan_sha256": str(source_plan["plan_sha256"]),
        "acceptance_challenge_plan": str(challenge_plan_path),
        "acceptance_challenge_plan_sha256": str(challenge["challenge_plan_sha256"]),
        "logical_role": "acceptance",
        "protocol_seed": int(challenge["seeds"]["acquisition"]),
        "phase_protocols": phase_protocols,
        "state_generation": (
            "reset exact locked acceptance parent; apply the predeclared v3c sparse "
            "two-axis bounded action perturbation; advance only through authoritative "
            "env.step; reject terminal/nonfinite/phase-crossing states"
        ),
        "training_transitions": 0,
        "expert_switching_used": False,
        "test_data_used": False,
        "validation_data_used": False,
        "final_evaluation_data_used": False,
        "claim_boundary": {
            "fresh_acceptance_bank_before_candidate_training": True,
            "candidate_policy_outcomes_inspected": False,
            "tube_expansion_claim": False,
            "certified_safe_set_claim": False,
            "jce_jel_claim": False,
        },
    }
    protocol_sha = canonical_sha256(protocol)
    protocol_with_sha = {**protocol, "protocol_sha256": protocol_sha}

    for phase in PHASES:
        catalog = phase_catalogs[phase]
        attempted += int(catalog["attempted_candidate_count"])
        interactions += int(catalog["environment_interactions"])
        maximum_interactions += int(catalog["maximum_environment_interactions"])
        anchor_count += int(catalog["anchor_count"])
        phase_attempted[phase] = int(catalog["phase_attempted_candidate_counts"][phase])
        phase_candidates[phase] = int(catalog["phase_candidate_counts"][phase])
        phase_exclusions[phase] = {
            str(key): int(value)
            for key, value in dict(catalog["phase_exclusion_counts"][phase]).items()
        }
        exclusions.update(
            {str(key): int(value) for key, value in dict(catalog["exclusion_counts"]).items()}
        )
        for source in catalog["entries"]:
            row = dict(source)
            candidate_id = str(row["candidate_id"])
            state_sha = str(row["state_sha256"])
            if candidate_id in seen_ids:
                raise ValueError("v3c duplicate candidate id across phases")
            if state_sha in seen_states:
                raise ValueError("v3c duplicate physical state across phases")
            seen_ids.add(candidate_id)
            seen_states.add(state_sha)
            row["phase_acquisition_protocol_sha256"] = str(row["protocol_sha256"])
            row["protocol_sha256"] = protocol_sha
            row["source_bank"] = f"phase_{phase}/{row['source_bank']}"
            entries.append(row)

    report = {
        "schema": "jit_unified_boundary_catalog_v1",
        "status": "completed",
        "artifact_role": "candidate_blind_acceptance_challenge_candidates",
        "split": "train",
        "iteration": int(first["iteration"]),
        "policy_name": str(first["policy_name"]),
        "policy_actor_sha256": str(first["policy_actor_sha256"]),
        "policy_payload_sha256": str(first["policy_payload_sha256"]),
        "frozen_unified_manifest_sha256": str(first["frozen_unified_manifest_sha256"]),
        "source_tube_manifest_sha256": str(first["source_tube_manifest_sha256"]),
        "protocol_sha256": protocol_sha,
        "frontier_score_ceiling": 1.0,
        "anchor_count": anchor_count,
        "attempted_candidate_count": attempted,
        "candidate_count": len(entries),
        "phase_attempted_candidate_counts": phase_attempted,
        "phase_candidate_counts": phase_candidates,
        "phase_exclusion_counts": phase_exclusions,
        "environment_interactions": interactions,
        "maximum_environment_interactions": maximum_interactions,
        "training_transitions": 0,
        "expert_switching_used": False,
        "test_data_used": False,
        "validation_data_used": False,
        "final_evaluation_data_used": False,
        "exclusion_counts": dict(sorted(exclusions.items())),
        "claim_boundary": {
            "fresh_acceptance_bank_before_candidate_training": True,
            "candidate_policy_outcomes_inspected": False,
            "tube_expansion_claim": False,
            "certified_safe_set_claim": False,
            "jce_jel_claim": False,
        },
        "entries": entries,
    }
    acquisition_dir.mkdir(parents=True, exist_ok=True)
    _write(acquisition_dir / "protocol.json", protocol_with_sha)
    _write(acquisition_dir / "catalog.json", report)
    _write(
        acquisition_dir / "summary.json",
        {key: value for key, value in report.items() if key != "entries"},
    )
    return report


def _run(*, challenge_plan_path: Path, output_dir: Path) -> dict[str, Any]:
    challenge = _read(challenge_plan_path)
    if not isinstance(challenge, dict) or challenge.get("schema") != CHALLENGE_SCHEMA:
        raise ValueError("invalid v3c acceptance challenge plan")
    if challenge.get("status") != "predeclared_before_candidate_training" or challenge.get("name") != CHALLENGE_NAME:
        raise ValueError("v3c acceptance challenge plan status/name drift")
    _verify_hash(challenge, "challenge_plan_sha256")

    source_plan_path = Path(str(challenge["source_plan"]))
    plan = _load_plan(source_plan_path)
    if plan["plan_sha256"] != challenge["source_plan_sha256"]:
        raise ValueError("v3c source plan identity drift")
    failed_root = Path(str(challenge["source_failed_acceptance_root"]))
    if file_sha256(failed_root / "logical_labels.json") != challenge["source_failed_logical_labels_file_sha256"]:
        raise ValueError("v3c failed acceptance evidence drift")

    output_dir = Path(output_dir)
    manifest_path = output_dir / "role_manifest.json"
    if manifest_path.is_file():
        manifest = _read(manifest_path)
        _verify_hash(manifest, "role_manifest_sha256")
        return manifest
    output_dir.mkdir(parents=True, exist_ok=True)

    selected, record, frozen_path = _selected_policy(Path(str(plan["selected_policy"])))
    artifact = _source_tube(selected, record, Path(str(plan["source_tube"])))
    if artifact.manifest["manifest_sha256"] != challenge["source_tube_manifest_sha256"]:
        raise ValueError("v3c source Tube drift")

    anchors = _anchors_from_plan(plan, artifact, "acceptance")
    frozen_sha = file_sha256(frozen_path)
    env = policy = None

    def ensure_runtime():
        nonlocal env, policy
        if env is None or policy is None:
            env, policy = _runtime(record=record, artifact=artifact)
        return env, policy

    acquisition_dir = output_dir / "acquisition"
    phase_catalogs: dict[str, Mapping[str, Any]] = {}
    for phase in PHASES:
        phase_dir = acquisition_dir / f"phase_{phase}"
        panel = challenge["phase_probe_panels"][phase]
        completed = _completed_phase_acquisition(
            phase_dir=phase_dir,
            phase=phase,
            panel=panel,
            policy_record=record,
            frozen_manifest_sha256=frozen_sha,
        )
        if completed is None:
            runtime_env, runtime_policy = ensure_runtime()
            phase_anchors = tuple(anchor for anchor in anchors if anchor.phase == phase)
            if not phase_anchors:
                raise ValueError(f"v3c no locked acceptance anchors in {phase}")
            completed = collect_unified_boundary_candidates(
                phase_anchors,
                phase_dir,
                env=runtime_env,
                policy=runtime_policy,
                policy_record=record,
                frozen_manifest_sha256=frozen_sha,
                protocol_seed=int(challenge["seeds"]["acquisition"]),
                frontier_score_ceiling=1.0,
                strengths=tuple(float(value) for value in panel["strengths"]),
                durations=tuple(int(value) for value in panel["durations"]),
                action_names=tuple(str(value) for value in panel["action_names"]),
                signs=tuple(int(value) for value in panel["signs"]),
                active_action_dimensions=int(panel["active_action_dimensions"]),
            )
        phase_catalogs[phase] = completed

    if not (acquisition_dir / "catalog.json").is_file():
        acquisition = _merge_phase_acquisitions(
            challenge_plan_path=challenge_plan_path,
            challenge=challenge,
            source_plan=plan,
            phase_catalogs=phase_catalogs,
            acquisition_dir=acquisition_dir,
        )
    else:
        acquisition = _read(acquisition_dir / "catalog.json")

    for phase in PHASES:
        count = int(acquisition["phase_candidate_counts"].get(phase, 0))
        groups = {
            str(row["parent_group_id"])
            for row in acquisition["entries"]
            if row.get("phase") == phase
        }
        if count < 2 or len(groups) < 1:
            raise ValueError(
                f"v3c acceptance acquisition support insufficient in {phase}: "
                f"candidate_count={count}, parent_group_count={len(groups)}"
            )

    labels_dir = output_dir / "labels"
    if not (labels_dir / "summary.json").is_file():
        runtime_env, runtime_policy = ensure_runtime()
        labeling = label_unified_continuations(
            acquisition_dir / "catalog.json",
            labels_dir,
            env=runtime_env,
            policy=runtime_policy,
            policy_record=record,
            frozen_manifest_sha256=frozen_sha,
            max_ticks=int(challenge["continuation_max_label_ticks"]),
            protocol_seed=int(challenge["seeds"]["labeling"]),
        )
    else:
        labeling = _read(labels_dir / "summary.json")
        if labeling.get("status") != "completed":
            raise RuntimeError("v3c existing labeling is not completed")

    logical_rows = _logical_rows(
        labels_dir=labels_dir,
        output_dir=output_dir,
        role="acceptance",
        labeling=labeling,
    )
    support = _challenge_gate(logical_rows)

    manifest = {
        "schema": ROLE_SCHEMA,
        "status": "completed",
        "iteration": int(plan["iteration"]),
        "role": "acceptance",
        "logical_split": "acceptance",
        "legacy_low_level_split_marker": "train",
        "legacy_marker_is_not_logical_data_role": True,
        "plan": str(source_plan_path),
        "plan_sha256": str(plan["plan_sha256"]),
        "selected_policy_sha256": str(plan["selected_policy_sha256"]),
        "policy_actor_sha256": str(record["actor_sha256"]),
        "policy_payload_sha256": str(record["payload_sha256"]),
        "source_tube_manifest_sha256": str(artifact.manifest["manifest_sha256"]),
        "phase_specific_probe_panels": {
            phase: dict(challenge["phase_probe_panels"][phase]) for phase in PHASES
        },
        "acceptance_challenge_plan": str(challenge_plan_path),
        "acceptance_challenge_plan_sha256": str(challenge["challenge_plan_sha256"]),
        "source_failed_acceptance_root": str(failed_root),
        "source_failed_acceptance_retained_as_failure_evidence": True,
        "logical_labels": str(output_dir / "logical_labels.json"),
        "logical_labels_file_sha256": file_sha256(output_dir / "logical_labels.json"),
        "source_acquisition_catalog": str(acquisition_dir / "catalog.json"),
        "source_acquisition_catalog_sha256": file_sha256(acquisition_dir / "catalog.json"),
        "source_label_summary": str(labels_dir / "summary.json"),
        "source_label_summary_sha256": file_sha256(labels_dir / "summary.json"),
        "phase_counts": support["phase_counts"],
        "total_negative_parent_group_count": support["total_negative_parent_group_count"],
        "acceptance_challenge_gate": support["gate"],
        "environment_interactions": int(acquisition["environment_interactions"])
        + int(labeling["environment_interactions"]),
        "training_transitions": 0,
        "expert_switching_used": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "claim_boundary": {
            "role_locked_before_next_policy_training": True,
            "fresh_candidate_blind_acceptance_bank": True,
            "candidate_policy_outcomes_inspected": False,
            "train_rows_may_fit_fields": False,
            "rows_may_calibrate_thresholds": False,
            "rows_may_gate_next_policy": True,
            "rows_may_enter_tube": False,
            "original_v3_acceptance_pass_claim": False,
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
    prepare.add_argument("--failed-acceptance-root", type=Path, required=True)
    prepare.add_argument("--source-workflow", type=Path, required=True)
    prepare.add_argument("--challenge-plan-out", type=Path, required=True)
    prepare.add_argument("--challenge-root", type=Path, required=True)
    prepare.add_argument("--workflow-out", type=Path, required=True)
    prepare.add_argument("--state-dir", type=Path, required=True)

    run = subs.add_parser("run")
    run.add_argument("--challenge-plan", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "prepare":
        result = _prepare(
            source_plan=args.source_plan,
            failed_acceptance_root=args.failed_acceptance_root,
            source_workflow=args.source_workflow,
            challenge_plan_out=args.challenge_plan_out,
            challenge_root=args.challenge_root,
            workflow_out=args.workflow_out,
            state_dir=args.state_dir,
        )
    else:
        result = _run(
            challenge_plan_path=args.challenge_plan,
            output_dir=args.output_dir,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
