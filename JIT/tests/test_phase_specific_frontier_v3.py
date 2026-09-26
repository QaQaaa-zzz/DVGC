from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from jit_dvgc.config import file_sha256
from jit_dvgc.iterative_frontier_protocol import canonical_sha256
from jit_dvgc.phase_specific_frontier import (
    _merged_acquisition_payloads,
    panel_variant_count,
    required_label_shard_count,
)


def _v2_plan() -> dict:
    plan = {
        "schema": "jit_iterative_frontier_plan_v1",
        "status": "predeclared_before_frontier_outcomes",
        "iteration": 1,
        "policy_name": "pi_1",
        "selected_policy": "selected.json",
        "selected_policy_sha256": "a" * 64,
        "policy_actor_sha256": "b" * 64,
        "policy_payload_sha256": "c" * 64,
        "source_tube": "tube1",
        "source_tube_manifest_sha256": "d" * 64,
        "source_tube_entry_count": 3119,
        "source_tube_core_retained_count": 222,
        "frontier_definition": (
            "newest_expansion_shell_only_lowest_score_parent_unique_local_horizon_v2"
        ),
        "role_pattern": ["train", "train", "train", "calibration", "acceptance"],
        "role_semantics": {},
        "role_parent_group_counts": {},
        "fixed_probe_panel": {
            "action_names": ["steer", "rear_wheel_drive", "hip", "knee"],
            "signs": [-1, 1],
            "strengths": [0.025, 0.05, 0.10],
            "durations": [1, 2, 4, 8],
            "max_label_ticks": 400,
        },
        "seeds": {
            "train": {"acquisition": 11, "labeling": 12},
            "calibration": {"acquisition": 21, "labeling": 22},
            "acceptance": {"acquisition": 31, "labeling": 32},
        },
        "anchors": [
            {"role": "train", "phase": "upstream"},
            {"role": "train", "phase": "downstream"},
        ],
        "training_transitions": 0,
        "expert_switching_used": False,
        "validation_data_used": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "claim_boundary": {},
        "protocol_revision": {
            "name": "local_horizon_v2",
            "supersedes_plan": "frontier_v1.json",
            "supersedes_plan_sha256": "e" * 64,
            "revision_predeclared_before_v2_outcomes": True,
        },
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def _diagnostic(plan: dict) -> dict:
    return {
        "schema": "jit_frontier_support_diagnostics_v1",
        "status": "completed",
        "artifact_role": "post_failure_read_only_probe_support_diagnostic",
        "role_root": "v2/frontier_train",
        "iteration": 1,
        "policy_actor_sha256": plan["policy_actor_sha256"],
        "policy_payload_sha256": plan["policy_payload_sha256"],
        "acquisition_protocol_sha256": "f" * 64,
        "label_protocol_sha256": "1" * 64,
        "candidate_count": 932,
        "strengths": [0.025, 0.05, 0.10],
        "durations": [1, 2, 4, 8],
        "by_phase": {
            "upstream": {
                "candidate_count": 821,
                "positive_count": 785,
                "negative_count": 36,
                "parent_group_count": 9,
            },
            "downstream": {
                "candidate_count": 111,
                "positive_count": 111,
                "negative_count": 0,
                "parent_group_count": 3,
            },
        },
        "training_transitions": 0,
        "environment_interactions": 0,
        "new_labels_generated": 0,
        "role_membership_changed": False,
        "tube_construction_authorized": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "claim_boundary": {},
    }


def test_v3_revision_is_phase_specific_and_preserves_v2_upstream(tmp_path: Path) -> None:
    source = tmp_path / "frontier_v2.json"
    diagnostic_path = tmp_path / "support_diagnostics.json"
    output = tmp_path / "frontier_v3.json"
    plan = _v2_plan()
    diagnostic = _diagnostic(plan)
    source.write_text(json.dumps(plan), encoding="utf-8")
    diagnostic_path.write_text(json.dumps(diagnostic), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "JIT/cli/run_iterative_frontier_protocol.py",
            "revise-plan-phase-specific-two-axis-v3",
            "--source-plan",
            str(source),
            "--v2-support-diagnostic",
            str(diagnostic_path),
            "--output",
            str(output),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    revised = json.loads(output.read_text(encoding="utf-8"))

    assert revised["fixed_probe_panel"] == plan["fixed_probe_panel"]
    assert revised["anchors"] == plan["anchors"]
    assert revised["seeds"] == plan["seeds"]
    assert revised["phase_probe_panels"]["upstream"] == {
        "action_names": ["steer", "rear_wheel_drive", "hip", "knee"],
        "signs": [-1, 1],
        "strengths": [0.025, 0.05, 0.10],
        "durations": [1, 2, 4, 8],
        "active_action_dimensions": 1,
    }
    assert revised["phase_probe_panels"]["downstream"] == {
        "action_names": ["steer", "rear_wheel_drive", "hip", "knee"],
        "signs": [-1, 1],
        "strengths": [0.15, 0.30, 0.50],
        "durations": [2, 4, 8],
        "active_action_dimensions": 2,
    }
    assert revised["label_execution"]["max_candidates_per_independent_process"] == 930
    assert (
        revised["protocol_revision"]["evidence"]["v2_support_diagnostic_file_sha256"]
        == file_sha256(diagnostic_path)
    )
    declared = revised.pop("plan_sha256")
    assert canonical_sha256(revised) == declared


def test_v3_variant_counts_match_single_axis_and_two_axis_panels() -> None:
    plan = _v2_plan()
    upstream = {
        **plan["fixed_probe_panel"],
        "active_action_dimensions": 1,
    }
    upstream.pop("max_label_ticks")
    downstream = {
        "action_names": ["steer", "rear_wheel_drive", "hip", "knee"],
        "signs": [-1, 1],
        "strengths": [0.15, 0.30, 0.50],
        "durations": [2, 4, 8],
        "active_action_dimensions": 2,
    }
    assert panel_variant_count(upstream) == 96
    assert panel_variant_count(downstream) == 216


def test_historical_3720_bank_maps_to_four_independent_930_shards() -> None:
    assert required_label_shard_count(930) == 1
    assert required_label_shard_count(931) == 2
    assert required_label_shard_count(3720) == 4


def _phase_catalog(phase: str, protocol_sha: str) -> dict:
    other = "downstream" if phase == "upstream" else "upstream"
    return {
        "iteration": 1,
        "policy_name": "pi_1",
        "policy_actor_sha256": "b" * 64,
        "policy_payload_sha256": "c" * 64,
        "frozen_unified_manifest_sha256": "2" * 64,
        "source_tube_manifest_sha256": "d" * 64,
        "protocol_sha256": protocol_sha,
        "anchor_count": 1,
        "attempted_candidate_count": 2,
        "candidate_count": 1,
        "phase_attempted_candidate_counts": {phase: 2, other: 0},
        "phase_candidate_counts": {phase: 1, other: 0},
        "phase_exclusion_counts": {
            phase: {
                "terminal": 1,
                "nonfinite": 0,
                "phase_transition": 0,
                "existing_support": 0,
                "duplicate": 0,
            },
            other: {
                "terminal": 0,
                "nonfinite": 0,
                "phase_transition": 0,
                "existing_support": 0,
                "duplicate": 0,
            },
        },
        "exclusion_counts": {
            "terminal": 1,
            "nonfinite": 0,
            "phase_transition": 0,
            "existing_support": 0,
            "duplicate": 0,
        },
        "environment_interactions": 2,
        "maximum_environment_interactions": 4,
        "entries": [
            {
                "candidate_id": f"pi1_{phase}_000000",
                "phase": phase,
                "state_sha256": ("3" if phase == "upstream" else "4") * 64,
                "protocol_sha256": protocol_sha,
                "source_bank": "boundary_bank",
                "snapshot": "snapshots/candidate_000000",
            }
        ],
    }


def test_phase_catalog_merge_preserves_nested_snapshot_location_and_one_root_protocol() -> None:
    plan = _v2_plan()
    plan = {key: value for key, value in plan.items() if key != "plan_sha256"}
    plan["phase_probe_panels"] = {
        "upstream": {
            "action_names": ["steer", "rear_wheel_drive", "hip", "knee"],
            "signs": [-1, 1],
            "strengths": [0.025, 0.05, 0.10],
            "durations": [1, 2, 4, 8],
            "active_action_dimensions": 1,
        },
        "downstream": {
            "action_names": ["steer", "rear_wheel_drive", "hip", "knee"],
            "signs": [-1, 1],
            "strengths": [0.15, 0.30, 0.50],
            "durations": [2, 4, 8],
            "active_action_dimensions": 2,
        },
    }
    plan["protocol_revision"] = {"name": "phase_specific_two_axis_v3"}
    plan["plan_sha256"] = canonical_sha256(plan)

    protocol, catalog = _merged_acquisition_payloads(
        plan_path=Path("frontier_v3.json"),
        plan=plan,
        role="train",
        phase_catalogs={
            "upstream": _phase_catalog("upstream", "5" * 64),
            "downstream": _phase_catalog("downstream", "6" * 64),
        },
    )

    assert catalog["candidate_count"] == 2
    assert catalog["phase_candidate_counts"] == {"upstream": 1, "downstream": 1}
    assert catalog["entries"][0]["source_bank"] == "phase_upstream/boundary_bank"
    assert catalog["entries"][1]["source_bank"] == "phase_downstream/boundary_bank"
    assert all(
        row["protocol_sha256"] == protocol["protocol_sha256"]
        for row in catalog["entries"]
    )
    assert catalog["entries"][0]["phase_acquisition_protocol_sha256"] == "5" * 64
    assert catalog["entries"][1]["phase_acquisition_protocol_sha256"] == "6" * 64
