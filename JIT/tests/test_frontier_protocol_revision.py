from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from jit_dvgc.iterative_frontier_protocol import (
    _acquisition_phase_support,
    canonical_sha256,
)


def _catalog(up_count: int, down_count: int, *, groups_per_phase: int = 3):
    entries = []
    for phase, count in (("upstream", up_count), ("downstream", down_count)):
        for index in range(count):
            entries.append(
                {
                    "phase": phase,
                    "parent_group_id": f"{phase}-g{index % groups_per_phase}",
                }
            )
    return {
        "entries": entries,
        "exclusion_counts": {"terminal": 7},
    }


def test_train_acquisition_preflight_stops_before_labeling_on_zero_downstream() -> None:
    with pytest.raises(ValueError, match="DOWNSTREAM|downstream"):
        _acquisition_phase_support(_catalog(80, 0), role="train")


def test_train_acquisition_preflight_requires_logically_possible_label_count() -> None:
    with pytest.raises(ValueError, match="before continuation labeling"):
        _acquisition_phase_support(_catalog(39, 80), role="train")


def test_train_acquisition_preflight_accepts_structurally_possible_two_phase_bank() -> None:
    support = _acquisition_phase_support(_catalog(80, 80), role="train")
    assert support["upstream"]["candidate_count"] == 80
    assert support["downstream"]["candidate_count"] == 80
    assert support["upstream"]["parent_group_count"] == 3
    assert support["downstream"]["parent_group_count"] == 3


def test_local_horizon_v2_revision_preserves_identity_and_changes_only_probe_time(tmp_path: Path) -> None:
    source = tmp_path / "frontier_v1.json"
    output = tmp_path / "frontier_v2.json"
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
        "frontier_definition": "newest_expansion_shell_only_lowest_score_parent_unique",
        "role_pattern": ["train", "train", "train", "calibration", "acceptance"],
        "role_semantics": {},
        "role_parent_group_counts": {},
        "fixed_probe_panel": {
            "action_names": ["steer", "rear_wheel_drive", "hip", "knee"],
            "signs": [-1, 1],
            "strengths": [0.025, 0.05, 0.10],
            "durations": [4, 8, 16, 32],
            "max_label_ticks": 400,
        },
        "seeds": {"train": {"acquisition": 1, "labeling": 2}},
        "anchors": [{"role": "train", "phase": "upstream"}],
        "training_transitions": 0,
        "expert_switching_used": False,
        "validation_data_used": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "claim_boundary": {},
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    source.write_text(json.dumps(plan), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "JIT/cli/run_iterative_frontier_protocol.py",
            "revise-plan-local-horizon-v2",
            "--source-plan",
            str(source),
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
    assert revised["fixed_probe_panel"]["durations"] == [1, 2, 4, 8]
    assert revised["fixed_probe_panel"]["strengths"] == plan["fixed_probe_panel"]["strengths"]
    assert revised["anchors"] == plan["anchors"]
    assert revised["seeds"] == plan["seeds"]
    assert revised["protocol_revision"]["supersedes_plan_sha256"] == plan["plan_sha256"]
    declared = revised.pop("plan_sha256")
    assert canonical_sha256(revised) == declared
