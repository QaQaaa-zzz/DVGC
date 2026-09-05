from __future__ import annotations

import json
from pathlib import Path

import pytest

from jit_dvgc.acquisition.fresh_parent import (
    baseline_probe_declaration,
    consumed_baseline_probe_exclusions,
    planned_parent_groups,
)


ACTOR = "a" * 64
PAYLOAD = "b" * 64


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _probe(root: Path) -> dict:
    labels = []
    for index, (phase, label) in enumerate(
        (("upstream", 1), ("upstream", 0), ("downstream", 1))
    ):
        labels.append(
            {
                "split": "train",
                "phase": phase,
                "label": label,
                "state_sha256": f"{index + 1:064x}",
                "parent_group_id": f"group_{phase}_{index}",
                "policy_actor_sha256": ACTOR,
                "policy_payload_sha256": PAYLOAD,
            }
        )
    summary = {
        "status": "completed",
        "policy_actor_sha256": ACTOR,
        "policy_payload_sha256": PAYLOAD,
        "protocol_sha256": "protocol",
        "candidate_count": 3,
        "label_count": 3,
        "training_transitions": 0,
        "validation_data_used": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
    }
    _write(root / "labels.json", labels)
    _write(root / "summary.json", summary)
    declaration, _ = baseline_probe_declaration(
        root,
        baseline_actor_sha256=ACTOR,
        baseline_payload_sha256=PAYLOAD,
    )
    return declaration


def test_baseline_only_probe_binding_tracks_exact_states_and_labels(tmp_path: Path):
    declaration = _probe(tmp_path / "probe")
    assert declaration["state_count"] == 3
    assert declaration["positive_count"] == 2
    assert declaration["negative_count"] == 1
    assert declaration["candidate_policy_outcomes_inspected"] is False


def test_consumed_baseline_probe_exclusion_is_exact_state_only(tmp_path: Path):
    declaration = _probe(tmp_path / "probe")
    protocol = {
        "baseline_actor_sha256": ACTOR,
        "baseline_payload_sha256": PAYLOAD,
        "consumed_baseline_probes": [declaration],
    }
    states, audit = consumed_baseline_probe_exclusions(protocol)
    assert states == {f"{value:064x}" for value in (1, 2, 3)}
    assert audit["probe_count"] == 1
    assert audit["union_state_count"] == 3
    assert "parent_groups" not in audit


def test_consumed_baseline_probe_detects_post_binding_mutation(tmp_path: Path):
    root = tmp_path / "probe"
    declaration = _probe(root)
    labels = json.loads((root / "labels.json").read_text())
    labels[0]["label"] = 0
    _write(root / "labels.json", labels)
    protocol = {
        "baseline_actor_sha256": ACTOR,
        "baseline_payload_sha256": PAYLOAD,
        "consumed_baseline_probes": [declaration],
    }
    with pytest.raises(ValueError, match="labels_file_sha256"):
        consumed_baseline_probe_exclusions(protocol)


def test_parent_groups_are_action_direction_groups_not_seed_or_magnitude_groups():
    predeclared = {
        "protocol": {
            "source_iteration": 0,
            "acquisition": {
                "anchor_source": {
                    "type": "natural_action_excitation_handoff_v1",
                    "action_names": ["steer", "rear_wheel_drive", "hip", "knee"],
                    "signs": [-1, 1],
                    "active_action_dimensions": 1,
                    "strength": 0.10,
                    "duration": 2,
                }
            },
        }
    }
    groups = planned_parent_groups(predeclared)
    assert len(groups) == 8
    assert len({row["parent_group_id"] for row in groups}) == 8
    assert all(row["strength"] == pytest.approx(0.10) for row in groups)
    assert all(row["duration"] == 2 for row in groups)
    assert [row["parent_group_id"] for row in groups] == [
        "pi0_natural_excitation_steer_neg",
        "pi0_natural_excitation_steer_pos",
        "pi0_natural_excitation_rear_wheel_drive_neg",
        "pi0_natural_excitation_rear_wheel_drive_pos",
        "pi0_natural_excitation_hip_neg",
        "pi0_natural_excitation_hip_pos",
        "pi0_natural_excitation_knee_neg",
        "pi0_natural_excitation_knee_pos",
    ]
