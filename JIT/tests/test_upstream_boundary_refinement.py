from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _write(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_action_basis_can_focus_on_discovered_hip_positive_direction():
    from jit_dvgc.upstream_boundary import action_basis_directions

    directions = action_basis_directions(action_names=("hip",), signs=(1,))
    assert len(directions) == 1
    row = directions[0]
    assert row["action_name"] == "hip"
    assert row["action_dimension"] == 2
    assert row["sign"] == 1
    assert row["basis_vector"] == [0.0, 0.0, 1.0, 0.0]


def test_focused_collection_attempts_only_selected_direction(monkeypatch, tmp_path):
    import jit_dvgc.upstream_boundary as ub

    snapshot = SimpleNamespace(
        qpos=np.asarray([0.0, 0.0, 0.1, 0.0, 0.01], dtype=np.float32),
        qvel=np.asarray([0.7, 0.0, 0.1, 0.0, 0.8], dtype=np.float32),
        observation=np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
        tick=34,
        parent_trajectory="transition_4988928__1000001",
    )
    anchor = ub.AuditedAnchor(
        {
            "source_bank": "source_4988928",
            "parent_group_id": "transition_4988928__1000001",
            "seed": 1000001,
            "role": "ascending_entry",
            "tick": 34,
            "state_sha256": "anchor",
            "snapshot": "snapshots/anchor",
            "source_checkpoint": "checkpoint",
            "source_training_transitions": 4988928,
            "source_actor_sha256": "a" * 64,
        },
        {
            "split": "train",
            "success_count": 0,
            "branch_count": 1,
            "branches": [{"reason": "pitch_limit"}],
        },
        snapshot,
        "failure_anchor",
    )

    monkeypatch.setattr(ub, "save_snapshot", lambda *_args, **_kwargs: None)

    class Events:
        apex_seen = False

    class State:
        def __init__(self, tick=34):
            self.tick = tick
            self.info = {
                "events": Events(),
                "terminated": False,
                "truncated": False,
                "timeout": False,
            }

    captured = {"count": 0}

    def capture(state, _anchor):
        captured["count"] += 1
        value = captured["count"] * 0.01
        return SimpleNamespace(
            qpos=np.asarray([value, 0.0, 0.1, 0.0, 0.01], dtype=np.float32),
            qvel=np.asarray([0.7, 0.0, 0.1, 0.0, 0.8], dtype=np.float32),
            observation=np.asarray([value, 1.0, 2.0], dtype=np.float32),
            tick=state.tick,
            parent_trajectory="transition_4988928__1000001",
        )

    report = ub.collect_reachable_boundary_candidates(
        [anchor],
        tmp_path / "out",
        restore=lambda _snapshot: State(),
        policy_action=lambda _state, _variant, _step: np.zeros(4, dtype=np.float32),
        step=lambda state, _action: State(state.tick + 1),
        capture=capture,
        protocol={"protocol_seed": 1},
        strengths=(0.05,),
        durations=(1,),
        action_names=("hip",),
        signs=(1,),
    )
    assert report["attempted_candidate_count"] == 1
    assert report["candidate_count"] == 1
    assert report["selection"] == "train_unique_reachable_boundary_action_basis_subset"
    perturbation = report["entries"][0]["perturbation"]
    assert perturbation["action_name"] == "hip"
    assert perturbation["sign"] == 1
    protocol = json.loads((tmp_path / "out" / "protocol.json").read_text())
    assert protocol["selected_action_names"] == ["hip"]
    assert protocol["selected_signs"] == [1]


def _write_boundary_case(tmp_path: Path, outcomes):
    entries = []
    labels = []
    for index, (strength, success) in enumerate(outcomes):
        state = f"state-{index}"
        entries.append(
            {
                "source_bank": "boundary_bank",
                "parent_group_id": "transition_1__1000001",
                "seed": 1000001,
                "role": "ascending_entry",
                "tick": 34 + index,
                "state_sha256": state,
                "protocol_sha256": "protocol",
                "anchor_kind": "failure_anchor",
                "anchor_parent_group_id": "transition_1__1000001",
                "anchor_state_sha256": "anchor",
                "anchor_role": "ascending_entry",
                "perturbation": {
                    "action_name": "hip",
                    "sign": 1,
                    "strength": strength,
                    "duration": 1,
                },
            }
        )
        labels.append(
            {
                "boundary_candidate_index": index,
                "source_bank": "boundary_bank",
                "parent_group_id": "transition_1__1000001",
                "seed": 1000001,
                "state_sha256": state,
                "success_count": int(success),
                "boundary_protocol_sha256": "protocol",
                "branches": [
                    {"reason": "apex_success" if success else "pitch_limit"}
                ],
            }
        )
    catalog = tmp_path / "catalog.json"
    label_path = tmp_path / "labels.json"
    _write(catalog, {"entries": entries})
    _write(label_path, labels)
    return catalog, label_path


def test_sparse_boundary_evidence_requests_train_only_refinement(tmp_path):
    from jit_dvgc.upstream_boundary_analysis import analyze_boundary_pilot

    catalog, labels = _write_boundary_case(
        tmp_path,
        [(0.025, False), (0.05, False), (0.10, True)],
    )
    report = analyze_boundary_pilot(catalog, labels)
    assert report["decision"] == "BOUNDARY_FOUND"
    assert report["boundary_evidence"]
    assert not report["dataset_lock_ready"]
    assert report["next_step"] == "refine_discovered_crossing_direction_train_only"
    assert report["failure_anchor_crossing_brackets"] == [
        {
            "action_name": "hip",
            "sign": 1,
            "lower_failure_strength": 0.05,
            "upper_success_strength": 0.1,
            "width": 0.05,
        }
    ]


def test_dense_boundary_evidence_can_pass_operational_lock_gate(tmp_path):
    from jit_dvgc.upstream_boundary_analysis import analyze_boundary_pilot

    outcomes = [(0.05, False)] * 4 + [(0.10, True)] * 4
    catalog, labels = _write_boundary_case(tmp_path, outcomes)
    report = analyze_boundary_pilot(catalog, labels)
    assert report["boundary_evidence"]
    assert report["dataset_lock_ready"]
    assert report["next_step"] == "lock_train_protocol_then_validate_without_retuning"
