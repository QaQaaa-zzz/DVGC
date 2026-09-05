from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@dataclass
class _Events:
    apex_seen: bool = False


class _State:
    def __init__(self, tick=0, *, terminal=False, apex=False, marker=0.0):
        self.tick = tick
        self.marker = float(marker)
        self.info = {
            "events": _Events(apex_seen=apex),
            "terminated": terminal,
            "truncated": False,
            "timeout": False,
        }


def _snapshot(value: float, *, tick: int = 34, parent="transition_4988928__1000001"):
    return SimpleNamespace(
        qpos=np.asarray([value, 0.0, 0.1, 0.0, 0.01], dtype=np.float32),
        qvel=np.asarray([0.7, 0.0, 0.1, 0.0, 0.8 + value], dtype=np.float32),
        observation=np.asarray([value, 1.0, 2.0], dtype=np.float32),
        tick=tick,
        parent_trajectory=parent,
    )


def _write_nominal(tmp_path: Path):
    catalog_dir = tmp_path / "nominal"
    catalog_dir.mkdir()
    entries = []
    labels = []
    snapshot_map = {}
    for seed, split, delta in [
        (1000001, "train", 0.0),
        (1000002, "train", 1.0e-7),
        (1000006, "validation", 2.0e-7),
        (1000007, "test", 3.0e-7),
    ]:
        parent = f"transition_4988928__{seed}"
        for role, success, reason, tick in [
            ("jump_zone_entry", True, "apex_success", 30),
            ("ascending_entry", False, "pitch_limit", 34),
            ("height_entry", True, "apex_success", 38),
        ]:
            state_hash = f"{seed}_{role}"
            row = {
                "source_bank": "source_4988928",
                "parent_group_id": parent,
                "seed": seed,
                "role": role,
                "tick": tick,
                "state_sha256": state_hash,
                "snapshot": f"snapshots/{state_hash}",
                "source_training_transitions": 4988928,
                "source_checkpoint": "checkpoint",
                "source_actor_sha256": "a" * 64,
            }
            entries.append(row)
            labels.append(
                {
                    **{key: row[key] for key in ("source_bank", "parent_group_id", "seed", "role", "tick", "state_sha256", "snapshot")},
                    "split": split,
                    "branch_count": 1,
                    "success_count": int(success),
                    "branches": [{"reason": reason}],
                }
            )
            base = delta if role == "ascending_entry" else delta + (0.01 if role == "jump_zone_entry" else 0.02)
            snapshot_map[str(catalog_dir / row["source_bank"] / row["snapshot"])] = _snapshot(base, tick=tick, parent=parent)
    catalog = catalog_dir / "catalog.json"
    label_path = catalog_dir / "labels.json"
    catalog.write_text(json.dumps({"entries": entries}))
    label_path.write_text(json.dumps(labels))
    return catalog, label_path, snapshot_map


def test_action_basis_is_deterministic_and_symmetric():
    from jit_dvgc.upstream_boundary import action_basis_directions

    first = action_basis_directions()
    second = action_basis_directions()
    assert first == second
    assert len(first) == 8
    for dimension in range(4):
        subset = [row for row in first if row["action_dimension"] == dimension]
        assert {row["sign"] for row in subset} == {-1, 1}
        assert all(sum(abs(x) for x in row["basis_vector"]) == 1.0 for row in subset)


def test_strength_validation():
    from jit_dvgc.upstream_boundary import validate_strengths

    assert validate_strengths([0.025, 0.05, 0.1]) == (0.025, 0.05, 0.1)
    with pytest.raises(ValueError):
        validate_strengths([0.0])
    with pytest.raises(ValueError):
        validate_strengths([1.1])
    with pytest.raises(ValueError):
        validate_strengths([0.1, 0.1])


def test_duration_validation():
    from jit_dvgc.upstream_boundary import validate_durations

    assert validate_durations([1, 2]) == (1, 2)
    with pytest.raises(ValueError):
        validate_durations([0])
    with pytest.raises(ValueError):
        validate_durations([1, 1])


def test_near_duplicate_detection_is_deterministic():
    from jit_dvgc.upstream_boundary import are_near_duplicates, snapshot_distance

    left = _snapshot(0.0)
    right = _snapshot(1.0e-7)
    assert are_near_duplicates(left, right, qpos_atol=1e-5, qvel_atol=1e-5, observation_atol=1e-5)
    assert snapshot_distance(left, right) == snapshot_distance(left, right)
    assert not are_near_duplicates(left, _snapshot(0.1), qpos_atol=1e-5, qvel_atol=1e-5, observation_atol=1e-5)


def test_train_only_anchor_selection_and_representative(monkeypatch, tmp_path):
    import jit_dvgc.upstream_boundary as ub

    catalog, labels, snapshot_map = _write_nominal(tmp_path)
    monkeypatch.setattr(ub, "load_snapshot", lambda path: snapshot_map[str(path)])
    selected, audit = ub.audit_and_select_train_anchors(
        catalog,
        labels,
        train_seeds=(1000001, 1000002),
        source_training_transitions=4988928,
        max_negative_anchors=1,
    )
    failure = [item for item in selected if item.anchor_kind == "failure_anchor"]
    guards = [item for item in selected if item.anchor_kind == "positive_guard"]
    assert [item.row["seed"] for item in failure] == [1000001]
    assert all(item.label["split"] == "train" for item in selected)
    assert {item.row["role"] for item in guards} == {"jump_zone_entry", "height_entry"}
    assert audit["near_duplicate_cluster_count"] == 1
    assert audit["near_duplicate_clusters"][0]["member_seeds"] == [1000001, 1000002]


def test_validation_and_test_seeds_never_enter_train_pilot(monkeypatch, tmp_path):
    import jit_dvgc.upstream_boundary as ub

    catalog, labels, snapshot_map = _write_nominal(tmp_path)
    monkeypatch.setattr(ub, "load_snapshot", lambda path: snapshot_map[str(path)])
    selected, audit = ub.audit_and_select_train_anchors(
        catalog,
        labels,
        train_seeds=(1000001, 1000002),
        source_training_transitions=4988928,
    )
    assert {item.row["seed"] for item in selected} <= {1000001, 1000002}
    assert 1000006 not in audit["train_seeds"]
    assert 1000007 not in audit["train_seeds"]


def test_train_label_with_nontrain_seed_is_rejected(monkeypatch, tmp_path):
    import jit_dvgc.upstream_boundary as ub

    catalog, labels, snapshot_map = _write_nominal(tmp_path)
    rows = json.loads(labels.read_text())
    for row in rows:
        if row["seed"] == 1000006:
            row["split"] = "train"
    labels.write_text(json.dumps(rows))
    monkeypatch.setattr(ub, "load_snapshot", lambda path: snapshot_map[str(path)])
    with pytest.raises(ValueError, match="outside train_seeds"):
        ub.audit_and_select_train_anchors(
            catalog,
            labels,
            train_seeds=(1000001, 1000002),
            source_training_transitions=4988928,
        )


def _anchor(snapshot=None, *, role="ascending_entry"):
    from jit_dvgc.upstream_boundary import AuditedAnchor

    row = {
        "source_bank": "source_4988928",
        "parent_group_id": "transition_4988928__1000001",
        "seed": 1000001,
        "role": role,
        "tick": 34,
        "state_sha256": "anchor_hash",
        "snapshot": "snapshots/anchor",
        "source_checkpoint": "checkpoint",
        "source_training_transitions": 4988928,
        "source_actor_sha256": "b" * 64,
    }
    label = {"split": "train", "success_count": 0, "branch_count": 1, "branches": [{"reason": "pitch_limit"}]}
    return AuditedAnchor(row, label, _snapshot(0.0) if snapshot is None else snapshot, "failure_anchor")


def _run_collection(monkeypatch, tmp_path, *, step_behavior=None, capture_values=None, strengths=(0.05,), durations=(1,)):
    import jit_dvgc.upstream_boundary as ub

    saved = []
    monkeypatch.setattr(ub, "save_snapshot", lambda path, snapshot: saved.append((path, snapshot)))
    anchor = _anchor()
    capture_values = iter(capture_values or [0.1] * 100)

    def restore(_snapshot):
        return _State(34, marker=0.0)

    def policy_action(_state, _variant, _step):
        return np.zeros(4, dtype=np.float32)

    def step(state, action):
        if step_behavior is not None:
            return step_behavior(state, action)
        return _State(state.tick + 1, marker=float(np.sum(action)))

    def capture(state, selected_anchor):
        value = next(capture_values)
        return _snapshot(value, tick=state.tick, parent=selected_anchor.row["parent_group_id"])

    report = ub.collect_reachable_boundary_candidates(
        [anchor],
        tmp_path / "out",
        restore=restore,
        policy_action=policy_action,
        step=step,
        capture=capture,
        protocol={"protocol_seed": 820401},
        strengths=strengths,
        durations=durations,
    )
    return report, saved


def test_parent_lineage_and_provenance_are_preserved(monkeypatch, tmp_path):
    report, saved = _run_collection(monkeypatch, tmp_path, capture_values=[0.1 + i * 0.01 for i in range(8)])
    assert report["candidate_count"] == 8
    entry = report["entries"][0]
    assert entry["parent_group_id"] == "transition_4988928__1000001"
    assert entry["anchor_parent_group_id"] == entry["parent_group_id"]
    assert entry["anchor_seed"] == 1000001
    assert entry["anchor_role"] == "ascending_entry"
    assert entry["anchor_source_actor_sha256"] == "b" * 64
    assert entry["anchor_state_sha256"] == "anchor_hash"
    assert entry["protocol_sha256"]
    assert len(saved) == 8


def test_terminal_during_perturbation_is_excluded(monkeypatch, tmp_path):
    def terminal_step(state, _action):
        return _State(state.tick + 1, terminal=True)

    report, _ = _run_collection(monkeypatch, tmp_path, step_behavior=terminal_step)
    assert report["candidate_count"] == 0
    assert report["exclusion_counts"]["terminal"] == 8


def test_apex_during_perturbation_is_excluded(monkeypatch, tmp_path):
    def apex_step(state, _action):
        return _State(state.tick + 1, apex=True)

    report, _ = _run_collection(monkeypatch, tmp_path, step_behavior=apex_step)
    assert report["candidate_count"] == 0
    assert report["exclusion_counts"]["apex"] == 8


def test_physical_duplicate_candidates_are_deduplicated(monkeypatch, tmp_path):
    report, saved = _run_collection(monkeypatch, tmp_path, capture_values=[0.1] * 8)
    assert report["candidate_count"] == 1
    assert report["exclusion_counts"]["duplicate"] == 7
    assert len(saved) == 1


def test_training_transition_accounting_stays_zero(monkeypatch, tmp_path):
    report, _ = _run_collection(monkeypatch, tmp_path, capture_values=[0.1 + i * 0.01 for i in range(8)])
    assert report["training_transitions"] == 0
    protocol = json.loads((tmp_path / "out" / "protocol.json").read_text())
    assert protocol["training_transitions"] == 0
    assert report["collection_transitions"] == 8


def test_perturbation_action_is_clipped_and_records_effective_delta(monkeypatch, tmp_path):
    import jit_dvgc.upstream_boundary as ub

    actions = []
    monkeypatch.setattr(ub, "save_snapshot", lambda *_args, **_kwargs: None)

    def restore(_snapshot):
        return _State(34)

    def policy_action(_state, _variant, _step):
        return np.asarray([0.99, 0.0, 0.0, 0.0], dtype=np.float32)

    def step(state, action):
        actions.append(np.asarray(action))
        return _State(state.tick + 1)

    counter = {"value": 0}
    def capture(state, anchor):
        counter["value"] += 1
        return _snapshot(counter["value"] * 0.01, tick=state.tick, parent=anchor.row["parent_group_id"])

    report = ub.collect_reachable_boundary_candidates(
        [_anchor()],
        tmp_path / "out",
        restore=restore,
        policy_action=policy_action,
        step=step,
        capture=capture,
        protocol={"protocol_seed": 1},
        strengths=(0.05,),
        durations=(1,),
    )
    assert all(np.max(action) <= 1.0 for action in actions)
    steer_plus = next(
        row for row in report["entries"]
        if row["perturbation"]["action_dimension"] == 0 and row["perturbation"]["sign"] == 1
    )
    assert steer_plus["perturbation"]["perturbed_actions"][0][0] == pytest.approx(1.0)
    assert steer_plus["perturbation"]["effective_deltas"][0][0] == pytest.approx(0.01, abs=1e-6)


def test_candidate_catalog_keeps_labeler_required_fields(monkeypatch, tmp_path):
    report, _ = _run_collection(monkeypatch, tmp_path, capture_values=[0.1 + i * 0.01 for i in range(8)])
    required = {"source_bank", "parent_group_id", "seed", "role", "tick", "snapshot", "state_sha256"}
    assert required <= set(report["entries"][0])
