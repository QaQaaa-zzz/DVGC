from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from jit_dvgc.acquisition import select_disjoint_tube_boundary_anchors
from jit_dvgc.unified_boundary import select_tube_boundary_anchors
from jit_dvgc.unified_envelope_snapshot import (
    DOWN_EVENT_FIELDS,
    UP_EVENT_FIELDS,
    UnifiedEnvelopeSnapshot,
    load_unified_envelope_snapshot,
    physical_state_sha256,
    save_unified_envelope_snapshot,
)


def _entry(phase: str, index: int, score: float, group: str, *, split: str = "train"):
    return {
        "phase": phase,
        "split": split,
        "state_sha256": f"{index + (0 if phase == 'upstream' else 100):064x}",
        "parent_group_id": group,
        "value_score": score,
        "sampling_weight": 0.05 + 0.95 * score,
        "role": f"role_{index}",
        "source_bank": f"bank_{phase}",
    }


def _artifact(entries, **manifest_overrides):
    manifest = {
        "schema": "jit_soft_tube_v1",
        "status": "completed",
        "training_guidance_only": True,
        "certified_safe": False,
        "test_data_used": False,
        "validation_data_used": False,
        "manifest_sha256": "a" * 64,
        "entry_count": len(entries),
    }
    manifest.update(manifest_overrides)
    return SimpleNamespace(manifest=manifest, entries=tuple(entries))


def test_boundary_anchor_selection_is_weak_score_group_unique_and_phase_local():
    entries = [
        _entry("upstream", 0, 0.10, "u0"),
        _entry("upstream", 1, 0.11, "u0"),
        _entry("upstream", 2, 0.20, "u1"),
        _entry("upstream", 3, 0.99, "u2"),
        _entry("downstream", 0, 0.05, "d0"),
        _entry("downstream", 1, 0.06, "d0"),
        _entry("downstream", 2, 0.07, "d1"),
        _entry("downstream", 3, 0.90, "d2"),
    ]
    anchors, audit = select_tube_boundary_anchors(
        _artifact(entries),
        max_per_phase=8,
        frontier_score_ceiling=0.5,
    )

    assert [(a.phase, a.entry_index, a.parent_group_id) for a in anchors] == [
        ("upstream", 0, "u0"),
        ("upstream", 2, "u1"),
        ("downstream", 0, "d0"),
        ("downstream", 2, "d1"),
    ]
    assert [a.global_index for a in anchors] == [0, 2, 4, 6]
    assert all(a.value_score <= 0.5 for a in anchors)
    assert audit["split"] == "train"
    assert audit["selection"] == (
        "bootstrap_score_at_or_below_ceiling_parent_group_unique_state_unique"
    )
    assert audit["anchor_semantics"] == "weak_bootstrap_frontier_probe_not_certified_boundary"
    assert audit["frontier_score_ceiling"] == pytest.approx(0.5)
    assert audit["selected_anchor_count"] == 4
    assert audit["by_phase"]["upstream"]["eligible_support_count"] == 3
    assert audit["by_phase"]["upstream"]["eligible_parent_group_count"] == 2
    assert audit["by_phase"]["upstream"]["excluded_above_score_ceiling_count"] == 1
    assert audit["by_phase"]["downstream"]["eligible_support_count"] == 3
    assert audit["by_phase"]["downstream"]["eligible_parent_group_count"] == 2
    assert audit["by_phase"]["downstream"]["excluded_above_score_ceiling_count"] == 1
    assert audit["test_data_used"] is False
    assert audit["validation_data_used"] is False


def test_disjoint_boundary_selection_excludes_consumed_states_and_parent_groups():
    entries = [
        _entry("upstream", 0, 0.01, "u0"),
        _entry("upstream", 1, 0.02, "u1"),
        _entry("upstream", 2, 0.03, "u2"),
        _entry("upstream", 3, 0.04, "u3"),
        _entry("downstream", 0, 0.01, "d0"),
        _entry("downstream", 1, 0.02, "d1"),
        _entry("downstream", 2, 0.03, "d2"),
        _entry("downstream", 3, 0.04, "d3"),
    ]
    artifact = _artifact(entries)
    anchors, audit = select_disjoint_tube_boundary_anchors(
        artifact,
        max_per_phase=2,
        minimum_per_phase=2,
        frontier_score_ceiling=0.5,
        excluded_state_sha256=(entries[0]["state_sha256"],),
        excluded_parent_groups={"upstream": ("u1",), "downstream": ("d0",)},
    )

    assert [(row.phase, row.parent_group_id) for row in anchors] == [
        ("upstream", "u2"),
        ("upstream", "u3"),
        ("downstream", "d1"),
        ("downstream", "d2"),
    ]
    assert audit["selected_phase_counts"] == {"downstream": 2, "upstream": 2}
    assert audit["excluded_state_sha256_count"] == 1
    assert audit["excluded_parent_group_counts"] == {"upstream": 1, "downstream": 1}
    assert audit["by_phase"]["upstream"]["excluded_consumed_state_count"] == 1
    assert audit["by_phase"]["upstream"]["excluded_consumed_parent_group_count"] == 1
    assert audit["by_phase"]["downstream"]["excluded_consumed_parent_group_count"] == 1
    assert audit["test_data_used"] is False
    assert audit["validation_data_used"] is False
    assert audit["final_evaluation_data_used"] is False

    with pytest.raises(ValueError, match="disjoint frontier has only"):
        select_disjoint_tube_boundary_anchors(
            artifact,
            max_per_phase=3,
            minimum_per_phase=3,
            frontier_score_ceiling=0.5,
            excluded_parent_groups={
                "upstream": ("u0", "u1"),
                "downstream": ("d0", "d1"),
            },
        )


def test_boundary_anchor_selection_never_fills_quota_with_core_or_same_parent_group():
    entries = [
        _entry("upstream", 0, 0.01, "u0"),
        _entry("upstream", 1, 0.02, "u0"),
        _entry("upstream", 2, 0.03, "u0"),
        _entry("upstream", 3, 0.95, "u1"),
        _entry("downstream", 0, 0.04, "d0"),
        _entry("downstream", 1, 0.05, "d0"),
        _entry("downstream", 2, 0.96, "d1"),
    ]
    anchors, audit = select_tube_boundary_anchors(
        _artifact(entries), max_per_phase=8, frontier_score_ceiling=0.5
    )

    assert [(a.phase, a.parent_group_id, a.value_score) for a in anchors] == [
        ("upstream", "u0", 0.01),
        ("downstream", "d0", 0.04),
    ]
    assert audit["selected_anchor_count"] == 2
    assert audit["by_phase"]["upstream"]["selected_count"] == 1
    assert audit["by_phase"]["downstream"]["selected_count"] == 1


def test_boundary_anchor_selection_rejects_nontrain_leaky_or_invalid_ceiling():
    entries = [
        _entry("upstream", 0, 0.2, "u0"),
        _entry("downstream", 0, 0.2, "d0", split="validation"),
    ]
    with pytest.raises(ValueError, match="non-TRAIN"):
        select_tube_boundary_anchors(_artifact(entries), max_per_phase=1)

    clean = [
        _entry("upstream", 0, 0.2, "u0"),
        _entry("downstream", 0, 0.2, "d0"),
    ]
    with pytest.raises(ValueError, match="not TRAIN-only"):
        select_tube_boundary_anchors(
            _artifact(clean, validation_data_used=True),
            max_per_phase=1,
        )
    with pytest.raises(ValueError, match="frontier_score_ceiling"):
        select_tube_boundary_anchors(
            _artifact(clean), max_per_phase=1, frontier_score_ceiling=1.1
        )


def _events(fields):
    values = {}
    for index, name in enumerate(fields):
        if name in {"stuck_anchor_x", "contact_x"}:
            values[name] = np.asarray(1.5 + index, np.float32)
        elif name in {"stuck_ticks", "episode_step", "post_contact_ticks"}:
            values[name] = np.asarray(index, np.int32)
        else:
            values[name] = np.asarray(bool(index % 2))
    return values


def test_unified_envelope_snapshot_roundtrip_preserves_policy_and_phase_identity(tmp_path):
    snapshot = UnifiedEnvelopeSnapshot(
        qpos=np.arange(9, dtype=np.float32),
        qvel=np.arange(8, dtype=np.float32),
        observation_fifo=np.arange(75, dtype=np.float32).reshape(3, 25),
        history_valid_count=3,
        observation=np.arange(76, dtype=np.float32),
        last_action=np.zeros(4, dtype=np.float32),
        ctrl=np.zeros(4, dtype=np.float32),
        rng=np.asarray([1, 2], dtype=np.uint32),
        up_events=_events(UP_EVENT_FIELDS),
        down_events=_events(DOWN_EVENT_FIELDS),
        active_phase=1,
        start_phase=0,
        phase_transitioned=True,
        episode_step=17,
        phase_episode_step=4,
        episode_return=3.25,
        reset_from_soft_tube=True,
        source_tick=9,
        parent_group_index=2,
        tube_entry_index=3,
        tube_global_index=8,
        parent_trajectory="parent",
        parent_state_sha256="1" * 64,
        config_sha256="2" * 64,
        xml_sha256="3" * 64,
        policy_actor_sha256="4" * 64,
        policy_payload_sha256="5" * 64,
        policy_iteration=0,
        compatibility_identity={"contract": "test"},
    )
    path = tmp_path / "snapshot"
    save_unified_envelope_snapshot(path, snapshot)
    loaded = load_unified_envelope_snapshot(path)

    assert physical_state_sha256(loaded) == physical_state_sha256(snapshot)
    assert loaded.policy_actor_sha256 == "4" * 64
    assert loaded.policy_payload_sha256 == "5" * 64
    assert loaded.active_phase == 1
    assert loaded.start_phase == 0
    assert loaded.phase_transitioned is True
    assert loaded.episode_step == 17
    assert loaded.phase_episode_step == 4
    assert loaded.episode_return == pytest.approx(3.25)
    np.testing.assert_array_equal(loaded.observation_fifo, snapshot.observation_fifo)
    assert tuple(loaded.up_events) == UP_EVENT_FIELDS
    assert tuple(loaded.down_events) == DOWN_EVENT_FIELDS
