from __future__ import annotations

import json
from pathlib import Path


def _write(path: Path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_failure_anchor_mix_not_confounded_by_positive_guards(tmp_path):
    from jit_dvgc.upstream_boundary_analysis import analyze_boundary_pilot

    entries = []
    labels = []
    outcomes = [
        ("failure_anchor", "ascending_entry", "hip", 1, 0.025, 1, True, "apex_success"),
        ("failure_anchor", "ascending_entry", "hip", 1, 0.05, 1, False, "pitch_limit"),
        ("positive_guard", "jump_zone_entry", "hip", 1, 0.025, 1, True, "apex_success"),
        ("positive_guard", "height_entry", "hip", 1, 0.025, 1, True, "apex_success"),
    ]
    for index, (kind, role, action, sign, strength, duration, success, reason) in enumerate(outcomes):
        state = f"state-{index}"
        entries.append({
            "source_bank": "boundary_bank", "parent_group_id": "transition_1__1000001", "seed": 1000001,
            "role": role, "tick": 10 + index, "state_sha256": state, "protocol_sha256": "protocol",
            "anchor_kind": kind, "anchor_parent_group_id": "transition_1__1000001", "anchor_state_sha256": "anchor",
            "anchor_role": role,
            "perturbation": {"action_name": action, "sign": sign, "strength": strength, "duration": duration},
        })
        labels.append({
            "boundary_candidate_index": index, "source_bank": "boundary_bank", "parent_group_id": "transition_1__1000001",
            "seed": 1000001, "state_sha256": state, "success_count": int(success), "boundary_protocol_sha256": "protocol",
            "branches": [{"reason": reason}],
        })
    catalog = tmp_path / "catalog.json"
    label_path = tmp_path / "labels.json"
    _write(catalog, {"entries": entries})
    _write(label_path, labels)

    report = analyze_boundary_pilot(catalog, label_path)
    assert report["aggregate"]["success_rate"] == 0.75
    assert report["failure_anchor"]["success_rate"] == 0.5
    assert report["failure_anchor"]["mixed_outcomes"]
    assert report["decision"] == "BOUNDARY_FOUND"
    assert report["mixed_failure_direction_count"] == 1


def test_no_failure_anchor_crossing_requests_refinement(tmp_path):
    from jit_dvgc.upstream_boundary_analysis import analyze_boundary_pilot

    catalog = tmp_path / "catalog.json"
    labels = tmp_path / "labels.json"
    _write(catalog, {"entries": [{
        "source_bank": "boundary_bank", "parent_group_id": "p", "seed": 1000001, "role": "ascending_entry", "tick": 1,
        "state_sha256": "s", "protocol_sha256": "p1", "anchor_kind": "failure_anchor", "anchor_parent_group_id": "p",
        "anchor_state_sha256": "a", "anchor_role": "ascending_entry",
        "perturbation": {"action_name": "knee", "sign": -1, "strength": 0.05, "duration": 1},
    }]})
    _write(labels, [{
        "boundary_candidate_index": 0, "source_bank": "boundary_bank", "parent_group_id": "p", "seed": 1000001,
        "state_sha256": "s", "success_count": 0, "boundary_protocol_sha256": "p1", "branches": [{"reason": "pitch_limit"}],
    }])
    report = analyze_boundary_pilot(catalog, labels)
    assert report["decision"] == "REFINE_FAILURE_ANCHOR_ONLY"
    assert not report["failure_anchor"]["mixed_outcomes"]


def test_pseudo_diversity_sensitivity_distinguishes_strict_and_practical(tmp_path):
    from jit_dvgc.upstream_boundary_analysis import analyze_boundary_pilot

    catalog = tmp_path / "catalog.json"
    labels = tmp_path / "labels.json"
    audit = tmp_path / "audit.json"
    _write(catalog, {"entries": [{
        "source_bank": "boundary_bank", "parent_group_id": "p", "seed": 1000001, "role": "ascending_entry", "tick": 1,
        "state_sha256": "s", "protocol_sha256": "p1", "anchor_kind": "failure_anchor", "anchor_parent_group_id": "p",
        "anchor_state_sha256": "a", "anchor_role": "ascending_entry",
        "perturbation": {"action_name": "hip", "sign": 1, "strength": 0.05, "duration": 1},
    }]})
    _write(labels, [{
        "boundary_candidate_index": 0, "source_bank": "boundary_bank", "parent_group_id": "p", "seed": 1000001,
        "state_sha256": "s", "success_count": 1, "boundary_protocol_sha256": "p1", "branches": [{"reason": "apex_success"}],
    }])
    _write(audit, {
        "near_duplicate_tolerances": {"qpos_atol": 1e-5, "qvel_atol": 1e-5, "observation_atol": 1e-5},
        "pairwise_distances": [{"exact": False, "near_duplicate": False, "qpos_linf": 3e-4, "qvel_linf": 1e-3, "observation_linf": 8e-3}],
    })
    report = analyze_boundary_pilot(catalog, labels, audit_path=audit)
    sensitivity = report["pseudo_diversity_audit"]
    assert sensitivity["strict_near_duplicate_pair_count"] == 0
    assert sensitivity["practical_near_duplicate_pair_count"] == 1
    assert sensitivity["all_pairs_practically_near_duplicate"]
