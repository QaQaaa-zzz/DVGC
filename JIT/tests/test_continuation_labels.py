from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def _load_label_unified_continuations():
    cli_path = Path(__file__).parents[1] / "cli" / "label_unified_continuations.py"
    spec = importlib.util.spec_from_file_location(
        "label_unified_continuations_test", cli_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _repair_predeclaration_files(tmp_path: Path):
    module = _load_label_unified_continuations()
    protocol = {"contract": "fresh-repair", "version": 1}
    payload = {
        "expected_protocol_sha256": module._canonical_sha256(protocol),
        "protocol": protocol,
    }
    source_path = tmp_path / "source-predeclaration.json"
    copied_path = tmp_path / "predeclaration.json"
    source_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    copied_path.write_text(
        json.dumps(payload, separators=(",", ":"), sort_keys=False), encoding="utf-8"
    )
    audit_path = tmp_path / "anchor_audit.json"
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    audit_path.write_text(
        json.dumps({"predeclaration_file_sha256": source_sha}), encoding="utf-8"
    )
    return module, payload, source_sha, copied_path, audit_path


def test_repair_predeclaration_binding_accepts_semantically_equal_canonical_copy(
    tmp_path,
):
    module, payload, source_sha, copied_path, audit_path = _repair_predeclaration_files(
        tmp_path
    )

    assert copied_path.read_bytes() != (tmp_path / "source-predeclaration.json").read_bytes()
    assert module._validate_repair_predeclaration_binding(
        payload, source_sha, copied_path, audit_path
    ) == {"predeclaration_file_sha256": source_sha}


def test_repair_predeclaration_binding_rejects_semantic_copy_drift(tmp_path):
    module, payload, source_sha, copied_path, audit_path = _repair_predeclaration_files(
        tmp_path
    )
    changed = dict(payload)
    changed["semantic_field"] = "changed"
    copied_path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ValueError, match="semantic drift"):
        module._validate_repair_predeclaration_binding(
            payload, source_sha, copied_path, audit_path
        )


def test_repair_predeclaration_binding_rejects_incorrect_audit_source_sha(tmp_path):
    module, payload, source_sha, copied_path, audit_path = _repair_predeclaration_files(
        tmp_path
    )
    audit_path.write_text(
        json.dumps({"predeclaration_file_sha256": "0" * 64}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="audit/source predeclaration SHA-256 drift"):
        module._validate_repair_predeclaration_binding(
            payload, source_sha, copied_path, audit_path
        )


def test_repair_predeclaration_binding_rejects_copied_protocol_hash_drift(tmp_path):
    module, payload, source_sha, copied_path, audit_path = _repair_predeclaration_files(
        tmp_path
    )
    changed = dict(payload, expected_protocol_sha256="f" * 64)
    source_path = tmp_path / "source-predeclaration.json"
    source_path.write_text(
        json.dumps(changed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    copied_path.write_text(
        json.dumps(changed, separators=(",", ":"), sort_keys=False), encoding="utf-8"
    )
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    audit_path.write_text(
        json.dumps({"predeclaration_file_sha256": source_sha}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="canonical protocol SHA-256 drift"):
        module._validate_repair_predeclaration_binding(
            changed, source_sha, copied_path, audit_path
        )


def test_global_seed_splits_are_disjoint_complete_and_deterministic():
    from jit_dvgc.continuation_labels import assign_global_seed_splits

    observed = [1000001, 1000002, 1000003, 1000004, 1000005, 1000006, 1000007, 1000008]
    first = assign_global_seed_splits(observed)
    second = assign_global_seed_splits(reversed(observed))
    assert first == second
    assert first == {
        1000001: "train",
        1000002: "train",
        1000003: "train",
        1000004: "train",
        1000005: "train",
        1000006: "validation",
        1000007: "test",
        1000008: "test",
    }


def test_global_seed_split_keeps_same_seed_across_source_checkpoints():
    from jit_dvgc.continuation_labels import assign_global_seed_splits

    source_checkpoints = ("transition_4988928", "transition_7987200", "transition_9977856")
    seed_splits = assign_global_seed_splits(range(1000001, 1000009))
    parent_splits = {
        f"{checkpoint}__{seed}": seed_splits[seed]
        for checkpoint in source_checkpoints
        for seed in range(1000001, 1000009)
    }
    for seed in range(1000001, 1000009):
        assert {
            parent_splits[f"{checkpoint}__{seed}"] for checkpoint in source_checkpoints
        } == {seed_splits[seed]}


def test_global_seed_splits_reject_overlap_and_incomplete_coverage():
    from jit_dvgc.continuation_labels import assign_global_seed_splits

    with pytest.raises(ValueError, match="must be disjoint"):
        assign_global_seed_splits(
            [1, 2, 3],
            train_seeds=(1,),
            validation_seeds=(2,),
            test_seeds=(2, 3),
        )
    with pytest.raises(ValueError, match="exactly cover observed"):
        assign_global_seed_splits(
            [1, 2, 3],
            train_seeds=(1,),
            validation_seeds=(2,),
            test_seeds=(4,),
        )


def test_branch_key_is_stable_and_branch_specific():
    from jit_dvgc.continuation_labels import derive_branch_key

    seed0, _ = derive_branch_key(protocol_seed=7, candidate_id="abc", branch_index=0)
    seed0_again, _ = derive_branch_key(protocol_seed=7, candidate_id="abc", branch_index=0)
    seed1, _ = derive_branch_key(protocol_seed=7, candidate_id="abc", branch_index=1)
    assert seed0 == seed0_again
    assert seed0 != seed1


def _acceptance_row(index: int, phase: str, label: int, parent: str):
    return {
        "candidate_id": f"c{index}",
        "split": "train",
        "phase": phase,
        "state_sha256": f"{index:064x}",
        "parent_group_id": parent,
        "label": label,
    }


def test_negative_acceptance_selection_keeps_all_negatives_outside_target_support():
    from jit_dvgc.continuation import select_negative_acceptance_rows

    rows = [
        _acceptance_row(1, "upstream", 0, "u0"),
        _acceptance_row(2, "upstream", 0, "u1"),
        _acceptance_row(3, "upstream", 0, "u2"),
        _acceptance_row(4, "upstream", 1, "u3"),
        _acceptance_row(5, "downstream", 0, "d0"),
        _acceptance_row(6, "downstream", 0, "d1"),
        _acceptance_row(7, "downstream", 0, "d2"),
        _acceptance_row(8, "downstream", 1, "d3"),
    ]
    selected, audit = select_negative_acceptance_rows(
        rows,
        excluded_state_sha256=(f"{3:064x}", f"{7:064x}"),
        minimum_negative_states_per_phase=2,
        minimum_negative_parent_groups_per_phase=2,
    )

    assert [row["candidate_id"] for row in selected] == ["c1", "c2", "c5", "c6"]
    assert audit["selection"] == "all_baseline_continuation_negative_candidates"
    assert audit["input_negative_count"] == 6
    assert audit["excluded_target_negative_count"] == 2
    assert audit["locked_negative_count"] == 4
    assert audit["readiness"]["upstream"]["ready"] is True
    assert audit["readiness"]["downstream"]["ready"] is True


def test_negative_acceptance_selection_stops_before_training_when_phase_not_ready():
    from jit_dvgc.continuation import select_negative_acceptance_rows

    rows = [
        _acceptance_row(1, "upstream", 0, "u0"),
        _acceptance_row(2, "upstream", 0, "u1"),
        _acceptance_row(3, "downstream", 0, "d0"),
        _acceptance_row(4, "downstream", 1, "d1"),
    ]
    with pytest.raises(ValueError, match="not phasewise ready before repair training"):
        select_negative_acceptance_rows(
            rows,
            minimum_negative_states_per_phase=2,
            minimum_negative_parent_groups_per_phase=2,
        )
