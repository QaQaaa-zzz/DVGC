from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

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


def _lockable_acceptance_row(index: int, phase: str, label: int, parent: str):
    return {
        **_acceptance_row(index, phase, label, parent),
        "policy_actor_sha256": "a" * 64,
        "policy_payload_sha256": "b" * 64,
        "label_protocol_sha256": "c" * 64,
        "acquisition_protocol_sha256": "d" * 64,
    }


def _write_acceptance_lock_inputs(tmp_path: Path, rows):
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(rows), encoding="utf-8")
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps({"status": "completed", "protocol_sha256": "e" * 64}),
        encoding="utf-8",
    )
    return labels_path, catalog_path


def _lock_acceptance_bank(labels_path: Path, catalog_path: Path):
    from jit_dvgc.continuation import lock_negative_acceptance_bank

    return lock_negative_acceptance_bank(
        labels_path,
        catalog_path,
        labels_path.parent / "acceptance_bank.json",
        target_tube_path=labels_path.parent / "target_tube.json",
        expected_target_tube_manifest_sha256="f" * 64,
        predeclaration_file_sha256="1" * 64,
        predeclared_protocol_sha256="2" * 64,
        minimum_negative_states_per_phase=2,
        minimum_negative_parent_groups_per_phase=2,
    )


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


def test_negative_acceptance_lock_persists_phasewise_readiness_before_repair_training(
    tmp_path, monkeypatch
):
    import jit_dvgc.continuation as continuation

    rows = [
        _lockable_acceptance_row(1, "upstream", 0, "u0"),
        _lockable_acceptance_row(2, "upstream", 0, "u1"),
        _lockable_acceptance_row(3, "downstream", 0, "d0"),
        _lockable_acceptance_row(4, "downstream", 1, "d1"),
    ]
    labels_path, catalog_path = _write_acceptance_lock_inputs(tmp_path, rows)
    original_labels = labels_path.read_bytes()
    monkeypatch.setattr(
        continuation,
        "load_soft_tube",
        lambda _path: SimpleNamespace(
            entries=(
                {"state_sha256": rows[0]["state_sha256"], "split": "train"},
            ),
            manifest={"manifest_sha256": "f" * 64},
        )
    )

    with pytest.raises(continuation.AcceptanceBankReadinessError) as error:
        _lock_acceptance_bank(labels_path, catalog_path)

    assert labels_path.read_bytes() == original_labels
    readiness_path = tmp_path / "acceptance_readiness.json"
    assert readiness_path.is_file()
    assert not (tmp_path / "acceptance_bank.json").exists()

    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    assert readiness["schema"] == "jit_repair_acceptance_readiness_v1"
    assert readiness["status"] == "not_ready_before_repair_training"
    assert readiness["selection"] == "all_baseline_continuation_negative_candidates"
    assert readiness["input_label_count"] == 4
    assert readiness["input_negative_count"] == 3
    assert readiness["excluded_target_state_count"] == 1
    assert readiness["excluded_target_negative_count"] == 1
    assert readiness["locked_negative_count"] == 2
    assert readiness["readiness"] == {
        "upstream": {
            "negative_state_count": 1,
            "negative_parent_group_count": 1,
            "minimum_negative_states": 2,
            "minimum_negative_parent_groups": 2,
            "ready": False,
        },
        "downstream": {
            "negative_state_count": 1,
            "negative_parent_group_count": 1,
            "minimum_negative_states": 2,
            "minimum_negative_parent_groups": 2,
            "ready": False,
        },
    }
    assert readiness["training_transitions"] == 0
    assert readiness["validation_data_used"] is False
    assert readiness["test_data_used"] is False
    assert readiness["final_evaluation_data_used"] is False
    assert error.value.audit["readiness"] == readiness["readiness"]


def test_negative_acceptance_lock_writes_locked_bank_when_both_phases_are_ready(
    tmp_path, monkeypatch
):
    import jit_dvgc.continuation as continuation

    rows = [
        _lockable_acceptance_row(1, "upstream", 0, "u0"),
        _lockable_acceptance_row(2, "upstream", 0, "u1"),
        _lockable_acceptance_row(3, "upstream", 0, "u2"),
        _lockable_acceptance_row(4, "downstream", 0, "d0"),
        _lockable_acceptance_row(5, "downstream", 0, "d1"),
        _lockable_acceptance_row(6, "downstream", 0, "d2"),
    ]
    labels_path, catalog_path = _write_acceptance_lock_inputs(tmp_path, rows)
    monkeypatch.setattr(
        continuation,
        "load_soft_tube",
        lambda _path: SimpleNamespace(
            entries=(
                {"state_sha256": rows[0]["state_sha256"], "split": "train"},
            ),
            manifest={"manifest_sha256": "f" * 64},
        ),
    )

    bank = _lock_acceptance_bank(labels_path, catalog_path)

    bank_path = tmp_path / "acceptance_bank.json"
    assert bank_path.is_file()
    assert not (tmp_path / "acceptance_readiness.json").exists()
    persisted_bank = json.loads(bank_path.read_text(encoding="utf-8"))
    assert persisted_bank == bank
    assert persisted_bank["status"] == "locked_before_repair_training"
    assert persisted_bank["target_tube_state_count"] == 1
    assert persisted_bank["entry_count"] == 5
    assert [row["candidate_id"] for row in persisted_bank["entries"]] == [
        "c2",
        "c3",
        "c4",
        "c5",
        "c6",
    ]
    assert persisted_bank["selection_audit"]["input_negative_count"] == 6
    assert persisted_bank["selection_audit"]["excluded_target_state_count"] == 1
    assert persisted_bank["selection_audit"]["excluded_target_negative_count"] == 1
    assert persisted_bank["selection_audit"]["locked_negative_count"] == 5
    assert persisted_bank["selection_audit"]["readiness"] == {
        "upstream": {
            "negative_state_count": 2,
            "negative_parent_group_count": 2,
            "minimum_negative_states": 2,
            "minimum_negative_parent_groups": 2,
            "ready": True,
        },
        "downstream": {
            "negative_state_count": 3,
            "negative_parent_group_count": 3,
            "minimum_negative_states": 2,
            "minimum_negative_parent_groups": 2,
            "ready": True,
        },
    }


def test_negative_acceptance_lock_preserves_existing_readiness_evidence(
    tmp_path, monkeypatch
):
    import jit_dvgc.continuation as continuation

    insufficient_rows = [
        _lockable_acceptance_row(1, "upstream", 0, "u0"),
        _lockable_acceptance_row(2, "upstream", 0, "u1"),
        _lockable_acceptance_row(3, "downstream", 0, "d0"),
    ]
    labels_path, catalog_path = _write_acceptance_lock_inputs(tmp_path, insufficient_rows)
    monkeypatch.setattr(
        continuation,
        "load_soft_tube",
        lambda _path: SimpleNamespace(
            entries=(
                {
                    "state_sha256": insufficient_rows[0]["state_sha256"],
                    "split": "train",
                },
            ),
            manifest={"manifest_sha256": "f" * 64},
        ),
    )

    with pytest.raises(continuation.AcceptanceBankReadinessError):
        _lock_acceptance_bank(labels_path, catalog_path)

    readiness_path = tmp_path / "acceptance_readiness.json"
    original_readiness = readiness_path.read_bytes()
    ready_rows = [
        _lockable_acceptance_row(1, "upstream", 0, "u0"),
        _lockable_acceptance_row(2, "upstream", 0, "u1"),
        _lockable_acceptance_row(3, "upstream", 0, "u2"),
        _lockable_acceptance_row(4, "downstream", 0, "d0"),
        _lockable_acceptance_row(5, "downstream", 0, "d1"),
        _lockable_acceptance_row(6, "downstream", 0, "d2"),
    ]
    labels_path.write_text(json.dumps(ready_rows), encoding="utf-8")

    with pytest.raises(FileExistsError, match="acceptance readiness already exists"):
        _lock_acceptance_bank(labels_path, catalog_path)

    assert readiness_path.read_bytes() == original_readiness
    assert not (tmp_path / "acceptance_bank.json").exists()
