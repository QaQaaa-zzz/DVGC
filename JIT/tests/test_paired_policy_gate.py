from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from jit_dvgc.analysis import paired_policy_gate as gate


def _file_sha256(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _v2_config(tmp_path, *, source_iteration=2, candidate_iteration=3):
    sha_a = "a" * 64
    sha_b = "b" * 64
    sha_c = "c" * 64
    sha_d = "d" * 64
    sha_e = "e" * 64
    protocol = {
        "schema": gate.PROTOCOL_SCHEMA_V2,
        "status": "predeclared_before_gate_execution",
        "source_iteration": source_iteration,
        "candidate_iteration": candidate_iteration,
        "policies": {
            "baseline": {
                "iteration": source_iteration,
                "name": f"pi_{source_iteration}",
                "frozen_manifest": "baseline.json",
                "actor_sha256": sha_a,
                "payload_sha256": sha_b,
            },
            "candidate": {
                "iteration": candidate_iteration,
                "name": f"pi_{candidate_iteration}",
                "frozen_manifest": "candidate.json",
                "actor_sha256": sha_c,
                "payload_sha256": sha_d,
            },
        },
        "core": {
            "source_tube": "tube_k",
            "source_tube_manifest_sha256": sha_e,
            "selection": "all_source_tube_entries",
            "preservation_rule": "zero_baseline_success_to_candidate_failure",
            "require_baseline_success_each_phase": True,
        },
        "boundary": {
            "selection": gate.LOCKED_BOUNDARY_SELECTION,
            "locked_bank": "acceptance_bank.json",
            "locked_bank_sha256": sha_a,
            "target_tube": "tube_kplus1",
            "target_tube_manifest_sha256": sha_b,
            "source_catalog_root": "acquisition_root",
            "source_catalog_file_sha256": sha_c,
            "source_catalog_protocol_sha256": sha_d,
            "snapshot_provenance": gate.LOCKED_SNAPSHOT_PROVENANCE,
            "require_baseline_negative_reproduction": True,
            "minimum_candidate_success_parent_groups": 2,
        },
        "runtime": {
            "policy_mode": "deterministic",
            "max_ticks": 400,
            "protocol_seed": 12345,
        },
        "data_policy": {
            "split": "train_audit_only",
            "validation_data_used": False,
            "test_data_used": False,
            "final_evaluation_data_used": False,
            "training_transitions": 0,
            "expert_switching_used": False,
        },
        "claim_boundary": {
            "iteration_selection_gate_only": True,
            "empirical_envelope_expansion_claim_requires_both_gates": True,
            "jce_jel_claim": False,
            "certified_safe_set_claim": False,
        },
    }
    config = {
        "schema": gate.CONFIG_SCHEMA_V2,
        "output_dir": "gate_output",
        "expected_protocol_sha256": gate.canonical_sha256(protocol),
        "protocol": protocol,
    }
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_historical_v1_gate_config_remains_loadable(jit_root):
    path = jit_root / "configs/envelope_iter1_paired_policy_gate_retry01.json"
    config = gate.load_paired_policy_gate_config(path)
    assert config["schema"] == gate.CONFIG_SCHEMA
    assert config["protocol"]["source_iteration"] == 0
    assert config["protocol"]["candidate_iteration"] == 1


def test_locked_bank_gate_config_is_generic_k_to_kplus1(tmp_path):
    path = _v2_config(tmp_path, source_iteration=2, candidate_iteration=3)
    config = gate.load_paired_policy_gate_config(path)
    assert config["schema"] == gate.CONFIG_SCHEMA_V2
    assert config["protocol"]["source_iteration"] == 2
    assert config["protocol"]["candidate_iteration"] == 3
    assert (
        config["protocol"]["boundary"]["snapshot_provenance"]
        == gate.LOCKED_SNAPSHOT_PROVENANCE
    )


def test_locked_bank_gate_rejects_non_adjacent_iterations(tmp_path):
    path = _v2_config(tmp_path, source_iteration=2, candidate_iteration=4)
    with pytest.raises(ValueError, match=r"k -> k\+1"):
        gate.load_paired_policy_gate_config(path)


def test_locked_bank_uses_original_catalog_root_for_snapshot_provenance(tmp_path):
    actor = "1" * 64
    payload = "2" * 64
    target_manifest = "3" * 64
    acquisition_protocol = "4" * 64
    label_protocol = "5" * 64
    source_root = tmp_path / "acquisition"
    source_root.mkdir()
    catalog = source_root / "catalog.json"

    entries = [
        {
            "candidate_id": "up",
            "candidate_kind": "reachable_unified_frontier_probe",
            "split": "train",
            "phase": "upstream",
            "phase_index": 0,
            "snapshot": "snapshot_up",
            "source_bank": "bank_a",
            "state_sha256": "6" * 64,
            "parent_group_id": "up_parent",
            "parent_state_sha256": "7" * 64,
            "label": 0,
            "policy_iteration": 7,
            "policy_actor_sha256": actor,
            "policy_payload_sha256": payload,
            "acquisition_protocol_sha256": acquisition_protocol,
            "label_protocol_sha256": label_protocol,
        },
        {
            "candidate_id": "down",
            "candidate_kind": "reachable_unified_frontier_probe",
            "split": "train",
            "phase": "downstream",
            "phase_index": 1,
            "snapshot": "snapshot_down",
            "source_bank": "bank_b",
            "state_sha256": "8" * 64,
            "parent_group_id": "down_parent",
            "parent_state_sha256": "9" * 64,
            "label": 0,
            "policy_iteration": 7,
            "policy_actor_sha256": actor,
            "policy_payload_sha256": payload,
            "acquisition_protocol_sha256": acquisition_protocol,
            "label_protocol_sha256": label_protocol,
        },
    ]
    catalog.write_text(
        json.dumps(
            {
                "schema": "jit_unified_boundary_catalog_v1",
                "status": "completed",
                "protocol_sha256": acquisition_protocol,
                "entries": [
                    {
                        key: value
                        for key, value in row.items()
                        if key
                        not in {
                            "label",
                            "label_protocol_sha256",
                            "acquisition_protocol_sha256",
                        }
                    }
                    for row in entries
                ],
            }
        ),
        encoding="utf-8",
    )

    base = {
        "schema": gate.NEGATIVE_ACCEPTANCE_BANK_SCHEMA,
        "status": "locked_before_repair_training",
        "artifact_role": "fresh_nonfinal_baseline_negative_iteration_acceptance_bank",
        "split": "train_audit_only",
        "selection": "all_baseline_continuation_negative_candidates",
        "source_labels": str(tmp_path / "merged" / "labels.json"),
        "source_labels_file_sha256": "a" * 64,
        "source_catalog": str(catalog),
        "source_catalog_file_sha256": _file_sha256(catalog),
        "source_catalog_root": str(source_root),
        "source_catalog_protocol_sha256": acquisition_protocol,
        "baseline_policy_actor_sha256": actor,
        "baseline_policy_payload_sha256": payload,
        "label_protocol_sha256": label_protocol,
        "acquisition_protocol_sha256": acquisition_protocol,
        "predeclaration_file_sha256": "b" * 64,
        "predeclared_protocol_sha256": "c" * 64,
        "target_tube": "tube_8",
        "target_tube_manifest_sha256": target_manifest,
        "target_tube_state_count": 0,
        "entry_count": len(entries),
        "selection_audit": {
            "locked_negative_count": len(entries),
        },
        "entries": entries,
        "environment_interactions": 0,
        "training_transitions": 0,
        "expert_switching_used": False,
        "validation_data_used": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "claim_boundary": {
            "nonfinal_iteration_acceptance_audit_only": True,
            "candidate_policy_outcomes_inspected": False,
            "repair_training_data": False,
            "tube_construction_data": False,
            "continuation_field_fit_data": False,
            "jce_jel_claim": False,
            "certified_safe_set_claim": False,
        },
    }
    bank_sha = gate.canonical_sha256(base)
    bank_path = tmp_path / "merged" / "acceptance_bank.json"
    bank_path.parent.mkdir()
    bank_path.write_text(
        json.dumps({**base, "bank_sha256": bank_sha}),
        encoding="utf-8",
    )
    boundary = {
        "locked_bank": str(bank_path),
        "locked_bank_sha256": bank_sha,
        "target_tube": "tube_8",
        "target_tube_manifest_sha256": target_manifest,
        "source_catalog_root": str(source_root),
        "source_catalog_file_sha256": _file_sha256(catalog),
        "source_catalog_protocol_sha256": acquisition_protocol,
    }
    baseline_record = {
        "iteration": 7,
        "actor_sha256": actor,
        "payload_sha256": payload,
    }
    target_tube = SimpleNamespace(
        manifest={"manifest_sha256": target_manifest},
        entries=[],
    )

    bank, selected = gate._load_locked_negative_bank(
        boundary,
        baseline_record=baseline_record,
        target_tube=target_tube,
    )
    assert len(selected) == 2
    assert gate._locked_snapshot_path(bank, selected[0]).parent.parent == source_root
    assert "merged" not in str(gate._locked_snapshot_path(bank, selected[0]))
