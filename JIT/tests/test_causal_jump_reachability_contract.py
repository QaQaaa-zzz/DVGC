from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

import jit_dvgc.acquisition.causal_jump as causal_jump
from jit_dvgc.analysis.causal_jump_capability import (
    _canonical_sha256,
    _file_sha256,
    _load_logical_role_sources,
)


validate_jump_start_reachability_payload = getattr(
    causal_jump, "validate_jump_start_reachability_payload", None
)


def _payload() -> dict:
    return {
        "schema": "jit_jump_start_reachability_provenance_v1",
        "jump_start_connected": True,
        "natural_start_connected": False,
        "jump_start_state_sha256": "a" * 64,
        "generated_by_env_step_only": True,
        "rsi_used_to_establish_reachability": False,
        "qpos_qvel_injection_used": False,
        "proposal_anchor_used_as_reset": False,
        "proposal_anchor_state_sha256": "b" * 64,
        "proposal_parent_group_id": "g",
        "target_x_m": 3.2,
        "lookback_m": 0.3,
        "perturbation_start_target_x_m": 2.9,
        "perturbation_start_actual_x_m": 2.91,
        "perturbation_start_state_sha256": "c" * 64,
        "environment_transitions_before_perturbation": 60,
        "perturbed_environment_transitions": 8,
        "environment_transitions_from_jump_start": 68,
        "proposal_family_index": 0,
        "variant_ordinal": 0,
    }


def test_jump_start_reachability_requires_forward_env_step_ancestry() -> None:
    assert callable(validate_jump_start_reachability_payload), (
        "causal acquisition must expose jump-start reachability validation"
    )
    validate_jump_start_reachability_payload(_payload())


def test_jump_start_reachability_rejects_rsi_as_reachability_proof() -> None:
    payload = _payload()
    payload["rsi_used_to_establish_reachability"] = True
    with pytest.raises(ValueError, match="RSI"):
        validate_jump_start_reachability_payload(payload)


def test_jump_start_reachability_rejects_candidate_qpos_injection() -> None:
    payload = _payload()
    payload["qpos_qvel_injection_used"] = True
    with pytest.raises(ValueError, match="qpos/qvel"):
        validate_jump_start_reachability_payload(payload)


def test_jump_start_reachability_rejects_proposal_anchor_reset() -> None:
    payload = _payload()
    payload["proposal_anchor_used_as_reset"] = True
    with pytest.raises(ValueError, match="proposal anchor"):
        validate_jump_start_reachability_payload(payload)


def test_jump_start_reachability_contract_is_used_by_all_prospective_consumers() -> None:
    assert causal_jump.ACQUISITION_MODE == "jump_start_connected_causal_rollout_v1"
    for module_name in (
        "jit_dvgc.causal_frontier_protocol",
        "jit_dvgc.analysis.causal_jump_capability",
        "jit_dvgc.iterative_tube",
    ):
        module = importlib.import_module(module_name)
        assert (
            module.validate_jump_start_reachability_payload
            is validate_jump_start_reachability_payload
        )


def test_capability_analysis_resolves_derived_role_to_original_catalog(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / "raw" / "acquisition" / "catalog.json"
    catalog_path.parent.mkdir(parents=True)
    catalog = {
        "schema": "jit_unified_boundary_catalog_v1",
        "status": "completed",
        "entries": [
            {"candidate_id": "excluded", "state_sha256": "a" * 64},
            {"candidate_id": "kept", "state_sha256": "b" * 64},
        ],
    }
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    role_root = tmp_path / "derived" / "calibration"
    role_root.mkdir(parents=True)
    logical = {
        "role": "calibration",
        "entries": [
            {
                "candidate_id": "kept",
                "state_sha256": "b" * 64,
                "logical_role": "calibration",
                "split": "calibration",
                "label": 1,
            }
        ],
    }
    logical["labels_sha256"] = _canonical_sha256(logical)
    logical_path = role_root / "logical_labels.json"
    logical_path.write_text(json.dumps(logical), encoding="utf-8")
    manifest = {
        "status": "completed",
        "role": "calibration",
        "source_acquisition_catalog": str(catalog_path),
        "source_acquisition_catalog_sha256": _file_sha256(catalog_path),
        "logical_labels_file_sha256": _file_sha256(logical_path),
    }
    manifest["role_manifest_sha256"] = _canonical_sha256(manifest)
    (role_root / "role_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    resolved_path, pairs = _load_logical_role_sources(
        role_root, "calibration"
    )

    assert resolved_path == catalog_path
    assert [(row["candidate_id"], label["candidate_id"]) for row, label in pairs] == [
        ("kept", "kept")
    ]
