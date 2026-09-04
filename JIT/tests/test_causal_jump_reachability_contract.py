from __future__ import annotations

import importlib

import pytest

import jit_dvgc.acquisition.causal_jump as causal_jump


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
