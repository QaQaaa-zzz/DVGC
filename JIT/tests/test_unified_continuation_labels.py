from __future__ import annotations

import inspect

import pytest

from jit_dvgc.unified_continuation_labels import (
    classify_first_valid_landing_outcome,
    classify_unified_continuation_outcome,
    label_unified_continuations,
    validate_unified_boundary_catalog,
)


def _record():
    return {
        "name": "pi_0",
        "iteration": 0,
        "policy_role": "envelope_expansion_authority",
        "actor_sha256": "1" * 64,
        "payload_sha256": "2" * 64,
        "formal_config_sha256": "3" * 64,
        "xml_sha256": "4" * 64,
    }


def _row(candidate_id: str, state_sha: str, phase: str, phase_index: int):
    return {
        "candidate_id": candidate_id,
        "candidate_kind": "reachable_unified_frontier_probe",
        "split": "train",
        "phase": phase,
        "phase_index": phase_index,
        "snapshot": f"snapshots/{candidate_id}",
        "source_bank": "boundary_bank",
        "state_sha256": state_sha,
        "parent_group_id": f"group_{candidate_id}",
        "parent_state_sha256": "5" * 64,
        "policy_iteration": 0,
        "policy_actor_sha256": "1" * 64,
        "policy_payload_sha256": "2" * 64,
        "protocol_sha256": "6" * 64,
    }


def _catalog(rows):
    return {
        "schema": "jit_unified_boundary_catalog_v1",
        "status": "completed",
        "artifact_role": "unlabeled_policy_conditioned_frontier_candidates",
        "split": "train",
        "iteration": 0,
        "policy_name": "pi_0",
        "policy_actor_sha256": "1" * 64,
        "policy_payload_sha256": "2" * 64,
        "frozen_unified_manifest_sha256": "7" * 64,
        "protocol_sha256": "6" * 64,
        "candidate_count": len(rows),
        "training_transitions": 0,
        "expert_switching_used": False,
        "test_data_used": False,
        "validation_data_used": False,
        "final_evaluation_data_used": False,
        "claim_boundary": {
            "unlabeled_acquisition_only": True,
            "tube_expansion_claim": False,
            "jce_jel_claim": False,
            "certified_safe_set_claim": False,
        },
        "entries": rows,
    }


def test_unified_catalog_requires_unique_train_policy_bound_candidates():
    rows = [
        _row("u0", "a" * 64, "upstream", 0),
        _row("d0", "b" * 64, "downstream", 1),
    ]
    validated = validate_unified_boundary_catalog(
        _catalog(rows),
        policy_record=_record(),
        frozen_manifest_sha256="7" * 64,
    )
    assert tuple(validated) == tuple(rows)

    duplicate = [rows[0], {**rows[1], "state_sha256": "a" * 64}]
    with pytest.raises(ValueError, match="duplicate unified boundary physical state"):
        validate_unified_boundary_catalog(
            _catalog(duplicate),
            policy_record=_record(),
            frozen_manifest_sha256="7" * 64,
        )


def test_unified_catalog_rejects_validation_or_claim_drift():
    rows = [_row("u0", "a" * 64, "upstream", 0)]
    payload = _catalog(rows)
    payload["validation_data_used"] = True
    with pytest.raises(ValueError, match="validation"):
        validate_unified_boundary_catalog(
            payload,
            policy_record=_record(),
            frozen_manifest_sha256="7" * 64,
        )

    payload = _catalog(rows)
    payload["claim_boundary"] = {**payload["claim_boundary"], "tube_expansion_claim": True}
    with pytest.raises(ValueError, match="claim boundary"):
        validate_unified_boundary_catalog(
            payload,
            policy_record=_record(),
            frozen_manifest_sha256="7" * 64,
        )


def test_upstream_positive_requires_full_apex_to_recovery_chain():
    positive = classify_unified_continuation_outcome(
        start_phase=0,
        terminal_success=True,
        physical_failure=False,
        timeout=False,
        done=True,
        apex_seen=True,
        phase_transitioned=True,
        recovery_success=True,
        reached_rollout_horizon=False,
    )
    assert positive == (True, "success")

    no_transition = classify_unified_continuation_outcome(
        start_phase=0,
        terminal_success=True,
        physical_failure=False,
        timeout=False,
        done=True,
        apex_seen=True,
        phase_transitioned=False,
        recovery_success=True,
        reached_rollout_horizon=False,
    )
    assert no_transition == (False, "task_failure")

    alive_only = classify_unified_continuation_outcome(
        start_phase=0,
        terminal_success=False,
        physical_failure=False,
        timeout=False,
        done=False,
        apex_seen=True,
        phase_transitioned=True,
        recovery_success=False,
        reached_rollout_horizon=True,
    )
    assert alive_only == (False, "horizon_exhausted")


def test_downstream_positive_requires_recovery_success():
    assert classify_unified_continuation_outcome(
        start_phase=1,
        terminal_success=True,
        physical_failure=False,
        timeout=False,
        done=True,
        apex_seen=False,
        phase_transitioned=False,
        recovery_success=True,
        reached_rollout_horizon=False,
    ) == (True, "success")

    assert classify_unified_continuation_outcome(
        start_phase=1,
        terminal_success=False,
        physical_failure=True,
        timeout=False,
        done=True,
        apex_seen=False,
        phase_transitioned=False,
        recovery_success=False,
        reached_rollout_horizon=False,
    ) == (False, "physical_failure")


def test_first_valid_landing_is_positive_without_post_landing_recovery():
    assert classify_first_valid_landing_outcome(
        valid_contact_seen=True,
        physical_failure_before_landing=False,
        timeout=False,
        done=False,
        reached_rollout_horizon=False,
    ) == (True, "first_valid_landing")


def test_airborne_physical_failure_is_negative_before_landing():
    assert classify_first_valid_landing_outcome(
        valid_contact_seen=False,
        physical_failure_before_landing=True,
        timeout=False,
        done=True,
        reached_rollout_horizon=False,
    ) == (False, "airborne_physical_failure")


def test_labeling_accepts_one_search_lifetime_compiled_step():
    parameters = inspect.signature(label_unified_continuations).parameters

    assert "compiled_step_fn" in parameters


def test_labeling_separates_acquisition_policy_from_landing_evaluator():
    parameters = inspect.signature(label_unified_continuations).parameters

    assert "acquisition_policy_record" in parameters
    assert "acquisition_frozen_manifest_sha256" in parameters
    assert "success_criterion" in parameters
