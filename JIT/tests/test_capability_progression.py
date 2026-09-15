from __future__ import annotations

from jit_dvgc.analysis.capability_progression import analyze_capability_progression


def _gate(
    *,
    up_baseline: int,
    up_candidate: int,
    up_states: int,
    down_baseline: int,
    down_candidate: int,
    down_states: int,
    boundary_up: int = 1,
    boundary_down: int = 1,
    boundary_groups: int = 2,
) -> dict:
    baseline_total = up_baseline + down_baseline
    candidate_total = up_candidate + down_candidate
    return {
        "schema": "jit_paired_policy_gate_report_v1",
        "status": "completed",
        "source_iteration": 1,
        "candidate_iteration": 2,
        "baseline_policy_name": "pi_1",
        "candidate_policy_name": "pi_2",
        "candidate_actor_sha256": "a" * 64,
        "candidate_payload_sha256": "b" * 64,
        "core_source": {"baseline_success_criterion": "first_valid_landing", "candidate_success_criterion": "first_valid_landing"},
        "boundary_source": {"success_criterion": "first_valid_landing"},
        "core_gate": {
            "state_count": up_states + down_states,
            "baseline_success_count": baseline_total,
            "candidate_success_count": candidate_total,
            "regression_count": max(0, up_baseline - up_candidate)
            + max(0, down_baseline - down_candidate),
            "improvement_count": 0,
            "passed": False,
            "phase_counts": {
                "upstream": {
                    "state_count": up_states,
                    "baseline_success_count": up_baseline,
                    "candidate_success_count": up_candidate,
                    "regression_count": max(0, up_baseline - up_candidate),
                    "improvement_count": 0,
                },
                "downstream": {
                    "state_count": down_states,
                    "baseline_success_count": down_baseline,
                    "candidate_success_count": down_candidate,
                    "regression_count": max(0, down_baseline - down_candidate),
                    "improvement_count": 0,
                },
            },
        },
        "boundary_gate": {
            "state_count": boundary_up + boundary_down,
            "baseline_reproduction_failure_count": 0,
            "candidate_success_count": boundary_up + boundary_down,
            "candidate_success_parent_group_count": boundary_groups,
            "minimum_candidate_success_parent_groups": 2,
            "phase_counts": {
                "upstream": {
                    "state_count": max(boundary_up, 1),
                    "baseline_success_count": 0,
                    "candidate_success_count": boundary_up,
                    "regression_count": 0,
                    "improvement_count": boundary_up,
                },
                "downstream": {
                    "state_count": max(boundary_down, 1),
                    "baseline_success_count": 0,
                    "candidate_success_count": boundary_down,
                    "regression_count": 0,
                    "improvement_count": boundary_down,
                },
            },
        },
        "test_data_used": False,
        "final_evaluation_data_used": False,
    }


def test_small_paired_regressions_do_not_erase_envelope_or_block_authority() -> None:
    result = analyze_capability_progression(
        _gate(
            up_baseline=100,
            up_candidate=94,
            up_states=100,
            down_baseline=100,
            down_candidate=98,
            down_states=100,
        )
    )
    assert result["empirical_envelope_expansion_observed"] is True
    assert result["candidate_policy_authority_eligible"] is True
    assert result["strict_zero_regression_diagnostic_passed"] is False
    assert result["policy_realization"]["passed"] is True


def test_phase_collapse_blocks_policy_authority_even_when_global_drop_is_small() -> None:
    # Mirrors the scientific reason aggregate coverage alone is unsafe: a small
    # global loss can hide a severe upstream degradation when downstream support
    # is much larger.
    result = analyze_capability_progression(
        _gate(
            up_baseline=99,
            up_candidate=73,
            up_states=100,
            down_baseline=900,
            down_candidate=900,
            down_states=900,
        )
    )
    assert result["empirical_envelope_expansion_observed"] is True
    assert result["policy_realization"]["global"]["passed"] is True
    assert result["policy_realization"]["phase_retention_passed"]["upstream"] is False
    assert result["candidate_policy_authority_eligible"] is False
    assert result["decision"] == "envelope_progressed_but_candidate_policy_coverage_degraded"


def test_frontier_progression_requires_both_phases() -> None:
    result = analyze_capability_progression(
        _gate(
            up_baseline=100,
            up_candidate=100,
            up_states=100,
            down_baseline=100,
            down_candidate=100,
            down_states=100,
            boundary_up=2,
            boundary_down=0,
            boundary_groups=2,
        )
    )
    assert result["frontier_progression"]["candidate_success_each_phase"]["downstream"] is False
    assert result["empirical_envelope_expansion_observed"] is False
    assert result["candidate_policy_authority_eligible"] is False


def test_frontier_progression_does_not_require_success_in_empty_phase() -> None:
    gate = _gate(
        up_baseline=100,
        up_candidate=100,
        up_states=100,
        down_baseline=100,
        down_candidate=100,
        down_states=100,
        boundary_up=2,
        boundary_down=0,
        boundary_groups=2,
    )
    gate["boundary_gate"]["phase_counts"]["downstream"]["state_count"] = 0

    result = analyze_capability_progression(gate)

    assert result["frontier_progression"]["phase_required"] == {
        "upstream": True,
        "downstream": False,
    }
    assert result["empirical_envelope_expansion_observed"] is True
    assert result["candidate_policy_authority_eligible"] is True


def test_current_pi2_pattern_is_expansion_with_upstream_policy_coverage_collapse() -> None:
    result = analyze_capability_progression(
        _gate(
            up_baseline=423,
            up_candidate=312,
            up_states=427,
            down_baseline=2692,
            down_candidate=2690,
            down_states=2692,
            boundary_up=4,
            boundary_down=9,
            boundary_groups=3,
        ),
        retrospective=True,
    )
    assert result["empirical_envelope_expansion_observed"] is True
    assert result["candidate_policy_authority_eligible"] is False
    assert result["retrospective_analysis"] is True
    assert result["formal_prospective_selection_claim"] is False
    assert result["policy_realization"]["phase_retention_passed"]["upstream"] is False
    assert result["policy_realization"]["phase_retention_passed"]["downstream"] is True


def test_historical_mixed_endpoint_gate_cannot_become_authority(repository_root):
    import json
    import pytest
    path = repository_root / "JIT/runs/iteration_auto/pi_2_to_pi_3_pi0_centerline_family_landing_wide_20260904/pi3_landing_replay_acceptance_gate/summary.json"
    with pytest.raises(ValueError, match="mixes"):
        analyze_capability_progression(json.loads(path.read_text()))
