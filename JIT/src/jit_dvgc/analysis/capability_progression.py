"""Capability-progression analysis for iterative JIT policy gates.

The JIT envelope is cumulative empirical capability evidence for one fixed task
model.  A newly trained policy is both a capability probe and a candidate single
policy realization of that cumulative envelope.  These are related but distinct
questions:

1. did the candidate demonstrate new capability on the locked frontier; and
2. does the candidate still realize enough of the previously demonstrated Tube
   support to become the sole authority for the next automatic iteration?

This module intentionally does not equate one paired state regression with loss
of the cumulative empirical envelope.  Historical Tube evidence remains valid as
provenance even when a later stochastic/reward-guided policy does not reproduce
every earlier single rollout.

The current v1 decision uses fixed-panel coverage as an engineering proxy.  It is
not a per-state calibrated success probability and is not a physical feasibility
proof, viability certificate, or final JCE/JEL result.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import json

from jit_dvgc.config import file_sha256
from jit_dvgc.iterative_frontier_protocol import canonical_sha256


SCHEMA = "jit_capability_progression_decision_v1"
SUPPORTED_GATE_SCHEMA = "jit_paired_policy_gate_report_v1"

# Method-level non-inferiority margins for policy realization.  These are not
# tuned per candidate.  They deliberately allow small paired-rollout variation
# while preventing aggregate numbers from hiding a phase-specific collapse.
MAX_GLOBAL_CORE_COVERAGE_DROP = 0.05
MAX_PHASE_CORE_COVERAGE_DROP = 0.10
REQUIRE_BOUNDARY_SUCCESS_EACH_PHASE = True


def _rate(successes: int, total: int) -> float:
    if int(total) <= 0:
        raise ValueError("capability progression requires non-empty support")
    return float(successes) / float(total)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _phase_coverage(core: Mapping[str, Any], phase: str) -> dict[str, Any]:
    phase_counts = core.get("phase_counts")
    if not isinstance(phase_counts, Mapping) or phase not in phase_counts:
        raise ValueError(f"core gate missing {phase} phase counts")
    row = phase_counts[phase]
    if not isinstance(row, Mapping):
        raise ValueError(f"core gate {phase} phase counts invalid")
    state_count = int(row.get("state_count", 0))
    baseline_success = int(row.get("baseline_success_count", 0))
    candidate_success = int(row.get("candidate_success_count", 0))
    baseline_rate = _rate(baseline_success, state_count)
    candidate_rate = _rate(candidate_success, state_count)
    return {
        "state_count": state_count,
        "baseline_success_count": baseline_success,
        "candidate_success_count": candidate_success,
        "baseline_panel_coverage": baseline_rate,
        "candidate_panel_coverage": candidate_rate,
        "coverage_delta": candidate_rate - baseline_rate,
        "coverage_drop": max(0.0, baseline_rate - candidate_rate),
        "regression_count": int(row.get("regression_count", 0)),
        "improvement_count": int(row.get("improvement_count", 0)),
    }


def analyze_capability_progression(
    gate_summary: Mapping[str, Any],
    *,
    gate_summary_path: Path | None = None,
    retrospective: bool = False,
) -> dict[str, Any]:
    """Interpret one completed paired gate under the capability-progression v1 contract."""

    gate = dict(gate_summary)
    if gate.get("schema") != SUPPORTED_GATE_SCHEMA or gate.get("status") != "completed":
        raise ValueError("capability progression requires a completed paired policy gate")
    if gate.get("test_data_used") is not False or gate.get("final_evaluation_data_used") is not False:
        raise ValueError("capability progression cannot consume TEST/final evidence")

    core = gate.get("core_gate")
    boundary = gate.get("boundary_gate")
    if not isinstance(core, Mapping) or not isinstance(boundary, Mapping):
        raise ValueError("paired gate core/boundary sections missing")

    core_state_count = int(core.get("state_count", 0))
    baseline_core_success = int(core.get("baseline_success_count", 0))
    candidate_core_success = int(core.get("candidate_success_count", 0))
    baseline_global = _rate(baseline_core_success, core_state_count)
    candidate_global = _rate(candidate_core_success, core_state_count)
    global_drop = max(0.0, baseline_global - candidate_global)

    phases = {
        phase: _phase_coverage(core, phase)
        for phase in ("upstream", "downstream")
    }
    phase_retention = {
        phase: row["coverage_drop"] <= MAX_PHASE_CORE_COVERAGE_DROP
        for phase, row in phases.items()
    }
    policy_realization_retained = bool(
        global_drop <= MAX_GLOBAL_CORE_COVERAGE_DROP
        and all(phase_retention.values())
    )

    boundary_phase_counts = boundary.get("phase_counts")
    if not isinstance(boundary_phase_counts, Mapping):
        raise ValueError("boundary gate phase counts missing")
    boundary_success_each_phase = {
        phase: int(boundary_phase_counts.get(phase, {}).get("candidate_success_count", 0)) > 0
        for phase in ("upstream", "downstream")
    }
    boundary_success_count = int(boundary.get("candidate_success_count", 0))
    boundary_groups = int(boundary.get("candidate_success_parent_group_count", 0))
    minimum_groups = int(boundary.get("minimum_candidate_success_parent_groups", 1))
    reproduction_failures = int(boundary.get("baseline_reproduction_failure_count", 0))
    frontier_progression = bool(
        reproduction_failures == 0
        and boundary_success_count > 0
        and boundary_groups >= minimum_groups
        and (
            all(boundary_success_each_phase.values())
            if REQUIRE_BOUNDARY_SUCCESS_EACH_PHASE
            else True
        )
    )

    authority_eligible = bool(frontier_progression and policy_realization_retained)
    if authority_eligible:
        decision = "envelope_progressed_and_candidate_authority_eligible"
    elif frontier_progression:
        decision = "envelope_progressed_but_candidate_policy_coverage_degraded"
    else:
        decision = "no_accepted_frontier_progression"

    source_gate_path = str(gate_summary_path) if gate_summary_path is not None else None
    source_gate_file_sha256 = (
        file_sha256(gate_summary_path) if gate_summary_path is not None else None
    )
    result = {
        "schema": SCHEMA,
        "status": "completed",
        "decision": decision,
        "source_iteration": int(gate.get("source_iteration", -1)),
        "candidate_iteration": int(gate.get("candidate_iteration", -1)),
        "baseline_policy_name": str(gate.get("baseline_policy_name", "")),
        "candidate_policy_name": str(gate.get("candidate_policy_name", "")),
        "candidate_actor_sha256": str(gate.get("candidate_actor_sha256", "")),
        "candidate_payload_sha256": str(gate.get("candidate_payload_sha256", "")),
        "source_gate_summary": source_gate_path,
        "source_gate_file_sha256": source_gate_file_sha256,
        "retrospective_analysis": bool(retrospective),
        "formal_prospective_selection_claim": bool(authority_eligible and not retrospective),
        "empirical_envelope_expansion_observed": frontier_progression,
        "candidate_policy_authority_eligible": authority_eligible,
        "strict_zero_regression_diagnostic_passed": bool(
            int(core.get("regression_count", 0)) == 0 and core.get("passed") is True
        ),
        "policy_realization": {
            "metric": "fixed_locked_panel_success_coverage",
            "global": {
                "state_count": core_state_count,
                "baseline_success_count": baseline_core_success,
                "candidate_success_count": candidate_core_success,
                "baseline_panel_coverage": baseline_global,
                "candidate_panel_coverage": candidate_global,
                "coverage_delta": candidate_global - baseline_global,
                "coverage_drop": global_drop,
                "maximum_allowed_coverage_drop": MAX_GLOBAL_CORE_COVERAGE_DROP,
                "passed": global_drop <= MAX_GLOBAL_CORE_COVERAGE_DROP,
            },
            "phases": phases,
            "maximum_allowed_phase_coverage_drop": MAX_PHASE_CORE_COVERAGE_DROP,
            "phase_retention_passed": phase_retention,
            "passed": policy_realization_retained,
            "strict_regression_count": int(core.get("regression_count", 0)),
            "strict_improvement_count": int(core.get("improvement_count", 0)),
        },
        "frontier_progression": {
            "boundary_state_count": int(boundary.get("state_count", 0)),
            "candidate_success_count": boundary_success_count,
            "candidate_success_parent_group_count": boundary_groups,
            "minimum_candidate_success_parent_groups": minimum_groups,
            "baseline_reproduction_failure_count": reproduction_failures,
            "candidate_success_each_phase": boundary_success_each_phase,
            "require_candidate_success_each_phase": REQUIRE_BOUNDARY_SUCCESS_EACH_PHASE,
            "passed": frontier_progression,
        },
        "decision_contract": {
            "cumulative_envelope_semantics": "prior_empirical_capability_evidence_is_not_erased_by_later_policy_single_rollout_failure",
            "candidate_policy_semantics": "candidate_is_a_capability_probe_and_single_policy_realization_candidate_not_the_definition_of_physical_feasibility",
            "global_core_noninferiority_margin": MAX_GLOBAL_CORE_COVERAGE_DROP,
            "phase_core_noninferiority_margin": MAX_PHASE_CORE_COVERAGE_DROP,
            "frontier_requires_success_in_both_phases": REQUIRE_BOUNDARY_SUCCESS_EACH_PHASE,
            "zero_regression_required_for_envelope_progression": False,
            "zero_regression_required_for_policy_authority": False,
            "panel_coverage_is_calibrated_per_state_success_probability": False,
        },
        "claim_boundary": {
            "empirical_local_frontier_progression_claim": frontier_progression,
            "single_policy_realization_retention_claim": policy_realization_retained,
            "physical_feasibility_limit_claim": False,
            "certified_safe_set_claim": False,
            "jce_jel_final_claim": False,
            "retrospective_result_may_select_next_policy": False if retrospective else True,
        },
        "training_transitions": 0,
        "environment_interactions": 0,
        "validation_data_used": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
    }
    result["decision_sha256"] = canonical_sha256(result)
    return result


def analyze_capability_progression_file(
    gate_summary_path: Path,
    *,
    retrospective: bool = False,
) -> dict[str, Any]:
    path = Path(gate_summary_path)
    return analyze_capability_progression(
        _read(path),
        gate_summary_path=path,
        retrospective=retrospective,
    )
