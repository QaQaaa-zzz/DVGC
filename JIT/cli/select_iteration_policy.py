#!/usr/bin/env python3
"""Register one already-frozen unified policy as the selected pi_k handoff.

Historical selections may still use the original strict paired-gate semantics.
Future automatic iterations additionally provide a prospective
``jit_capability_progression_decision_v1`` artifact. That decision separates
cumulative envelope progression from single-policy realization coverage.

A retrospective capability reinterpretation may describe an already-observed
candidate, but it is forbidden from retroactively selecting that candidate as a
formal next-iteration authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from jit_dvgc.analysis.capability_progression import SCHEMA as CAPABILITY_DECISION_SCHEMA
from jit_dvgc.config import file_sha256
from jit_dvgc.unified_policy_freeze import load_frozen_unified_manifest


SCHEMA = "jit_selected_iteration_policy_v1"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


from jit_dvgc.evidence_integrity import validate_gate_endpoints


def _verify_capability_decision(
    path: Path,
    *,
    gate_summary: Path,
    gate: Mapping[str, Any],
    record: Mapping[str, Any],
) -> dict[str, Any]:
    validate_gate_endpoints(gate)
    decision = _read(path)
    if decision.get("schema") != CAPABILITY_DECISION_SCHEMA or decision.get("status") != "completed":
        raise ValueError("selected-policy capability decision schema/status drift")
    declared = str(decision.get("decision_sha256", ""))
    base = {key: value for key, value in decision.items() if key != "decision_sha256"}
    if len(declared) != 64 or _canonical(base) != declared:
        raise ValueError("selected-policy capability decision self-hash drift")
    if decision.get("source_gate_file_sha256") != file_sha256(gate_summary):
        raise ValueError("selected-policy capability decision gate identity drift")
    if int(decision.get("candidate_iteration", -1)) != int(record["iteration"]):
        raise ValueError("selected-policy capability candidate iteration drift")
    if decision.get("candidate_policy_name") != record["name"]:
        raise ValueError("selected-policy capability candidate name drift")
    if decision.get("candidate_actor_sha256") != record["actor_sha256"]:
        raise ValueError("selected-policy capability actor drift")
    if decision.get("candidate_payload_sha256") != record["payload_sha256"]:
        raise ValueError("selected-policy capability payload drift")
    if decision.get("retrospective_analysis") is not False:
        raise ValueError("retrospective capability analysis cannot formally select a policy")
    if decision.get("empirical_envelope_expansion_observed") is not True:
        raise ValueError("selected policy has no accepted frontier progression")
    if decision.get("candidate_policy_authority_eligible") is not True:
        raise ValueError("selected policy does not retain enough phase-aware Tube coverage")
    if decision.get("formal_prospective_selection_claim") is not True:
        raise ValueError("capability decision is not prospective selection evidence")
    if int(gate.get("candidate_iteration", -1)) != int(decision["candidate_iteration"]):
        raise ValueError("selected-policy gate/capability iteration mismatch")
    return decision


def select(
    *,
    frozen_policy: Path,
    gate_summary: Path,
    output_dir: Path,
    allow_baseline_reproduction_mismatch: bool,
    capability_decision: Path | None = None,
) -> dict[str, Any]:
    frozen_policy = Path(frozen_policy)
    gate_summary = Path(gate_summary)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"selection output already exists: {output_dir}")

    frozen = load_frozen_unified_manifest(frozen_policy)
    record = dict(frozen["policy"])
    gate = _read(gate_summary)
    if gate.get("schema") != "jit_paired_policy_gate_report_v1":
        raise ValueError("selected-policy gate schema drift")
    if gate.get("status") != "completed":
        raise ValueError("selected-policy gate is not completed")
    if int(gate.get("candidate_iteration", -1)) != int(record["iteration"]):
        raise ValueError("selected-policy candidate iteration drift")
    if gate.get("candidate_policy_name") != record["name"]:
        raise ValueError("selected-policy candidate name drift")
    if gate.get("candidate_actor_sha256") != record["actor_sha256"]:
        raise ValueError("selected-policy candidate actor drift")
    if gate.get("candidate_payload_sha256") != record["payload_sha256"]:
        raise ValueError("selected-policy candidate payload drift")
    if int(gate.get("source_iteration", -1)) + 1 != int(record["iteration"]):
        raise ValueError("selected-policy source/candidate iteration drift")

    validate_gate_endpoints(gate)
    core = gate.get("core_gate")
    boundary = gate.get("boundary_gate")
    if not isinstance(core, Mapping) or not isinstance(boundary, Mapping):
        raise ValueError("selected-policy gate sections missing")
    if int(core.get("state_count", -1)) <= 0:
        raise ValueError("selected-policy core gate is empty")
    if int(core.get("baseline_success_count", -1)) <= 0:
        raise ValueError("selected-policy core gate has no baseline-success support")

    boundary_success = int(boundary.get("candidate_success_count", 0))
    boundary_groups = int(boundary.get("candidate_success_parent_group_count", 0))
    minimum_groups = int(boundary.get("minimum_candidate_success_parent_groups", 1))
    reproduction_failures = int(boundary.get("baseline_reproduction_failure_count", 0))
    if boundary_success <= 0 or boundary_groups < minimum_groups:
        raise ValueError("selected policy has no sufficient empirical boundary gain")
    if reproduction_failures and not allow_baseline_reproduction_mismatch:
        raise ValueError(
            "historical gate has baseline-reproduction failures; rerun with the explicit "
            "engineering quarantine flag or repair the gate protocol"
        )

    strict_gate_accepted = bool(gate.get("iteration_accepted")) and reproduction_failures == 0
    decision: dict[str, Any] | None = None
    if capability_decision is not None:
        decision = _verify_capability_decision(
            Path(capability_decision),
            gate_summary=gate_summary,
            gate=gate,
            record=record,
        )
        selection_semantics = "prospective_capability_progression_v1"
        prospective_capability_selection = True
        formal_acceptance = True
    else:
        # Backward-compatible historical selection path. This is retained so
        # repair02/pi_1 provenance remains reproducible and is not rewritten by
        # the later method revision.
        if int(core.get("regression_count", -1)) != 0 or core.get("passed") is not True:
            raise ValueError("historical strict selection regresses a baseline-success core state")
        selection_semantics = "historical_strict_zero_regression"
        prospective_capability_selection = False
        formal_acceptance = strict_gate_accepted

    policy_realization = decision.get("policy_realization") if decision is not None else None
    frontier_progression = decision.get("frontier_progression") if decision is not None else None
    core_regressions = int(core.get("regression_count", 0))
    artifact = {
        "schema": SCHEMA,
        "status": "selected",
        "iteration": int(record["iteration"]),
        "policy_name": str(record["name"]),
        "frozen_policy": str(frozen_policy),
        "frozen_policy_file_sha256": file_sha256(frozen_policy),
        "actor_sha256": str(record["actor_sha256"]),
        "critic_sha256": str(record["critic_sha256"]),
        "normalizer_sha256": str(record["normalizer_sha256"]),
        "payload_sha256": str(record["payload_sha256"]),
        "xml_sha256": str(record["xml_sha256"]),
        "formal_config": str(record["formal_config"]),
        "formal_config_sha256": str(record["formal_config_sha256"]),
        "source_training_transitions": int(record["source_training_transitions"]),
        "selection_gate": str(gate_summary),
        "selection_gate_file_sha256": file_sha256(gate_summary),
        "selection_semantics": selection_semantics,
        "capability_decision": str(capability_decision) if capability_decision is not None else None,
        "capability_decision_file_sha256": (
            file_sha256(capability_decision) if capability_decision is not None else None
        ),
        "core_state_count": int(core["state_count"]),
        "core_baseline_success_count": int(core["baseline_success_count"]),
        "core_candidate_success_count": int(core["candidate_success_count"]),
        "core_regression_count": core_regressions,
        "strict_zero_regression_diagnostic_passed": core_regressions == 0,
        "strict_historical_gate_accepted": strict_gate_accepted,
        "boundary_state_count": int(boundary["state_count"]),
        "boundary_success_count": boundary_success,
        "boundary_success_parent_group_count": boundary_groups,
        "baseline_reproduction_failure_count": reproduction_failures,
        "empirical_envelope_expansion_observed": (
            bool(decision["empirical_envelope_expansion_observed"])
            if decision is not None
            else boundary_success > 0 and boundary_groups >= minimum_groups
        ),
        "policy_realization": policy_realization,
        "frontier_progression": frontier_progression,
        "engineering_selection": True,
        "formal_acceptance_claim": formal_acceptance,
        "prospective_capability_selection_claim": prospective_capability_selection,
        "baseline_reproduction_mismatch_quarantined": bool(reproduction_failures),
        "training_transitions": 0,
        "environment_interactions": 0,
        "validation_data_used": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "claim_boundary": {
            "selected_for_next_engineering_envelope_iteration": True,
            "historical_strict_gate_pass_claim": strict_gate_accepted,
            "prospective_capability_progression_selection_claim": prospective_capability_selection,
            "selection_semantics": selection_semantics,
            "cumulative_envelope_not_defined_by_latest_policy_only": decision is not None,
            "zero_single_state_regression_required": decision is None,
            "certified_safe_set_claim": False,
            "jce_jel_claim": False,
        },
    }
    artifact["selection_sha256"] = _canonical(artifact)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write(output_dir / "selected_policy.json", artifact)
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-policy", type=Path, required=True)
    parser.add_argument("--gate-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--capability-decision",
        type=Path,
        help="prospective jit_capability_progression_decision_v1 artifact for future iterations",
    )
    parser.add_argument(
        "--allow-baseline-reproduction-mismatch",
        action="store_true",
        help="historical engineering quarantine only; never converts an old formal gate to PASS",
    )
    args = parser.parse_args()
    result = select(
        frozen_policy=args.frozen_policy,
        gate_summary=args.gate_summary,
        output_dir=args.output_dir,
        allow_baseline_reproduction_mismatch=args.allow_baseline_reproduction_mismatch,
        capability_decision=args.capability_decision,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
