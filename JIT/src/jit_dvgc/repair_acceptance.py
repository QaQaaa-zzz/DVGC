"""Generic fresh acceptance-bank preparation for iterative pi_k -> pi_(k+1) repair."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import file_sha256
from .soft_tube import load_soft_tube
from .training import load_frozen_unified_manifest, load_unified_formal_config


REPAIR_ACCEPTANCE_SCHEMA = "jit_repair_acceptance_boundary_acquisition_v1"
PAIRED_GATE_BANK_SCHEMAS = {
    "jit_paired_policy_gate_bank_v1",
    "jit_paired_policy_gate_bank_v2",
}
PHASES = ("upstream", "downstream")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _gate_declaration(root: Path) -> tuple[dict[str, Any], set[str], dict[str, set[str]]]:
    root = Path(root)
    summary_path = root / "summary.json"
    bank_path = root / "bank.json"
    records_path = root / "records.json"
    for path in (summary_path, bank_path, records_path):
        if not path.is_file():
            raise FileNotFoundError(f"consumed paired-gate artifact missing: {path}")

    summary = _read_json(summary_path)
    bank = _read_json(bank_path)
    records = _read_json(records_path)
    if summary.get("status") != "completed":
        raise ValueError(f"consumed paired gate is not completed: {root}")
    if bank.get("schema") not in PAIRED_GATE_BANK_SCHEMAS:
        raise ValueError(f"consumed paired gate bank schema unsupported: {root}")
    if not isinstance(records.get("records"), list):
        raise ValueError(f"consumed paired gate records missing: {root}")
    if summary.get("bank_sha256") != bank.get("bank_sha256"):
        raise ValueError(f"consumed paired gate summary/bank identity mismatch: {root}")

    boundary = bank.get("boundary")
    if not isinstance(boundary, list) or not boundary:
        raise ValueError(f"consumed paired gate boundary bank missing: {root}")
    states: set[str] = set()
    groups: dict[str, set[str]] = {phase: set() for phase in PHASES}
    for row in boundary:
        phase = str(row.get("phase", ""))
        state_sha = str(row.get("state_sha256", ""))
        parent = str(row.get("parent_group_id", ""))
        if phase not in groups:
            raise ValueError(f"consumed paired gate boundary phase invalid: {root}")
        if len(state_sha) != 64 or not parent:
            raise ValueError(f"consumed paired gate boundary identity invalid: {root}")
        states.add(state_sha)
        groups[phase].add(parent)
    if len(states) != len(boundary):
        raise ValueError(f"consumed paired gate boundary contains duplicate states: {root}")

    declaration = {
        "root": str(root),
        "protocol_sha256": str(summary["protocol_sha256"]),
        "summary_file_sha256": file_sha256(summary_path),
        "bank_file_sha256": file_sha256(bank_path),
        "records_file_sha256": file_sha256(records_path),
        "bank_schema": str(bank["schema"]),
        "boundary_state_count": len(states),
        "boundary_parent_group_counts": {
            phase: len(groups[phase]) for phase in PHASES
        },
        "source_iteration": int(summary["source_iteration"]),
        "candidate_iteration": int(summary["candidate_iteration"]),
        "baseline_actor_sha256": str(summary["baseline_actor_sha256"]),
        "baseline_payload_sha256": str(summary["baseline_payload_sha256"]),
    }
    return declaration, states, groups


def consumed_gate_exclusions(
    protocol: Mapping[str, Any],
) -> tuple[set[str], dict[str, set[str]], dict[str, Any]]:
    """Validate and union every previously exposed paired-gate boundary.

    Historical predeclarations used ``consumed_gate``. New repair rounds use
    ``consumed_gates`` so every exposed boundary is excluded cumulatively.
    """
    raw = protocol.get("consumed_gates")
    if raw is None:
        legacy = protocol.get("consumed_gate")
        if not isinstance(legacy, Mapping):
            raise ValueError("repair acceptance consumed gate declaration missing")
        declarations = [dict(legacy)]
    else:
        if not isinstance(raw, list) or not raw:
            raise ValueError("repair acceptance consumed_gates must be a nonempty list")
        if not all(isinstance(value, Mapping) for value in raw):
            raise ValueError("repair acceptance consumed_gates entries must be objects")
        declarations = [dict(value) for value in raw]

    roots = [str(value.get("root", "")) for value in declarations]
    if any(not root for root in roots) or len(set(roots)) != len(roots):
        raise ValueError("repair acceptance consumed gate roots must be unique and nonempty")

    source_iteration = int(protocol.get("source_iteration", -1))
    candidate_iteration = int(protocol.get("candidate_iteration", -1))
    baseline_actor = str(protocol.get("baseline_actor_sha256", ""))
    baseline_payload = str(protocol.get("baseline_payload_sha256", ""))

    all_states: set[str] = set()
    all_groups: dict[str, set[str]] = {phase: set() for phase in PHASES}
    verified: list[dict[str, Any]] = []
    for declared in declarations:
        actual, states, groups = _gate_declaration(Path(str(declared["root"])))
        for field in (
            "protocol_sha256",
            "summary_file_sha256",
            "bank_file_sha256",
            "records_file_sha256",
            "boundary_state_count",
        ):
            if str(declared.get(field)) != str(actual[field]):
                raise ValueError(
                    f"repair acceptance consumed gate {field} drift: {declared['root']}"
                )
        if actual["source_iteration"] != source_iteration:
            raise ValueError("repair acceptance consumed gate source iteration drift")
        if actual["candidate_iteration"] != candidate_iteration:
            raise ValueError("repair acceptance consumed gate candidate iteration drift")
        if actual["baseline_actor_sha256"] != baseline_actor:
            raise ValueError("repair acceptance consumed gate baseline actor drift")
        if actual["baseline_payload_sha256"] != baseline_payload:
            raise ValueError("repair acceptance consumed gate baseline payload drift")
        all_states.update(states)
        for phase in PHASES:
            all_groups[phase].update(groups[phase])
        verified.append(actual)

    return all_states, all_groups, {
        "gate_count": len(verified),
        "gates": verified,
        "union_boundary_state_count": len(all_states),
        "union_boundary_parent_group_counts": {
            phase: len(all_groups[phase]) for phase in PHASES
        },
    }


def prepare_repair_acceptance_predeclaration(
    *,
    baseline_frozen_policy: Path,
    target_tube: Path,
    consumed_gate_roots: Sequence[Path],
    acquisition_seed: int,
    labeling_seed: int,
    anchors_per_phase: int = 10,
    minimum_anchors_per_phase: int = 10,
    frontier_score_ceiling: float = 1.0,
    strengths: Sequence[float] = (0.15, 0.30, 0.50),
    durations: Sequence[int] = (2, 4, 8),
    action_names: Sequence[str] = (
        "steer",
        "rear_wheel_drive",
        "hip",
        "knee",
    ),
    signs: Sequence[int] = (-1, 1),
    active_action_dimensions: int = 2,
    minimum_negative_states_per_phase: int = 10,
    minimum_negative_parent_groups_per_phase: int = 3,
) -> dict[str, Any]:
    """Build an immutable fresh-bank declaration before replacement training."""
    baseline_frozen_policy = Path(baseline_frozen_policy)
    target_tube = Path(target_tube)
    roots = tuple(Path(root) for root in consumed_gate_roots)
    if not roots:
        raise ValueError("repair acceptance requires at least one consumed gate")
    if len({str(root) for root in roots}) != len(roots):
        raise ValueError("repair acceptance consumed gate roots must be unique")
    if int(acquisition_seed) < 0 or int(labeling_seed) < 0:
        raise ValueError("repair acceptance protocol seeds must be nonnegative")
    if anchors_per_phase <= 0 or minimum_anchors_per_phase <= 0:
        raise ValueError("repair acceptance anchor counts must be positive")
    if minimum_anchors_per_phase > anchors_per_phase:
        raise ValueError("repair acceptance minimum anchors exceed requested anchors")
    if not 0.0 <= float(frontier_score_ceiling) <= 1.0:
        raise ValueError("repair acceptance frontier score ceiling must lie in [0,1]")
    if active_action_dimensions <= 0 or active_action_dimensions > len(action_names):
        raise ValueError("repair acceptance active action dimensions invalid")
    if set(signs) != {-1, 1}:
        raise ValueError("repair acceptance signs must contain exactly -1 and +1")
    if any(float(value) <= 0.0 for value in strengths):
        raise ValueError("repair acceptance strengths must be positive")
    if any(int(value) <= 0 for value in durations):
        raise ValueError("repair acceptance durations must be positive")

    frozen = load_frozen_unified_manifest(baseline_frozen_policy)
    baseline = dict(frozen["policy"])
    if baseline.get("policy_role") != "envelope_expansion_authority":
        raise ValueError("repair acceptance baseline is not an expansion authority")
    source_iteration = int(baseline["iteration"])
    if baseline.get("name") != f"pi_{source_iteration}":
        raise ValueError("repair acceptance baseline policy name drift")
    candidate_iteration = source_iteration + 1

    formal = load_unified_formal_config(Path(str(baseline["formal_config"])))
    if formal.config_sha256 != baseline["formal_config_sha256"]:
        raise ValueError("repair acceptance baseline formal config identity drift")
    source_tube = load_soft_tube(Path(formal.soft_tube_path))
    if source_tube.manifest.get("manifest_sha256") != formal.soft_tube_manifest_sha256:
        raise ValueError("repair acceptance baseline source Tube drift")

    target = load_soft_tube(target_tube)
    target_manifest_sha = str(target.manifest.get("manifest_sha256", ""))
    if len(target_manifest_sha) != 64:
        raise ValueError("repair acceptance target Tube manifest invalid")

    consumed: list[dict[str, Any]] = []
    for root in roots:
        declaration, _states, _groups = _gate_declaration(root)
        if declaration["source_iteration"] != source_iteration:
            raise ValueError("consumed gate source iteration does not match baseline")
        if declaration["candidate_iteration"] != candidate_iteration:
            raise ValueError("consumed gate candidate iteration does not match k -> k+1")
        if declaration["baseline_actor_sha256"] != baseline["actor_sha256"]:
            raise ValueError("consumed gate baseline actor does not match frozen pi_k")
        if declaration["baseline_payload_sha256"] != baseline["payload_sha256"]:
            raise ValueError("consumed gate baseline payload does not match frozen pi_k")
        consumed.append(declaration)

    protocol = {
        "schema": "jit_repair_acceptance_boundary_acquisition_protocol_v1",
        "status": "predeclared_before_repair_training",
        "purpose": (
            "fresh_nonfinal_acceptance_bank_for_next_replacement_candidate_"
            "with_cumulative_consumed_gate_exclusion"
        ),
        "source_iteration": source_iteration,
        "candidate_iteration": candidate_iteration,
        "baseline_frozen_policy": str(baseline_frozen_policy),
        "baseline_policy_name": str(baseline["name"]),
        "baseline_actor_sha256": str(baseline["actor_sha256"]),
        "baseline_payload_sha256": str(baseline["payload_sha256"]),
        "source_tube_manifest_sha256": str(formal.soft_tube_manifest_sha256),
        "consumed_gates": consumed,
        "design_diagnosis": {
            "candidate_policy_outcomes_inspected_on_consumed_gates": True,
            "next_candidate_training_started": False,
            "fresh_acceptance_bank_required_before_next_candidate_training": True,
        },
        "acquisition": {
            "anchors_per_phase": int(anchors_per_phase),
            "minimum_anchors_per_phase": int(minimum_anchors_per_phase),
            "frontier_score_ceiling": float(frontier_score_ceiling),
            "strengths": [float(value) for value in strengths],
            "durations": [int(value) for value in durations],
            "action_names": [str(value) for value in action_names],
            "signs": [int(value) for value in signs],
            "active_action_dimensions": int(active_action_dimensions),
            "protocol_seed": int(acquisition_seed),
        },
        "labeling": {
            "policy_mode": "deterministic",
            "max_ticks": int(formal.ppo.episode_horizon),
            "protocol_seed": int(labeling_seed),
            "baseline_policy_only": True,
        },
        "bank_lock": {
            "selection": "all_baseline_continuation_negative_candidates",
            "exclude_target_tube_states": True,
            "target_tube": str(target_tube),
            "target_tube_manifest_sha256": target_manifest_sha,
            "physical_state_unique": True,
            "require_both_phases": True,
            "minimum_negative_states_per_phase": int(
                minimum_negative_states_per_phase
            ),
            "minimum_negative_parent_groups_per_phase": int(
                minimum_negative_parent_groups_per_phase
            ),
            "candidate_training_must_not_start_before_lock": True,
        },
        "isolation": {
            "exclude_consumed_boundary_states": True,
            "exclude_consumed_boundary_parent_groups_by_phase": True,
            "validation_data_used": False,
            "test_data_used": False,
            "final_evaluation_data_used": False,
        },
        "future_use": {
            "candidate_generation_before_repair_training": True,
            "baseline_label_before_repair_training": True,
            "lock_bank_before_repair_training": True,
            "forbid_repair_training_use": True,
            "forbid_tube_construction_use": True,
            "forbid_continuation_field_fit_use": True,
        },
        "claim_boundary": {
            "nonfinal_iteration_acceptance_audit_only": True,
            "candidate_policy_outcomes_inspected": False,
            "jce_jel_claim": False,
            "certified_safe_set_claim": False,
        },
    }
    expected = canonical_sha256(protocol)
    result = {
        "schema": REPAIR_ACCEPTANCE_SCHEMA,
        "expected_protocol_sha256": expected,
        "protocol": protocol,
    }
    consumed_gate_exclusions(protocol)
    return result


def prepare_core_replay_repair_config(
    *,
    base_config_path: Path,
    locked_acceptance_bank: Path,
    failed_gate_root: Path,
    core_probability: float,
    run_id: str,
) -> dict[str, Any]:
    """Prepare a fresh replacement candidate while changing only core replay mass."""
    base_config_path = Path(base_config_path)
    locked_acceptance_bank = Path(locked_acceptance_bank)
    failed_gate_root = Path(failed_gate_root)
    config = _read_json(base_config_path)
    if config.get("schema") != "jit_pi_unified_formal_v1":
        raise ValueError("repair training base config is not unified formal v1")

    tube_sampling = config.get("tube_sampling")
    if not isinstance(tube_sampling, Mapping):
        raise ValueError("repair training base config has no core replay contract")
    old_core = float(tube_sampling["core_probability"])
    new_core = float(core_probability)
    if not old_core < new_core < 1.0:
        raise ValueError("next repair core probability must increase and remain below 1")
    new_expansion = 1.0 - new_core

    bank = _read_json(locked_acceptance_bank)
    if bank.get("status") not in {
        "locked_before_repair_training",
        "locked_before_candidate_training",
    }:
        raise ValueError("fresh acceptance bank is not locked before training")
    if bank.get("artifact_role") != (
        "fresh_nonfinal_baseline_negative_iteration_acceptance_bank"
    ):
        raise ValueError("fresh acceptance bank artifact role drift")
    if bank.get("claim_boundary", {}).get("candidate_policy_outcomes_inspected") is not False:
        raise ValueError("fresh acceptance bank has inspected candidate outcomes")

    inputs = config.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("repair training config inputs missing")
    if bank.get("target_tube_manifest_sha256") != inputs.get(
        "soft_tube_manifest_sha256"
    ):
        raise ValueError("fresh acceptance bank target Tube does not match training Tube")

    summary_path = failed_gate_root / "summary.json"
    summary = _read_json(summary_path)
    if summary.get("status") != "completed" or summary.get("iteration_accepted") is not False:
        raise ValueError("repair training requires one completed rejected gate")
    source_iteration = int(summary["source_iteration"])
    candidate_iteration = int(summary["candidate_iteration"])
    if candidate_iteration != source_iteration + 1:
        raise ValueError("repair training failed gate is not k -> k+1")

    replacement = json.loads(json.dumps(config))
    replacement["tube_sampling"]["core_probability"] = new_core
    replacement["tube_sampling"]["expansion_probability"] = new_expansion
    replacement["run_declaration"] = {
        "run_id": str(run_id),
        "output_dir": f"JIT/runs/pi_unified/{run_id}",
        "purpose": (
            "replacement_candidate_after_failed_core_preservation_gate_"
            "with_stronger_retained_core_replay"
        ),
        "status": "predeclared_not_started",
    }
    claim = dict(replacement.get("claim_boundary", {}))
    claim.update(
        {
            "iteration": candidate_iteration,
            "policy_name": f"pi_{candidate_iteration}",
            "source_policy_name": f"pi_{source_iteration}",
            "candidate_revision": "stronger_core_replay_repair",
            "method_revision_after_nonfinal_gate": True,
            "single_variable_iteration": False,
            "single_variable_vs_previous_candidate": (
                "tube_sampling_core_vs_expansion_probability_only"
            ),
            "fresh_initialization_matches_previous_candidate": True,
            "ppo_hyperparameters_match_previous_candidate": True,
            "reward_physics_action_semantics_unchanged": True,
            "reset_probabilities_match_previous_candidate": True,
            "soft_tube_support_matches_previous_candidate": True,
            "previous_core_probability": old_core,
            "replacement_core_probability": new_core,
            "replacement_expansion_probability": new_expansion,
            "locked_acceptance_bank": str(locked_acceptance_bank),
            "locked_acceptance_bank_file_sha256": file_sha256(
                locked_acceptance_bank
            ),
            "locked_acceptance_bank_sha256": str(bank.get("bank_sha256", "")),
            "acceptance_bank_locked_before_training": True,
            "failed_gate_root": str(failed_gate_root),
            "failed_gate_summary_sha256": file_sha256(summary_path),
            "failed_gate_core_regression_count": int(
                summary["core_gate"]["regression_count"]
            ),
            "failed_gate_boundary_candidate_success_count": int(
                summary["boundary_gate"]["candidate_success_count"]
            ),
            "failed_gate_boundary_candidate_success_parent_group_count": int(
                summary["boundary_gate"]["candidate_success_parent_group_count"]
            ),
            "core_preservation_claim": False,
            "boundary_gain_claim": False,
            "empirical_envelope_expansion_claim": False,
            "final_policy_claim": False,
            "certified_safe_tube_claim": False,
            "jce_jel_claim": False,
            "test_data_used": False,
            "validation_data_used": False,
        }
    )
    replacement["claim_boundary"] = claim
    return replacement
