#!/usr/bin/env python3
"""Register one already-frozen unified policy as the selected pi_k handoff.

This command does not freeze, train, evaluate, or mutate a policy.  It binds an
existing frozen policy to an already-completed paired gate and writes a small,
self-hashed selection artifact that downstream iteration automation can trust.

A baseline-reproduction mismatch may be explicitly quarantined for engineering
selection.  In that case the artifact records ``formal_acceptance_claim=false``
and may not be used as evidence that the historical formal gate passed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

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


def select(
    *,
    frozen_policy: Path,
    gate_summary: Path,
    output_dir: Path,
    allow_baseline_reproduction_mismatch: bool,
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

    core = gate.get("core_gate")
    boundary = gate.get("boundary_gate")
    if not isinstance(core, Mapping) or not isinstance(boundary, Mapping):
        raise ValueError("selected-policy gate sections missing")
    if int(core.get("state_count", -1)) <= 0:
        raise ValueError("selected-policy core gate is empty")
    if int(core.get("baseline_success_count", -1)) <= 0:
        raise ValueError("selected-policy core gate has no baseline-success support")
    if int(core.get("regression_count", -1)) != 0 or core.get("passed") is not True:
        raise ValueError("selected policy regresses a baseline-success core state")
    # Later iterations preserve the capability actually demonstrated by pi_k on
    # Tube_k.  They do not require pi_k itself to solve every guidance state in
    # Tube_k; unsuccessful baseline states are allowed to become improvements.
    boundary_success = int(boundary.get("candidate_success_count", 0))
    boundary_groups = int(boundary.get("candidate_success_parent_group_count", 0))
    minimum_groups = int(boundary.get("minimum_candidate_success_parent_groups", 1))
    if boundary_success <= 0 or boundary_groups < minimum_groups:
        raise ValueError("selected policy has no sufficient empirical boundary gain")

    reproduction_failures = int(boundary.get("baseline_reproduction_failure_count", 0))
    if reproduction_failures and not allow_baseline_reproduction_mismatch:
        raise ValueError(
            "historical gate has baseline-reproduction failures; rerun with the explicit "
            "engineering quarantine flag or repair the gate protocol"
        )

    formal_acceptance = bool(gate.get("iteration_accepted")) and reproduction_failures == 0
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
        "core_state_count": int(core["state_count"]),
        "core_baseline_success_count": int(core["baseline_success_count"]),
        "core_candidate_success_count": int(core["candidate_success_count"]),
        "core_regression_count": 0,
        "boundary_state_count": int(boundary["state_count"]),
        "boundary_success_count": boundary_success,
        "boundary_success_parent_group_count": boundary_groups,
        "baseline_reproduction_failure_count": reproduction_failures,
        "engineering_selection": True,
        "formal_acceptance_claim": formal_acceptance,
        "baseline_reproduction_mismatch_quarantined": bool(reproduction_failures),
        "training_transitions": 0,
        "environment_interactions": 0,
        "validation_data_used": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "claim_boundary": {
            "selected_for_next_engineering_envelope_iteration": True,
            "historical_formal_gate_pass_claim": formal_acceptance,
            "core_preservation_semantics": "zero_pi_k_success_to_pi_kplus1_failure_on_declared_source_core",
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
        "--allow-baseline-reproduction-mismatch",
        action="store_true",
        help="engineering quarantine only; never converts the historical formal gate to PASS",
    )
    args = parser.parse_args()
    result = select(
        frozen_policy=args.frozen_policy,
        gate_summary=args.gate_summary,
        output_dir=args.output_dir,
        allow_baseline_reproduction_mismatch=args.allow_baseline_reproduction_mismatch,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
