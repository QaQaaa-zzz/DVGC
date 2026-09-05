"""CPU-only identity checks shared by labeling and legacy selection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_verified_protocol(path: Path) -> dict:
    protocol = json.loads(Path(path).read_text())
    base = {k: v for k, v in protocol.items() if k != "protocol_sha256"}
    if canonical_sha256(base) != protocol.get("protocol_sha256"):
        raise ValueError("label protocol self-hash drift")
    return protocol


def validate_gate_endpoints(gate: Mapping[str, Any]) -> str:
    core = gate.get("core_source", {})
    endpoints = (core.get("baseline_success_criterion"),
                 core.get("candidate_success_criterion"),
                 gate.get("boundary_source", {}).get("success_criterion"))
    if any(value not in {"first_valid_landing", "stable_recovery"} for value in endpoints):
        raise ValueError("paired gate endpoint identity missing or unsupported")
    if len(set(endpoints)) != 1:
        raise ValueError("paired gate mixes core/boundary success endpoints")
    return endpoints[0]


def validate_label_row(row: Mapping[str, Any], *, name: str, actor: str,
                       payload: str, criterion: str) -> None:
    expected = {"evaluator_policy_name": name, "evaluator_actor_sha256": actor,
                "evaluator_payload_sha256": payload, "success_criterion": criterion}
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"label row {key} drift")
    if type(row.get("label")) is not int or row["label"] not in (0, 1):
        raise ValueError("completed label must be an integer binary outcome")
    if row.get("continuation_success") is not bool(row["label"]):
        raise ValueError("label/continuation_success drift")
    cost = row.get("environment_interactions")
    if type(cost) is not int or cost < 0:
        raise ValueError("invalid label interaction count")
    if row.get("outcome_class") in {None, "engineering_error", "incomplete", "untested"}:
        raise ValueError("incomplete/error outcome cannot be a completed label")
    if criterion == "first_valid_landing":
        expected_outcomes = ({"first_valid_landing"} if row["label"] else {
            "airborne_physical_failure", "timeout_before_landing",
            "task_failure_before_landing", "horizon_exhausted_before_landing",
        })
        if row["outcome_class"] not in expected_outcomes:
            raise ValueError("first-landing label/outcome class drift")
        if "valid_contact_seen" in row and bool(row["valid_contact_seen"]) != bool(row["label"]):
            raise ValueError("first-landing label/contact evidence drift")
