"""Lifecycle and provenance helpers for immutable pointwise audits."""
from __future__ import annotations

from typing import Mapping


TERMINAL_AUDIT_STATUSES = {"COMPLETED_PASS", "COMPLETED_FAIL", "INVALID"}


def terminal_audit_status(analysis_status: str) -> str:
    value = str(analysis_status).upper()
    if value == "PASS":
        return "COMPLETED_PASS"
    if value == "FAIL":
        return "COMPLETED_FAIL"
    raise ValueError(f"Unsupported audit analysis status: {analysis_status}")


def completed_manifest(manifest: Mapping, analysis: Mapping, merged: Mapping) -> dict:
    """Return a terminal launch manifest after strict provenance checks."""
    checks = {
        "policy": (manifest.get("policy_hash"), merged.get("descent_policy_hash")),
        "candidate": (manifest.get("candidate_bank_sha256"), merged.get("candidate_bank_sha256")),
        "C_L": (manifest.get("landing_entry_set_sha256"), merged.get("landing_entry_set_sha256")),
        "pi_L": (manifest.get("landing_policy_hash"), merged.get("landing_policy_hash")),
    }
    failed = [name for name, (left, right) in checks.items() if not left or left != right]
    if failed:
        raise ValueError(f"Pointwise terminal manifest provenance mismatch: {failed}")
    result = dict(manifest)
    result.update(
        {
            "status": terminal_audit_status(str(analysis.get("status"))),
            "analysis_status": str(analysis.get("status")),
            "completed_states": int(merged.get("states", 0)),
            "completed_branches": int(merged.get("terminal_summary", {}).get("branches", 0)),
        }
    )
    if result["completed_states"] != int(result.get("states", -1)):
        raise ValueError("Pointwise terminal manifest state coverage mismatch")
    return result


def invalid_manifest(manifest: Mapping, reason: str) -> dict:
    result = dict(manifest)
    result.update({"status": "INVALID", "invalid_reason": str(reason)})
    return result
