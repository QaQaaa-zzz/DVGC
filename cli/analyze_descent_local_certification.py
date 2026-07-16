"""Analyze current-policy descent certification without reusing old labels."""
from __future__ import annotations

import argparse
import json
from collections import Counter

from cli.build_descent_entries import snapshot_identity
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256
from dvgc.runtime import save_json


def summarize(records, evidence_by_id):
    evidence = [item for row in records for item in evidence_by_id.get(row["id"], [])]
    n = len(evidence)
    return {
        "states": len(records), "branches": n,
        "chain_rate": sum(bool(item.get("chain_success")) for item in evidence) / n if n else 0.0,
        "final_rate": sum(bool(item.get("final_recovery")) for item in evidence) / n if n else 0.0,
        "physical_failure_rate": sum(item.get("terminal_cause") == "physical_failure" for item in evidence) / n if n else 0.0,
        "timeout_rate": sum(item.get("terminal_cause") in ("stage_timeout", "horizon") for item in evidence) / n if n else 0.0,
        "termination_reasons": dict(Counter(item.get("terminal_cause", "unknown") for item in evidence)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--cert-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    bank = SnapshotBank.load(args.bank)
    cert = json.load(open(args.cert_report, encoding="utf-8"))
    records = bank.records_for_phase("flight", include_training_only=False)
    evidence = {row["id"]: row["branch_evidence"] for row in cert["rows"]}
    labels = Counter(row["final"]["label"] for row in records)
    representatives = {}
    for row in records:
        representatives.setdefault(snapshot_identity(row), row)
    safe = [row for row in representatives.values() if row["final"]["label"] == "safe"]
    unique_safe = set(representatives) & {snapshot_identity(row) for row in safe}
    safe_sources = {str(row.get("entry_source_id", row.get("parent_candidate_id", row["id"]))) for row in safe}
    original = [row for row in records if row.get("candidate_kind") == "descent_diagnostic_anchor"]
    old_safe = [row for row in original if row.get("old_policy_label") == "safe"]
    old_boundary = [row for row in original if row.get("old_policy_label") == "boundary"]
    report = {
        "status": "PASS", "bank_sha256": file_sha256(args.bank),
        "policy_hash": cert["descent_policy_hash"], "candidate_source_policy_hash": cert["candidate_source_policy_hash"],
        "labels": dict(labels), "unique_state_count": len(representatives), "unique_final_safe_states": len(unique_safe),
        "safe_source_count": len(safe_sources), "safe_sources": sorted(safe_sources),
        "minimum_tube_support_ready": len(unique_safe) >= 4 and len(safe_sources) >= 2,
        "overall": summarize(records, evidence),
        "original_70": summarize(original, evidence),
        "old_provisional_safe": {**summarize(old_safe, evidence), "current_labels": dict(Counter(row["final"]["label"] for row in old_safe))},
        "old_boundary": {**summarize(old_boundary, evidence), "current_labels": dict(Counter(row["final"]["label"] for row in old_boundary))},
        "groups": {name: summarize([row for row in records if row.get("bootstrap_group") == name], evidence) for name in ("provisional_safe", "boundary", "successful_anchor")},
        "layers": {name: summarize([row for row in records if row.get("descent_layer") == name], evidence) for name in ("late", "middle", "early")},
        "health": {"nonfinite": cert["terminal_summary"].get("nonfinite", 0), "timeouts": cert["terminal_summary"].get("timeouts", 0)},
    }
    save_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
