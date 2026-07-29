"""Promote exact-success states to the next independent branch-audit level."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256
from dvgc.runtime import save_json


def exact_survivor_ids(labels: list[dict], required_branches: int) -> set[str]:
    return {str(row["candidate_id"]) for row in labels
            if int(row["n"]) == required_branches
            and int(row["s"]) == required_branches
            and row.get("label") == "positive"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--required-branches", type=int, choices=(4, 8), required=True)
    parser.add_argument("--next-branches", type=int, choices=(8, 32), required=True)
    parser.add_argument("--output-bank", required=True)
    parser.add_argument("--output-report", required=True)
    args = parser.parse_args()
    if (args.required_branches, args.next_branches) not in ((4, 8), (8, 32)):
        raise SystemExit("only the preregistered 4->8->32 funnel is permitted")
    output_bank, output_report = Path(args.output_bank), Path(args.output_report)
    if output_bank.exists() or output_report.exists():
        raise SystemExit("refusing overwrite branch-funnel output")
    bank = SnapshotBank.load(args.bank)
    report = json.loads(Path(args.report).read_text())
    labels = report.get("labels", [])
    if len(labels) != len(bank.records):
        raise SystemExit("audit report does not cover the complete input bank")
    if {str(row["candidate_id"]) for row in labels} != {str(row["id"]) for row in bank.records}:
        raise SystemExit("audit state identities do not match the input bank")
    survivors = exact_survivor_ids(labels, args.required_branches)
    rows = []
    for row in bank.records:
        if str(row["id"]) not in survivors:
            continue
        item = copy.deepcopy(row)
        item.update({"artifact_role": "independent_audit_candidate",
                     "safe_claim_allowed": False,
                     "passed_branch_level": args.required_branches,
                     "requires_branch_level": args.next_branches})
        rows.append(item)
    if not rows:
        raise SystemExit("no exact-success states survived the current audit level")
    output_bank.parent.mkdir(parents=True, exist_ok=True)
    root_source_hash = bank.metadata.get("root_source_bank_sha256", file_sha256(args.bank))
    SnapshotBank(rows, {
        "artifact_role": "independent_audit_candidate_bank",
        "safe_claim_allowed": False, "not_certified_tube": True,
        "passed_branch_level": args.required_branches,
        "requires_branch_level": args.next_branches,
        "source_bank_sha256": file_sha256(args.bank),
        "root_source_bank_sha256": root_source_hash,
        "source_report_sha256": file_sha256(args.report),
        "root_source_bank_sha256": root_source_hash,
    }).save(output_bank)
    payload = {
        "status": "PASS", "artifact_role": "stage_branch_audit_funnel",
        "safe_claim_allowed": False, "not_certified_tube": True,
        "input_states": len(bank.records), "survivors": len(rows),
        "passed_branch_level": args.required_branches,
        "next_branch_level": args.next_branches,
        "survivor_ids": sorted(survivors),
        "output_bank": str(output_bank), "output_bank_sha256": file_sha256(output_bank),
        "source_report_sha256": file_sha256(args.report),
    }
    save_json(output_report, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
