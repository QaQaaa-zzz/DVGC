"""Freeze exact-safe states from isolated 32-branch stage audits."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256
from dvgc.runtime import save_json


def exact_safe(label: dict, required_branches: int) -> bool:
    return (label.get("label") == "positive"
            and int(label["n"]) == required_branches
            and int(label["s"]) == required_branches)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-bank", action="append", required=True)
    parser.add_argument("--audit-report", action="append", required=True)
    parser.add_argument("--output-tube", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--branches", type=int, default=32)
    args = parser.parse_args()
    output_tube, output_report = Path(args.output_tube), Path(args.output_report)
    if output_tube.exists() or output_report.exists():
        raise SystemExit("refusing to overwrite frozen Tube")
    if len(args.audit_bank) != len(args.audit_report):
        raise SystemExit("audit bank/report group counts differ")
    labels = {}
    seed_bases = []
    reports = []
    for path in args.audit_report:
        report = json.loads(Path(path).read_text())
        reports.append(report)
        seed_bases.append(report.get("seed_base"))
        for label in report["labels"]:
            candidate_id = str(label["candidate_id"])
            if candidate_id in labels:
                raise SystemExit(f"duplicate audited candidate: {candidate_id}")
            labels[candidate_id] = label
    if None in seed_bases or len(seed_bases) != len(set(seed_bases)):
        raise SystemExit("independent audit seed namespaces are absent or duplicated")
    safe_rows, boundary_rows = [], []
    for bank_path in args.audit_bank:
        bank = SnapshotBank.load(bank_path)
        for row in bank.records:
            label = labels.get(str(row["id"]))
            if label is None:
                raise SystemExit(f"audit result missing for {row['id']}")
            if exact_safe(label, args.branches):
                item = copy.deepcopy(row)
                item.update({
                    "artifact_role": "expert_conditioned_provisional_envelope",
                    "certified_safe": True,
                    "safe_claim_allowed": True,
                    "final_shared_policy_jel": False,
                    "stage_safe_definition": "valid next-stage entry on every independent branch",
                    "independent_branch_successes": int(label["s"]),
                    "independent_branch_count": int(label["n"]),
                    "independent_seed_base": next(report["seed_base"] for report in reports
                                                  if any(x["candidate_id"] == row["id"] for x in report["labels"])),
                })
                safe_rows.append(item)
            else:
                boundary_rows.append({"id": row["id"], "s": int(label["s"]), "n": int(label["n"])})
    if not safe_rows:
        raise SystemExit("independent audit produced no exact-safe states")
    identity = {
        "stage": args.stage,
        "branches": args.branches,
        "safe_ids": sorted(row["id"] for row in safe_rows),
        "audit_report_sha256s": [file_sha256(path) for path in args.audit_report],
    }
    tube_version = f"{args.stage}-expert-tube-{hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12]}"
    metadata = {
        "artifact_role": "expert_conditioned_provisional_envelope",
        "certified_tube": True,
        "independent_audit": True,
        "formal_shared_policy_jel": False,
        "stage": args.stage,
        "tube_version": tube_version,
        "safe_definition": "valid next-stage entry on every isolated 32-branch audit rollout",
        "branch_count": args.branches,
        "seed_bases": seed_bases,
        "audit_bank_sha256s": [file_sha256(path) for path in args.audit_bank],
        "audit_report_sha256s": identity["audit_report_sha256s"],
    }
    output_tube.parent.mkdir(parents=True, exist_ok=True)
    SnapshotBank(safe_rows, metadata).save(output_tube)
    failures = Counter(branch["failure_reason"] for report in reports for label in report["labels"]
                       for branch in label["branches"] if not branch["success"])
    result = {
        "status": "PASS", **metadata,
        "tube": str(output_tube), "tube_sha256": file_sha256(output_tube),
        "audited_states": len(labels), "safe_states": len(safe_rows),
        "boundary_states": len(boundary_rows), "boundary": boundary_rows,
        "branch_successes": sum(int(label["s"]) for label in labels.values()),
        "total_branches": sum(int(label["n"]) for label in labels.values()),
        "failure_reasons": dict(failures),
    }
    save_json(output_report, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
