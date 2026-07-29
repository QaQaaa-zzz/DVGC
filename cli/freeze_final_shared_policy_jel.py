"""Freeze states that pass construction and disjoint independent Final audits."""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256
from dvgc.runtime import save_json


STAGES = ("takeoff", "ascent", "apex", "descent", "landing")


def exact_ids(report: dict, branches: int) -> set[str]:
    return {str(row["candidate_id"]) for row in report.get("labels", [])
            if int(row.get("n", 0)) == branches and int(row.get("s", -1)) == branches
            and row.get("label") == "positive"
            and len(row.get("branches", [])) == branches
            and all(branch.get("final_recovery") is True for branch in row["branches"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--construction-report", required=True)
    parser.add_argument("--independent-audit-report", required=True)
    parser.add_argument("--output-bank", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--branches", type=int, default=32)
    args = parser.parse_args()
    output, report_path = Path(args.output_bank), Path(args.output_report)
    if output.exists() or report_path.exists():
        raise SystemExit("refusing to overwrite final shared-policy JEL")
    bank = SnapshotBank.load(args.bank)
    construction = json.loads(Path(args.construction_report).read_text())
    audit = json.loads(Path(args.independent_audit_report).read_text())
    for name, report in (("construction", construction), ("audit", audit)):
        if (report.get("status") != "PASS"
                or report.get("artifact_role") != "final_shared_policy_branch_audit"
                or int(report.get("branches_per_state", 0)) != args.branches):
            raise SystemExit(f"{name} is not a valid {args.branches}-branch Final audit")
        if report.get("candidate_bank_sha256") != file_sha256(args.bank):
            raise SystemExit(f"{name} candidate bank identity mismatch")
    if construction.get("policy_params_sha256") != audit.get("policy_params_sha256"):
        raise SystemExit("construction and independent audit use different frozen policies")
    if construction.get("canonical_entry_bank_sha256") != audit.get("canonical_entry_bank_sha256"):
        raise SystemExit("construction and independent audit use different C_L banks")
    if construction.get("seed_namespace") == audit.get("seed_namespace"):
        raise SystemExit("independent audit reuses construction namespace")
    construction_seeds = {branch["seed"] for row in construction["labels"] for branch in row["branches"]}
    audit_seeds = {branch["seed"] for row in audit["labels"] for branch in row["branches"]}
    if construction_seeds & audit_seeds:
        raise SystemExit("independent Final audit reuses construction branch seeds")
    construction_safe = exact_ids(construction, args.branches)
    audit_safe = exact_ids(audit, args.branches)
    safe = construction_safe & audit_safe
    if not safe:
        raise SystemExit("no state passed both frozen-policy Final audits")
    construction_by_id = {str(row["candidate_id"]): row for row in construction["labels"]}
    audit_by_id = {str(row["candidate_id"]): row for row in audit["labels"]}
    rows = []
    for record in bank.records:
        if str(record["id"]) not in safe:
            continue
        item = copy.deepcopy(record)
        item.update({
            "artifact_role": "final_shared_policy_jel", "training_only": False,
            "certified_safe": True, "safe_claim_allowed": True,
            "formal_shared_policy_jel": True,
            "final_shared_policy_jel": True,
            "policy_params_sha256": construction["policy_params_sha256"],
            "construction_final_branches": construction_by_id[str(record["id"])]["branches"],
            "independent_audit_final_branches": audit_by_id[str(record["id"])]["branches"],
        })
        rows.append(item)
    phase_counts = Counter(row["phase_rsi_stage"] for row in rows)
    metadata = {
        "artifact_role": "final_shared_policy_jel", "formal_shared_policy_jel": True,
        "independent_audit": True, "safe_definition": (
            "Final-Recovery under one frozen shared Actor on 32 construction and 32 disjoint audit branches"
        ),
        "policy_params_sha256": construction["policy_params_sha256"],
        "canonical_entry_bank_sha256": construction["canonical_entry_bank_sha256"],
        "source_candidate_bank_sha256": file_sha256(args.bank),
        "construction_report_sha256": file_sha256(args.construction_report),
        "independent_audit_report_sha256": file_sha256(args.independent_audit_report),
        "branches_per_state_per_round": args.branches,
    }
    output.parent.mkdir(parents=True, exist_ok=True); SnapshotBank(rows, metadata).save(output)
    report = {
        "status": "PASS", **metadata, "safe_states": len(rows),
        "source_states": len(bank.records), "coverage": len(rows) / len(bank.records),
        "phase_safe_counts": {stage: phase_counts.get(stage, 0) for stage in STAGES},
        "construction_exact_safe": len(construction_safe), "independent_exact_safe": len(audit_safe),
        "output_bank": str(output), "output_bank_sha256": file_sha256(output),
    }
    save_json(report_path, report); print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
