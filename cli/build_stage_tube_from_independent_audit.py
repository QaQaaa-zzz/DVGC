"""Freeze exact-safe support from isolated 32-branch stage audits.

Local next-stage evidence creates certified proposal support, never a formal
Tube.  Only an explicit full-stack Final-Recovery audit may create an
expert-conditioned provisional envelope.
"""
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
    branches = label.get("branches", [])
    return (label.get("label") == "positive"
            and int(label["n"]) == required_branches
            and int(label["s"]) == required_branches
            and len(branches) == required_branches
            and all(branch.get("success") is True for branch in branches))


def final_safe(label: dict, required_branches: int) -> bool:
    branches = label.get("branches", [])
    return (len(branches) == required_branches
            and all(bool(branch.get("final_recovery")) for branch in branches))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-bank", action="append", required=True)
    parser.add_argument("--audit-report", action="append", required=True)
    parser.add_argument("--output-tube", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--branches", type=int, default=32)
    parser.add_argument("--evidence-scope", choices=("local_next_stage", "final_recovery"), required=True)
    parser.add_argument("--require-teacher-action-evidence", action="store_true")
    args = parser.parse_args()
    output_tube, output_report = Path(args.output_tube), Path(args.output_report)
    if output_tube.exists() or output_report.exists():
        raise SystemExit("refusing to overwrite frozen Tube")
    if len(args.audit_bank) != len(args.audit_report):
        raise SystemExit("audit bank/report group counts differ")
    labels = {}
    branch_seeds = []
    seed_bases = []
    reports = []
    controller_descriptors = []
    for path in args.audit_report:
        report = json.loads(Path(path).read_text())
        reports.append(report)
        seed_bases.append(report.get("seed_base"))
        descriptor = report.get("controller")
        if descriptor is not None and descriptor not in controller_descriptors:
            controller_descriptors.append(descriptor)
        for label in report["labels"]:
            candidate_id = str(label["candidate_id"])
            if candidate_id in labels:
                raise SystemExit(f"duplicate audited candidate: {candidate_id}")
            labels[candidate_id] = label
            seeds = [branch.get("seed", branch.get("branch_seed"))
                     for branch in label.get("branches", [])]
            if None in seeds:
                raise SystemExit(f"audit branches omit seeds for {candidate_id}")
            branch_seeds.extend(seeds)
    if None in seed_bases or len(seed_bases) != len(set(seed_bases)):
        raise SystemExit("independent audit seed namespaces are absent or duplicated")
    if len(branch_seeds) != len(set(branch_seeds)):
        raise SystemExit("independent audit branch seeds are not globally unique")
    safe_rows, boundary_rows = [], []
    for bank_path in args.audit_bank:
        bank = SnapshotBank.load(bank_path)
        for row in bank.records:
            label = labels.get(str(row["id"]))
            if label is None:
                raise SystemExit(f"audit result missing for {row['id']}")
            is_safe = (exact_safe(label, args.branches) if args.evidence_scope == "local_next_stage"
                       else final_safe(label, args.branches))
            if is_safe:
                item = copy.deepcopy(row)
                role = ("stage_entry_certified_proposal_support" if args.evidence_scope == "local_next_stage"
                        else "expert_conditioned_provisional_envelope")
                item.update({
                    "artifact_role": role,
                    "certified_safe": True,
                    "safe_claim_allowed": args.evidence_scope == "final_recovery",
                    "final_shared_policy_jel": False,
                    "stage_safe_definition": ("valid next-stage entry on every independent branch"
                                              if args.evidence_scope == "local_next_stage"
                                              else "Final-Recovery under immutable expert stack on every independent branch"),
                    "independent_branch_successes": int(label["s"]),
                    "independent_branch_count": int(label["n"]),
                    "independent_seed_base": next(report["seed_base"] for report in reports
                                                  if any(x["candidate_id"] == row["id"] for x in report["labels"])),
                    "certifying_controller_bank": list(label.get("controller_bank", [])),
                })
                teacher_evidence = [{
                    "branch_index": branch.get("branch_index"),
                    "seed": branch.get("seed"),
                    "dynamics_variant": branch.get("dynamics_variant"),
                    "first_action": branch.get("first_action"),
                    "action_sequence": branch.get("action_sequence"),
                } for branch in label.get("branches", [])
                    if branch.get("success") is True and branch.get("first_action") is not None]
                if args.require_teacher_action_evidence and len(teacher_evidence) != args.branches:
                    raise SystemExit(
                        f"safe state {row['id']} lacks successful teacher action evidence "
                        f"for all {args.branches} branches"
                    )
                if teacher_evidence:
                    item["certified_teacher_action_evidence"] = teacher_evidence
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
        "evidence_scope": args.evidence_scope,
    }
    version_kind = "entry-support" if args.evidence_scope == "local_next_stage" else "expert-tube"
    artifact_version = f"{args.stage}-{version_kind}-{hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12]}"
    role = ("stage_entry_certified_proposal_support" if args.evidence_scope == "local_next_stage"
            else "expert_conditioned_provisional_envelope")
    metadata = {
        "artifact_role": role,
        "certified_tube": args.evidence_scope == "final_recovery",
        "independent_audit": True,
        "formal_shared_policy_jel": False,
        "stage": args.stage,
        ("support_version" if args.evidence_scope == "local_next_stage" else "tube_version"): artifact_version,
        "evidence_scope": args.evidence_scope,
        "safe_definition": ("valid next-stage entry on every isolated 32-branch audit rollout"
                            if args.evidence_scope == "local_next_stage"
                            else "Final-Recovery under immutable expert stack on every isolated audit rollout"),
        "branch_count": args.branches,
        "seed_bases": seed_bases,
        "controller_descriptors": controller_descriptors,
        "audit_bank_sha256s": [file_sha256(path) for path in args.audit_bank],
        "audit_report_sha256s": identity["audit_report_sha256s"],
    }
    output_tube.parent.mkdir(parents=True, exist_ok=True)
    SnapshotBank(safe_rows, metadata).save(output_tube)
    failures = Counter(branch["failure_reason"] for report in reports for label in report["labels"]
                       for branch in label["branches"] if not branch["success"])
    result = {
        "status": "PASS", **metadata,
        "artifact": str(output_tube), "artifact_sha256": file_sha256(output_tube),
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
