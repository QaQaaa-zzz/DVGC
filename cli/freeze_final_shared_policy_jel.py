"""Freeze states that pass construction and disjoint independent Final audits."""
from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter
from pathlib import Path

from dvgc.bank import SnapshotBank, beta_posterior
from dvgc.config import file_sha256
from dvgc.runtime import save_json


STAGES = ("takeoff", "ascent", "apex", "descent", "landing")


def exact_ids(report: dict, branches: int) -> set[str]:
    return {str(row["candidate_id"]) for row in report.get("labels", [])
            if int(row.get("n", 0)) == branches and int(row.get("s", -1)) == branches
            and row.get("label") == "positive"
            and len(row.get("branches", [])) == branches
            and all(branch.get("final_recovery") is True for branch in row["branches"])}


def calibration_metrics(construction_by_id: dict, audit_by_id: dict, branches: int) -> dict:
    pairs = []
    bins = [[] for _ in range(10)]
    for key in sorted(set(construction_by_id) & set(audit_by_id)):
        probability = float(construction_by_id[key]["s"]) / branches
        outcomes = [float(branch["final_recovery"]) for branch in audit_by_id[key]["branches"]]
        if len(outcomes) != branches:
            raise ValueError(f"audit branch count mismatch for {key}")
        pairs.extend((probability, outcome) for outcome in outcomes)
        bins[min(9, int(probability * 10))].extend((probability, outcome) for outcome in outcomes)
    if not pairs:
        raise ValueError("no common construction/audit states")
    brier = sum((probability - outcome) ** 2 for probability, outcome in pairs) / len(pairs)
    ece = sum(
        len(group) / len(pairs) * abs(
            sum(probability for probability, _ in group) / len(group)
            - sum(outcome for _, outcome in group) / len(group)
        )
        for group in bins if group
    )
    return {"brier": brier, "ece_10_bin": ece, "audit_branches": len(pairs)}


def formal_final_outcome(branches_per_round: int) -> dict:
    successes = 2 * int(branches_per_round)
    return {
        "successes": successes, "failures": 0, "branches": successes,
        "posterior": beta_posterior(successes, 0), "label": "safe",
        "semantics": "Final-Recovery under construction and disjoint independent audit",
    }


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
    for key in ("xml_sha256", "action_mapping_version", "root_candidate_bank_sha256"):
        if construction.get(key) != audit.get(key):
            raise SystemExit(f"construction and independent audit disagree on {key}")
    if construction.get("seed_namespace") == audit.get("seed_namespace"):
        raise SystemExit("independent audit reuses construction namespace")
    construction_seed_list = [branch["seed"] for row in construction["labels"] for branch in row["branches"]]
    audit_seed_list = [branch["seed"] for row in audit["labels"] for branch in row["branches"]]
    if (len(construction_seed_list) != len(set(construction_seed_list))
            or len(audit_seed_list) != len(set(audit_seed_list))):
        raise SystemExit("a Final audit contains duplicate branch seeds")
    construction_seeds = set(construction_seed_list); audit_seeds = set(audit_seed_list)
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
        construction_branches = [dict(branch, certification_round="construction")
                                 for branch in construction_by_id[str(record["id"])]["branches"]]
        independent_branches = [dict(branch, certification_round="independent_audit")
                                for branch in audit_by_id[str(record["id"])]["branches"]]
        all_branches = construction_branches + independent_branches
        chain_successes = sum(bool(branch.get("chain_ever")) for branch in all_branches)
        item.update({
            "artifact_role": "final_shared_policy_jel", "training_only": False,
            "certified_safe": True, "safe_claim_allowed": True,
            "tube_metrics_eligible": True,
            "formal_shared_policy_jel": True,
            "final_shared_policy_jel": True,
            "policy_params_sha256": construction["policy_params_sha256"],
            "policy_version": construction["policy_params_sha256"],
            "final": formal_final_outcome(args.branches),
            "chain": {
                "successes": chain_successes, "failures": len(all_branches) - chain_successes,
                "branches": len(all_branches),
                "posterior": beta_posterior(chain_successes, len(all_branches) - chain_successes),
                "label": "safe" if chain_successes == len(all_branches) else
                         ("dead" if chain_successes == 0 else "boundary"),
                "semantics": "C_L Chain observed separately from Final-Recovery",
            },
            "certification_branches": all_branches,
            "construction_final_branches": construction_branches,
            "independent_audit_final_branches": independent_branches,
        })
        rows.append(item)
    phase_counts = Counter(row["phase_rsi_stage"] for row in rows)
    missing_phases = [stage for stage in STAGES if phase_counts.get(stage, 0) == 0]
    if missing_phases:
        raise SystemExit(
            "final shared-policy JEL lacks independently safe support in phases: "
            + ", ".join(missing_phases)
        )
    root_count = int(construction.get("root_candidate_state_count", len(bank.records)))
    root_phase_counts = {stage: int(construction.get("root_phase_state_counts", {}).get(stage, 0))
                         for stage in STAGES}
    if root_count <= 0 or sum(root_phase_counts.values()) != root_count:
        raise SystemExit("root five-stage candidate coverage denominator is invalid")
    construction_by_id = {str(row["candidate_id"]): row for row in construction["labels"]}
    audit_by_id = {str(row["candidate_id"]): row for row in audit["labels"]}
    calibration = calibration_metrics(construction_by_id, audit_by_id, args.branches)
    precision = len(safe) / len(construction_safe) if construction_safe else math.nan
    recall = len(safe) / len(audit_safe) if audit_safe else math.nan
    metadata = {
        "artifact_role": "final_shared_policy_jel", "formal_shared_policy_jel": True,
        "certified_tube": True,
        "independent_audit": True, "safe_definition": (
            "Final-Recovery under one frozen shared Actor on 32 construction and 32 disjoint audit branches"
        ),
        "policy_params_sha256": construction["policy_params_sha256"],
        "xml_sha256": construction["xml_sha256"],
        "action_mapping_version": construction["action_mapping_version"],
        "canonical_entry_bank_sha256": construction["canonical_entry_bank_sha256"],
        "source_candidate_bank_sha256": file_sha256(args.bank),
        "root_source_candidate_bank_sha256": construction["root_candidate_bank_sha256"],
        "construction_report_sha256": file_sha256(args.construction_report),
        "independent_audit_report_sha256": file_sha256(args.independent_audit_report),
        "branches_per_state_per_round": args.branches,
    }
    output.parent.mkdir(parents=True, exist_ok=True); SnapshotBank(rows, metadata).save(output)
    report = {
        "status": "PASS", **metadata, "safe_states": len(rows),
        "source_states_at_final_funnel_level": len(bank.records),
        "root_source_states": root_count, "coverage": len(rows) / root_count,
        "phase_safe_counts": {stage: phase_counts.get(stage, 0) for stage in STAGES},
        "phase_root_counts": root_phase_counts,
        "phase_coverage": {stage: phase_counts.get(stage, 0) / root_phase_counts[stage]
                           for stage in STAGES},
        "construction_exact_safe": len(construction_safe), "independent_exact_safe": len(audit_safe),
        "independent_audit_precision": precision,
        "independent_audit_recall_within_final_funnel": recall,
        "construction_vs_audit_brier": calibration["brier"],
        "construction_vs_audit_ece_10_bin": calibration["ece_10_bin"],
        "independent_audit_branches_for_calibration": calibration["audit_branches"],
        "output_bank": str(output), "output_bank_sha256": file_sha256(output),
    }
    save_json(report_path, report); print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
