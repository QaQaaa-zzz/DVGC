"""Build construction-only Apex reachability labels from fixed feedback runs."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256
from dvgc.runtime import save_json
from dvgc.stage_reachability import reachability_label


def first_stable_tick(outcome: dict) -> int | None:
    for row in outcome.get("trace", []):
        if row.get("stable_physical_descent"):
            return int(row["tick"])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-bank", action="append", required=True)
    parser.add_argument("--feedback-report", action="append", required=True)
    parser.add_argument("--output-bank", required=True)
    parser.add_argument("--output-labels", required=True)
    parser.add_argument("--required-parents", type=int, default=5)
    args = parser.parse_args()
    output_bank, output_labels = Path(args.output_bank), Path(args.output_labels)
    if output_bank.exists() or output_labels.exists():
        raise SystemExit("refusing overwrite Apex feedback dataset")
    states = {}
    for path in args.state_bank:
        bank = SnapshotBank.load(path)
        for row in bank.records:
            states.setdefault(str(row["id"]), row)
    fresh_by_parent = {}
    report_hashes = []
    for path in args.feedback_report:
        report = json.loads(Path(path).read_text())
        report_hashes.append(file_sha256(path))
        for outcome in report["outcomes"]:
            if outcome["branch_kind"] != "fresh_dynamics":
                continue
            parent = str(outcome["parent"])
            fresh_by_parent.setdefault(parent, []).append(outcome)
    if len(fresh_by_parent) != args.required_parents:
        raise SystemExit(f"fresh feedback labels cover {len(fresh_by_parent)}/{args.required_parents} parents")
    records, labels = [], []
    for parent in sorted(fresh_by_parent):
        outcomes = sorted(fresh_by_parent[parent], key=lambda row: int(row["branch"]))
        if len(outcomes) != 4 or len({int(row["branch"]) for row in outcomes}) != 4:
            raise SystemExit(f"parent {parent} does not have exactly four unique fresh branches")
        start_ids = {str(row["start_snapshot_id"]) for row in outcomes}
        if len(start_ids) != 1:
            raise SystemExit(f"parent {parent} mixes feedback start snapshots")
        start_id = next(iter(start_ids))
        if start_id not in states:
            raise SystemExit(f"feedback start snapshot is absent: {start_id}")
        item = copy.deepcopy(states[start_id])
        item.update({
            "id": start_id,
            "trajectory_parent_id": str(outcomes[0]["parent_id"]),
            "display_parent": parent,
            "candidate_kind": "apex_feedback_authority_state",
            "flight_subinterval": "apex",
            "artifact_role": "proposal_support_bank",
            "safe_claim_allowed": False,
            "feedback_controller": "receding_horizon_bounded_shooting_v1",
        })
        records.append(item)
        branches = []
        for outcome in outcomes:
            success = bool(outcome["stable_physical_descent"])
            branches.append({
                "branch_index": int(outcome["branch"]),
                "seed": int(outcome["seed"]),
                "dynamics_variant": outcome["dynamics_variant"],
                "success": success,
                "time_to_next_stage": first_stable_tick(outcome) if success else None,
                "failure_reason": None if success else outcome["termination_reason"],
                "formal_descent_support_entry": bool(outcome["formal_descent_support_entry"]),
                "final_recovery": bool(outcome["final_landing_recovery"]),
            })
        label = reachability_label(
            stage="apex", successes=sum(row["success"] for row in branches),
            branches=4, branch_records=branches, controller_bank_exhausted=True,
        )
        label.update({
            "candidate_id": start_id,
            "candidate_kind": item["candidate_kind"],
            "trajectory_parent": item["trajectory_parent_id"],
            "controller_bank": ["receding_horizon_bounded_shooting_v1"],
            "local_success_semantics": "four consecutive stable physical Descent ticks",
            "formal_successes": sum(row["formal_descent_support_entry"] for row in branches),
            "final_successes": sum(row["final_recovery"] for row in branches),
        })
        labels.append(label)
    output_bank.parent.mkdir(parents=True, exist_ok=True)
    SnapshotBank(records, {
        "artifact_role": "apex_feedback_reachability_construction_states",
        "safe_claim_allowed": False, "certified_tube": False,
        "feedback_report_sha256s": report_hashes,
        "state_bank_sha256s": [file_sha256(path) for path in args.state_bank],
    }).save(output_bank)
    counts = {name: sum(label["label"] == name for label in labels)
              for name in sorted({label["label"] for label in labels})}
    save_json(output_labels, {
        "status": "PASS", "artifact_role": "apex_feedback_construction_labels",
        "safe_claim_allowed": False, "not_a_tube": True,
        "local_success_semantics": "four consecutive stable physical Descent ticks",
        "formal_matcher_unchanged": True,
        "parents": len(labels), "labels": labels, "label_counts": counts,
        "branch_successes": sum(label["s"] for label in labels),
        "branches": sum(label["n"] for label in labels),
        "formal_successes": sum(label["formal_successes"] for label in labels),
        "final_successes": sum(label["final_successes"] for label in labels),
        "output_bank": str(output_bank), "output_bank_sha256": file_sha256(output_bank),
        "feedback_report_sha256s": report_hashes,
    })


if __name__ == "__main__":
    main()
