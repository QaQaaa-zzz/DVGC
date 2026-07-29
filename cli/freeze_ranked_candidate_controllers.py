"""Freeze one construction-selected controller for each ranked candidate.

Controller assignment may use construction branches, but the selected banks
carry no outcome label.  Fresh branch certification must establish safety.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256
from dvgc.runtime import save_json


def assign_controller(label: dict, controller_order: list[str]) -> tuple[str, dict[str, int]]:
    successes: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    for branch in label["branches"]:
        controller = str(branch["controller_id"])
        totals[controller] += 1
        successes[controller] += int(bool(branch["success"]))
    missing = set(controller_order) - set(totals)
    if missing:
        raise ValueError(f"construction label lacks controller branches: {sorted(missing)}")
    # Registry order is the preregistered deterministic tie break.
    chosen = max(controller_order, key=lambda item: (successes[item] / totals[item], -controller_order.index(item)))
    return chosen, {item: int(successes[item]) for item in controller_order}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-bank", required=True)
    parser.add_argument("--construction-labels", required=True)
    parser.add_argument("--controller-bank")
    parser.add_argument("--controller-policy", action="append")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    root = Path(args.output_root)
    if root.exists():
        raise SystemExit(f"refusing overwrite {root}")
    candidates = SnapshotBank.load(args.candidate_bank)
    label_report = json.loads(Path(args.construction_labels).read_text())
    if bool(args.controller_bank) == bool(args.controller_policy):
        raise SystemExit("provide exactly one of --controller-bank or --controller-policy")
    if args.controller_bank:
        registry = json.loads(Path(args.controller_bank).read_text())
        policies = registry["policies"]
        registry_sha256 = file_sha256(args.controller_bank)
    else:
        policies = [{"id": f"controller_{index}", "path": str(path),
                     "params_sha256": file_sha256(Path(path) / "params.pkl")}
                    for index, path in enumerate(args.controller_policy)]
        registry = {"status": "PASS", "artifact_role": "controller_proposal_bank",
                    "construction_assignment_only": True, "policies": policies}
        root.mkdir(parents=True)
        save_json(root / "controller_bank.json", registry)
        registry_sha256 = file_sha256(root / "controller_bank.json")
    controller_order = [str(policy["params_sha256"]) for policy in policies]
    policy_by_hash = {str(policy["params_sha256"]): policy for policy in policies}
    labels = {str(row["candidate_id"]): row for row in label_report["labels"]}
    grouped: dict[str, list[dict]] = defaultdict(list)
    assignments = []
    for row in candidates.records:
        candidate_id = str(row["id"])
        if candidate_id not in labels:
            raise SystemExit(f"candidate lacks construction label: {candidate_id}")
        chosen, successes = assign_controller(labels[candidate_id], controller_order)
        item = copy.deepcopy(row)
        item.update({
            "selected_controller_id": chosen,
            "selected_controller_path": policy_by_hash[chosen]["path"],
            "controller_assignment_source": "construction_only_branch_labels",
            "safe_claim_allowed": False,
            "requires_fresh_branch_certification": True,
        })
        grouped[chosen].append(item)
        assignments.append({
            "candidate_id": candidate_id,
            "selected_controller_id": chosen,
            "construction_successes_by_controller": successes,
            "used_as_safety_label": False,
        })
    root.mkdir(parents=True, exist_ok=True)
    outputs = []
    for index, controller in enumerate(controller_order):
        rows = grouped.get(controller, [])
        if not rows:
            continue
        output = root / f"controller_{index}_candidates.pkl"
        SnapshotBank(rows, {
            "artifact_role": "controller_assigned_reachability_proposals",
            "safe_claim_allowed": False,
            "requires_fresh_branch_certification": True,
            "controller_id": controller,
            "controller_path": policy_by_hash[controller]["path"],
            "source_candidate_bank_sha256": file_sha256(args.candidate_bank),
            "construction_labels_sha256": file_sha256(args.construction_labels),
        }).save(output)
        outputs.append({"controller_id": controller, "controller_path": policy_by_hash[controller]["path"],
                        "candidate_bank": str(output), "candidate_bank_sha256": file_sha256(output),
                        "states": len(rows)})
    save_json(root / "manifest.json", {
        "status": "PASS",
        "artifact_role": "frozen_candidate_controller_assignment",
        "selection_evidence_role": "construction_only",
        "safe_claim_allowed": False,
        "fresh_branch_certification_required": True,
        "candidate_bank_sha256": file_sha256(args.candidate_bank),
        "construction_labels_sha256": file_sha256(args.construction_labels),
        "controller_bank_sha256": registry_sha256,
        "outputs": outputs,
        "assignments": assignments,
    })
    print(json.dumps({"status": "PASS", "states": len(assignments), "groups": outputs}, indent=2))


if __name__ == "__main__":
    main()
