"""Select nearest unseen OOD parents when no in-support acquisition exists."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256
from dvgc.runtime import save_json


def select_boundary(records: list[dict], scores: list[dict], target: int) -> list[tuple[dict, dict]]:
    by_id = {str(row["id"]): row for row in records}
    best = {}
    for score in scores:
        if not score.get("unseen_parent"):
            continue
        candidate_id, parent = str(score["candidate_id"]), str(score["parent"])
        row = by_id[candidate_id]
        item = (float(score["normalized_training_distance"]), candidate_id, row, score)
        if parent not in best or item[:2] < best[parent][:2]:
            best[parent] = item
    ranked = sorted(best.values(), key=lambda item: item[:2])
    if len(ranked) < target:
        raise ValueError(f"only {len(ranked)}/{target} unseen root parents")
    return [(item[2], item[3]) for item in ranked[:target]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output-bank", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--target", type=int, default=4)
    args = parser.parse_args()
    output_bank, output_report = Path(args.output_bank), Path(args.output_report)
    if output_bank.exists() or output_report.exists():
        raise SystemExit("refusing overwrite OOD boundary selection")
    bank = SnapshotBank.load(args.bank)
    payload = json.loads(Path(args.scores).read_text())
    if payload.get("eligible_states") != 0:
        raise SystemExit("in-support candidates exist; OOD fallback is not authorized")
    selected = select_boundary(bank.records, payload["records"], args.target)
    rows = []
    for rank, (source, score) in enumerate(selected):
        row = copy.deepcopy(source)
        row.update({
            "artifact_role": "ood_boundary_active_learning_proposal",
            "safe_claim_allowed": False, "requires_fresh_branch_labels": True,
            "ood_boundary_rank": rank,
            "normalized_training_distance": float(score["normalized_training_distance"]),
            "training_support_radius_p95": float(score["training_support_radius_p95"]),
            "reachability_prediction_ood_untrusted": float(score["predicted_p_next"]),
            "root_parent_id": score["parent"],
        })
        rows.append(row)
    metadata = {
        "artifact_role": "ood_boundary_active_learning_bank",
        "safe_claim_allowed": False, "not_certified_tube": True,
        "selection_rule": "nearest unseen root-parent states outside p95 training support",
        "source_bank_sha256": file_sha256(args.bank),
        "scores_sha256": file_sha256(args.scores),
        "model_sha256": payload["model_sha256"],
    }
    output_bank.parent.mkdir(parents=True, exist_ok=True)
    SnapshotBank(rows, metadata).save(output_bank)
    report = {
        "status": "PASS", **metadata, "selected": len(rows),
        "unique_root_parents": len({row["root_parent_id"] for row in rows}),
        "distance_range": [min(row["normalized_training_distance"] for row in rows),
                           max(row["normalized_training_distance"] for row in rows)],
        "support_radius": rows[0]["training_support_radius_p95"],
        "output_bank": str(output_bank), "output_bank_sha256": file_sha256(output_bank),
    }
    save_json(output_report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
