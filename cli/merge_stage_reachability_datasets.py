"""Merge immutable next-stage reachability construction datasets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256
from dvgc.runtime import save_json
from cli.train_stage_reachability_model import parent_key


def merge_datasets(bank_paths: list[str], label_paths: list[str], stage: str):
    if len(bank_paths) != len(label_paths) or not bank_paths:
        raise ValueError("bank and label inputs must be non-empty one-to-one pairs")
    records: dict[str, dict] = {}
    labels: dict[str, dict] = {}
    sources = []
    for bank_path, label_path in zip(bank_paths, label_paths):
        bank = SnapshotBank.load(bank_path)
        payload = json.loads(Path(label_path).read_text())
        by_id = {str(row["id"]): row for row in bank.records}
        if len(by_id) != len(bank.records):
            raise ValueError(f"duplicate state id in {bank_path}")
        source_labels = payload.get("labels", [])
        if {str(row["candidate_id"]) for row in source_labels} != set(by_id):
            raise ValueError(f"state/label identity mismatch in {bank_path}")
        for label in source_labels:
            candidate_id = str(label["candidate_id"])
            if label.get("stage") not in (None, stage):
                raise ValueError(f"stage mismatch for {candidate_id}")
            if int(label["s"]) + int(label["n"] - label["s"]) != int(label["n"]):
                raise ValueError(f"invalid branch counts for {candidate_id}")
            row = by_id[candidate_id]
            expected_parent = parent_key(row)
            if str(label.get("trajectory_parent")) != expected_parent:
                raise ValueError(f"parent mismatch for {candidate_id}")
            if candidate_id in records:
                raise ValueError(f"duplicate state across inputs: {candidate_id}")
            records[candidate_id] = row
            labels[candidate_id] = label
        sources.append({"bank": bank_path, "bank_sha256": file_sha256(bank_path),
                        "labels": label_path, "labels_sha256": file_sha256(label_path)})
    ordered_ids = sorted(records)
    ordered_records = [records[key] for key in ordered_ids]
    ordered_labels = [labels[key] for key in ordered_ids]
    parents = {parent_key(row) for row in ordered_records}
    return ordered_records, ordered_labels, parents, sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", action="append", required=True)
    parser.add_argument("--labels", action="append", required=True)
    parser.add_argument("--stage", choices=("takeoff", "ascent", "apex", "descent", "landing"), required=True)
    parser.add_argument("--output-bank", required=True)
    parser.add_argument("--output-labels", required=True)
    args = parser.parse_args()
    output_bank, output_labels = Path(args.output_bank), Path(args.output_labels)
    if output_bank.exists() or output_labels.exists():
        raise SystemExit("refusing overwrite merged reachability dataset")
    records, labels, parents, sources = merge_datasets(args.bank, args.labels, args.stage)
    output_bank.parent.mkdir(parents=True, exist_ok=True)
    SnapshotBank(records, {
        "artifact_role": f"{args.stage}_reachability_construction_states",
        "safe_claim_allowed": False, "certified_tube": False, "sources": sources,
    }).save(output_bank)
    counts = {name: sum(row["label"] == name for row in labels)
              for name in sorted({row["label"] for row in labels})}
    save_json(output_labels, {
        "status": "PASS", "artifact_role": f"{args.stage}_reachability_construction_labels",
        "stage": args.stage, "safe_claim_allowed": False, "not_a_tube": True,
        "states": len(records), "parents": len(parents),
        "branches": sum(int(row["n"]) for row in labels),
        "branch_successes": sum(int(row["s"]) for row in labels),
        "formal_successes": sum(int(row.get("formal_successes", 0)) for row in labels),
        "final_successes": sum(int(row.get("final_successes", 0)) for row in labels),
        "label_counts": counts, "labels": labels, "sources": sources,
        "output_bank": str(output_bank), "output_bank_sha256": file_sha256(output_bank),
    })


if __name__ == "__main__":
    main()
