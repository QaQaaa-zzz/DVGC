"""Add explicit next-stage semantics to an existing atomic entry bank."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from cli.stage_label_pilot import NEXT_STAGE
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256
from dvgc.runtime import save_json


def normalize(records: list[dict], evaluation: dict) -> list[dict]:
    provenance = {}
    for label in evaluation["labels"]:
        for branch in label["branches"]:
            entry_id = branch.get("entry_snapshot_id")
            if entry_id:
                provenance[entry_id] = label["candidate_id"]
    output = []
    for source in records:
        row = copy.deepcopy(source)
        stage = row.get("entry_from_stage")
        if stage not in NEXT_STAGE or row.get("entry_to_stage") != NEXT_STAGE[stage]:
            raise ValueError(f"Invalid stage-entry semantics for {row.get('id')}")
        if row["id"] not in provenance:
            raise ValueError(f"Entry {row['id']} is absent from evaluation branches")
        row["upstream_candidate_id"] = provenance[row["id"]]
        if row["entry_to_stage"] in ("ascent", "apex", "descent"):
            row["flight_subinterval"] = row["entry_to_stage"]
        output.append(row)
    if len({row["id"] for row in output}) != len(output):
        raise ValueError("Entry IDs must be unique")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-bank", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--output-bank", required=True)
    parser.add_argument("--output-report", required=True)
    args = parser.parse_args()
    output = Path(args.output_bank)
    report = Path(args.output_report)
    if output.exists() or report.exists():
        raise SystemExit("Refusing to overwrite an existing normalized artifact")
    source = SnapshotBank.load(args.input_bank)
    evaluation = json.loads(Path(args.evaluation).read_text())
    rows = normalize(source.records, evaluation)
    metadata = copy.deepcopy(source.metadata)
    metadata.update({
        "artifact_role": "normalized_stage_entry_proposal_bank",
        "source_entry_bank_sha256": file_sha256(args.input_bank),
        "source_evaluation_sha256": file_sha256(args.evaluation),
        "not_certified_tube": True,
    })
    SnapshotBank(rows, metadata).save(output)
    save_json(report, {
        "status": "PASS", "records": len(rows),
        "subintervals": sorted({row.get("flight_subinterval") for row in rows}),
        "source_entry_bank_sha256": file_sha256(args.input_bank),
        "source_evaluation_sha256": file_sha256(args.evaluation),
        "output_bank_sha256": file_sha256(output),
        "physical_state_modified": False,
    })


if __name__ == "__main__":
    main()
