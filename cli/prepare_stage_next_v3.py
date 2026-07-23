"""Freeze the corrected Takeoff protocol and build a disjoint balanced eval bank."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256
from dvgc.runtime import save_json


def _even_unique(rows: list[dict], count: int) -> list[dict]:
    by_parent: dict[int, dict] = {}
    for row in rows:
        by_parent.setdefault(int(row.get("reference_index", -1)), row)
    values = list(by_parent.values())
    if len(rows) < count:
        raise ValueError(f"need {count} states, found {len(rows)}")
    chosen = ([values[int(i)] for i in np.linspace(0, len(values) - 1, count, dtype=int)]
              if len(values) >= count else list(values))
    used = {row["id"] for row in chosen}
    remaining = [row for row in rows if row["id"] not in used]
    if len(chosen) < count:
        chosen += [remaining[int(i)] for i in np.linspace(
            0, len(remaining) - 1, count - len(chosen), dtype=int
        )]
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--prior-training-bank", action="append", default=[])
    parser.add_argument("--output-bank", required=True)
    parser.add_argument("--output-report", required=True)
    args = parser.parse_args()

    source = SnapshotBank.load(args.bank)
    protocol = source.metadata.get("reset_protocol", {})
    if protocol.get("version") != "takeoff_reset_authenticity_v3":
        raise SystemExit("only takeoff_reset_authenticity_v3 can be frozen")
    excluded: set[str] = set()
    prior_hashes = []
    for path in args.prior_training_bank:
        prior = SnapshotBank.load(path)
        excluded.update(str(row["id"]) for row in prior.records)
        prior_hashes.append(file_sha256(path))
    eligible = [row for row in source.records if str(row["id"]) not in excluded]
    canonical = _even_unique(
        [row for row in eligible if row.get("candidate_kind") == "canonical_compressed"], 12
    )
    aligned = _even_unique(
        [row for row in eligible if row.get("candidate_kind") == "reference_aligned_compressed"], 12
    )
    rows = [copy.deepcopy(row) for pair in zip(canonical, aligned) for row in pair]
    contract = {
        "version": "takeoff_balanced_eval_v1",
        "source_bank_sha256": file_sha256(args.bank),
        "reset_protocol_sha256": source.metadata["reset_protocol_sha256"],
        "strata": {"canonical_compressed": 12, "reference_aligned_compressed": 12},
        "branches_per_state": 4,
        "selection": "parent-diverse evenly spaced after prior-training-ID exclusion",
        "future_training_excludes_evaluation_reference_parents": True,
        "historical_policies_predate_this_frozen_split": True,
        "prior_training_bank_sha256": prior_hashes,
    }
    contract["sha256"] = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    metadata = copy.deepcopy(source.metadata)
    metadata.update({
        "artifact_role": "fixed_takeoff_balanced_evaluation_bank",
        "evaluation_contract": contract,
        "accepted_and_frozen_reset_protocol": True,
        "training_allowed": False,
    })
    SnapshotBank(rows, metadata).save(args.output_bank)
    save_json(args.output_report, {
        "status": "PASS",
        "artifact_role": "takeoff_protocol_v3_freeze_and_eval_bank",
        "takeoff_reset_protocol_v3": "accepted_and_frozen",
        "source_bank": str(Path(args.bank).resolve()),
        "source_bank_sha256": file_sha256(args.bank),
        "evaluation_bank": str(Path(args.output_bank).resolve()),
        "evaluation_bank_sha256": file_sha256(args.output_bank),
        "evaluation_contract": contract,
        "excluded_training_ids": len(excluded),
        "candidate_ids": {
            kind: [row["id"] for row in rows if row["candidate_kind"] == kind]
            for kind in ("canonical_compressed", "reference_aligned_compressed")
        },
        "unique_reference_parents": {
            kind: len({row.get("reference_index") for row in rows if row["candidate_kind"] == kind})
            for kind in ("canonical_compressed", "reference_aligned_compressed")
        },
        "result_reclassification": {
            "takeoff_20260723_140623": "partial_controller_support",
            "label_pilot_v3_120x4": "single_controller_conditional_labels",
            "canonical_failures": "negative_under_policy_takeoff_20260723_140623",
            "reachability_model_e40235c": "source_confounded_pilot",
        },
    })


if __name__ == "__main__":
    main()
