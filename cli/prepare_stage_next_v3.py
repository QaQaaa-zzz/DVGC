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


def _heldout_parent_states(rows: list[dict], count: int, parent_count: int = 4) -> list[dict]:
    by_parent: dict[int, dict] = {}
    groups: dict[int, list[dict]] = {}
    for row in rows:
        by_parent.setdefault(int(row.get("reference_index", -1)), row)
        groups.setdefault(int(row.get("reference_index", -1)), []).append(row)
    per_parent = count // parent_count
    parents = sorted(parent for parent, values in groups.items() if len(values) >= per_parent)
    if len(rows) < count:
        raise ValueError(f"need {count} states, found {len(rows)}")
    if len(parents) < parent_count:
        raise ValueError(f"need {parent_count} held-out parents, found {len(parents)}")
    heldout = [parents[int(i)] for i in np.linspace(
        0, len(parents) - 1, parent_count, dtype=int
    )]
    chosen = []
    for parent in heldout:
        values = groups[parent]
        if len(values) < per_parent:
            raise ValueError(f"parent {parent} has only {len(values)} states")
        chosen.extend(values[int(i)] for i in np.linspace(
            0, len(values) - 1, per_parent, dtype=int
        ))
    if len(chosen) != count:
        raise ValueError("count must divide held-out parent count")
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
    canonical = _heldout_parent_states(
        [row for row in eligible if row.get("candidate_kind") == "canonical_compressed"], 12
    )
    aligned = _heldout_parent_states(
        [row for row in eligible if row.get("candidate_kind") == "reference_aligned_compressed"], 12
    )
    rows = [copy.deepcopy(row) for pair in zip(canonical, aligned) for row in pair]
    contract = {
        "version": "takeoff_balanced_eval_v2",
        "source_bank_sha256": file_sha256(args.bank),
        "reset_protocol_sha256": source.metadata["reset_protocol_sha256"],
        "strata": {"canonical_compressed": 12, "reference_aligned_compressed": 12},
        "branches_per_state": 4,
        "selection": "four held-out reference parents per stratum, three states per parent",
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
