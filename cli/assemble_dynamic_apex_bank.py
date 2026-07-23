"""Merge validated Apex proposals while preserving independent parent lineage."""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np

from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256
from dvgc.runtime import save_json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-bank", required=True)
    p.add_argument("--new-bank", required=True)
    p.add_argument("--output-bank", required=True)
    p.add_argument("--output-report", required=True)
    p.add_argument("--parent-cap", type=int, default=4)
    p.add_argument("--dedup-distance", type=float, default=.05)
    a = p.parse_args()
    base = SnapshotBank.load(a.base_bank)
    new = SnapshotBank.load(a.new_bank)
    scale = np.asarray(base.metadata["augmentation_feature_scale"], float)
    scale = np.maximum(scale, np.asarray([
        .05, .05, .05, .05, .05, .05, .2, .2, .2,
        .2, .2, .2, .1, .1, .2, .2,
    ]))
    rows, parent_counts = [], Counter()
    rejected = Counter()
    for source_name, records in (("base", base.records), ("new", new.records)):
        for source in records:
            row = copy.deepcopy(source)
            dynamic = bool(row.get("dynamically_reached")
                           or row.get("candidate_kind") == "apex_dynamically_reached")
            if dynamic:
                # Legacy parent-131 rows encoded controller/seed in
                # trajectory_parent_id.  Their shared source_parent_id is the
                # independent trajectory parent; new acquisitions already use
                # the canonical hashed trajectory_parent_id.
                parent = str(row.get(
                    "independent_trajectory_parent_id",
                    row.get("source_parent_id", row.get("trajectory_parent_id")),
                ))
                if parent_counts[parent] >= a.parent_cap:
                    rejected["parent_cap"] += 1
                    continue
            feature = np.asarray(row["physical_feature"], float)
            if any(np.linalg.norm(
                    (feature - np.asarray(old["physical_feature"], float)) / scale
            ) < a.dedup_distance for old in rows):
                rejected["normalized_duplicate"] += 1
                continue
            row["apex_bank_source"] = source_name
            if dynamic:
                row["candidate_kind"] = "apex_dynamically_reached"
                row["apex_support_class"] = "dynamically_reached_candidate"
                row["independent_trajectory_parent_id"] = parent
                parent_counts[parent] += 1
            rows.append(row)
    anchors = [row for row in rows if row["candidate_kind"] ==
               "apex_reference_anchor_reset_valid"]
    dynamic = [row for row in rows if row["candidate_kind"] ==
               "apex_dynamically_reached"]
    dynamic_parents = {
        str(row["independent_trajectory_parent_id"]) for row in dynamic
    }
    status = "PASS" if 16 <= len(dynamic) <= 32 and len(dynamic_parents) >= 4 else "FAIL"
    SnapshotBank(rows, {
        "artifact_role": "dynamic_apex_proposal_bank",
        "certified_tube": False, "safe_claim_allowed": False,
        "base_bank_sha256": file_sha256(a.base_bank),
        "new_bank_sha256": file_sha256(a.new_bank),
        "augmentation_feature_scale": scale.tolist(),
        "per_parent_snapshot_cap": a.parent_cap,
        "dedup_distance": a.dedup_distance,
    }).save(a.output_bank)
    save_json(a.output_report, {
        "status": status, "artifact_role": "dynamic_apex_bank_assembly",
        "bank": str(Path(a.output_bank).resolve()),
        "bank_sha256": file_sha256(a.output_bank),
        "reference_reset_valid": len(anchors),
        "dynamically_reached": len(dynamic),
        "dynamic_parent_count": len(dynamic_parents),
        "descent_positive": 0,
        "parent_snapshot_counts": dict(parent_counts),
        "rejections": dict(rejected),
        "late_ascent_training_authorized": len(dynamic_parents) >= 2,
        "apex_training_authorized": status == "PASS",
        "reference_valid_is_not_dynamic_support": True,
    })
    print(json.dumps({
        "status": status, "anchors": len(anchors), "dynamic": len(dynamic),
        "parents": len(dynamic_parents), "rejections": dict(rejected),
    }, indent=2))


if __name__ == "__main__":
    main()
