"""Select the latest empirically effective pre-Apex feedback start per parent."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256
from dvgc.runtime import save_json


def choose_start(candidates, classification, info, forced_offset=None):
    if forced_offset is not None:
        matches = [row for row in candidates if row["relative_to_apex"] == forced_offset]
        if len(matches) != 1:
            raise ValueError(f"expected one snapshot at relative_to_apex={forced_offset}, got {len(matches)}")
        return matches[0], "pre-registered forced event-relative authority offset"
    if classification == "apex_local_response_detected":
        return max(candidates, key=lambda row: row["relative_to_apex"]), "Apex-local rank and roll authority remain effective"
    if info["latest_effective_relative_to_apex"] is not None:
        return next(row for row in candidates if row["relative_to_apex"] == info["latest_effective_relative_to_apex"]), \
            "latest pre-Apex offset with rank-two pose response"
    return min(candidates, key=lambda row: row["relative_to_apex"]), \
        "no later effective roll authority; earliest bounded diagnostic"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--authority-bank", required=True)
    p.add_argument("--authority-report", required=True)
    p.add_argument("--parent", action="append", required=True)
    p.add_argument("--output-bank", required=True)
    p.add_argument("--output-report", required=True)
    p.add_argument("--relative-to-apex", type=int)
    a = p.parse_args()
    bank = SnapshotBank.load(a.authority_bank)
    audit = json.loads(Path(a.authority_report).read_text())
    selected = []
    rows = []
    for display_parent in a.parent:
        info = audit["parent_results"][display_parent]
        candidates = [
            row for row in bank.records
            if row["trajectory_parent_id"] == info["parent_id"]
        ]
        classification = (
            "apex_local_response_detected"
            if info["classification"] == "apex_local_correctable"
            else info["classification"]
        )
        chosen, reason = choose_start(candidates, classification, info, a.relative_to_apex)
        chosen = dict(chosen)
        chosen["candidate_kind"] = "selected_pre_apex_feedback_bridge_start"
        chosen["control_authority_class"] = classification
        chosen["bridge_start_selection_reason"] = reason
        selected.append(chosen)
        rows.append({
            "display_parent": display_parent, "parent_id": info["parent_id"],
            "classification": info["classification"],
            "snapshot_id": chosen["id"],
            "relative_to_apex": chosen["relative_to_apex"],
            "reason": reason,
        })
    SnapshotBank(selected, {
        "artifact_role": "selected_pre_apex_feedback_bridge_starts",
        "certified_tube": False, "safe_claim_allowed": False,
        "authority_bank_sha256": file_sha256(a.authority_bank),
        "authority_report_sha256": file_sha256(a.authority_report),
    }).save(a.output_bank)
    save_json(a.output_report, {
        "status": "PASS",
        "artifact_role": "pre_apex_feedback_bridge_start_selection",
        "authority_bank_sha256": file_sha256(a.authority_bank),
        "authority_report_sha256": file_sha256(a.authority_report),
        "forced_relative_to_apex": a.relative_to_apex,
        "parents": rows,
        "output_bank": str(Path(a.output_bank).resolve()),
        "output_bank_sha256": file_sha256(a.output_bank),
    })


if __name__ == "__main__":
    main()
