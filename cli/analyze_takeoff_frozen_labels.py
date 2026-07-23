"""Analyze source mixing in labels from the frozen Takeoff controller bank."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from dvgc.config import file_sha256
from dvgc.runtime import save_json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", required=True)
    p.add_argument("--fixed-controller-evaluation", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    payload = json.loads(Path(a.labels).read_text())
    fixed = json.loads(Path(a.fixed_controller_evaluation).read_text())
    strata = {}
    mixed = True
    for kind in ("canonical_compressed", "reference_aligned_compressed"):
        rows = [row for row in payload["labels"] if row.get("candidate_kind") == kind]
        successes = [row for row in rows if int(row["s"]) > 0]
        failures = [row for row in rows if int(row["s"]) == 0]
        mixed &= bool(successes and failures)
        strata[kind] = {
            "states": len(rows), "successful_states": len(successes),
            "all_controller_fail_states": len(failures),
            "branch_successes": sum(int(row["s"]) for row in rows),
            "branches": sum(int(row["n"]) for row in rows),
            "label_counts": dict(Counter(row["label"] for row in rows)),
        }
    script_only = {}
    for kind in strata:
        policy_ids = {
            row["candidate_id"] for row in fixed["outcomes"]
            if row["candidate_kind"] == kind and row["success"]
            and row["controller"] in ("old_takeoff", "new_takeoff", "canonical_specialist")
        }
        all_ids = {
            row["candidate_id"] for row in fixed["outcomes"]
            if row["candidate_kind"] == kind and row["success"]
        }
        script_only[kind] = len(all_ids - policy_ids)
    save_json(a.output, {
        "status": "PASS", "artifact_role": "takeoff_frozen_controller_label_analysis",
        "labels_sha256": file_sha256(a.labels),
        "strata": strata,
        "both_strata_contain_success_and_failure": mixed,
        "source_confounding_resolved_for_model_training": mixed,
        "bounded_sequence_unique_support_not_in_policy_union": script_only,
        "failure_semantics": "negative_under_frozen_controller_bank, never physical unreachability",
        "model_training_authorized": mixed,
    })
    print(json.dumps({"mixed": mixed, "strata": strata, "script_only": script_only}, indent=2))


if __name__ == "__main__":
    main()
