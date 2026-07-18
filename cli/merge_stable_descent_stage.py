"""Strictly merge selected-index stable-construction shards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cli.certify_stable_descent_shard import indices_hash
from dvgc.certification import branch_seed, detailed_terminal_summary
from dvgc.runtime import save_json


COMMON = (
    "stage", "seed", "seed_namespace", "candidate_bank_sha256",
    "candidate_source_policy_hash", "candidate_source_policy_hashes",
    "descent_policy_hash", "descent_policy_version",
    "landing_policy_hash", "landing_policy_version", "landing_entry_set_sha256",
    "xml_sha256", "config_hash", "runtime_source_fingerprint", "protocol",
    "certification_protocol_version", "construction_seed_epoch",
    "branch_horizon", "branches_per_state", "total_states", "selected_states",
    "selected_indices_sha256",
)


def merge_stage(shards, expected_indices):
    if not shards or any(row.get("status") != "PASS" or not row.get("complete") for row in shards):
        raise ValueError("Every stable construction shard must be complete and PASS")
    for key in COMMON:
        if len({json.dumps(row.get(key), sort_keys=True) for row in shards}) != 1:
            raise ValueError(f"Stable construction shard {key} mismatch")
    expected = [int(value) for value in expected_indices]
    if expected != sorted(set(expected)) or shards[0]["selected_indices_sha256"] != indices_hash(expected):
        raise ValueError("Stable construction selected index manifest mismatch")
    rows = sorted((item for shard in shards for item in shard["rows"]), key=lambda item: item["candidate_index"])
    if [int(row["candidate_index"]) for row in rows] != expected:
        raise ValueError("Stable construction candidate coverage is incomplete or duplicated")
    base_seed = int(shards[0]["seed"])
    branches = int(shards[0]["branches_per_state"])
    evidence = []
    for row in rows:
        items = row["branch_evidence"]
        if len(items) != branches:
            raise ValueError("Stable construction branch budget mismatch")
        for branch_index, item in enumerate(items):
            if int(item["branch_seed"]) != branch_seed(base_seed, int(row["candidate_index"]), branch_index):
                raise ValueError("Stable construction branch seed mismatch")
        evidence.extend(items)
    seeds = [(row["seed_namespace"], int(row["branch_seed"])) for row in evidence]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Stable construction branch seeds are not unique")
    report = {key: shards[0][key] for key in COMMON}
    report.update({
        "status": "PASS", "artifact_role": "merged_stable_construction_stage",
        "states": len(rows), "candidate_indices": expected,
        "shards": [{"selection_start": row["selection_start"], "selection_end": row["selection_end"]} for row in shards],
        "terminal_summary": detailed_terminal_summary(evidence), "rows": rows,
    })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", action="append", required=True)
    parser.add_argument("--indices-file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"Output exists: {output}")
    shards = [json.loads(Path(path).read_text()) for path in args.shard]
    indices = json.loads(Path(args.indices_file).read_text())
    try:
        report = merge_stage(shards, indices)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    save_json(output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
