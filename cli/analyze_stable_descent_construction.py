"""Combine independent construction batches into a stable empirical label bank."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from dvgc.bank import SnapshotBank
from dvgc.certification import detailed_terminal_summary
from dvgc.config import file_sha256, load_config
from dvgc.discrete_tube import snapshot_identity
from dvgc.runtime import save_json
from dvgc.stable_construction import adaptive_indices, outcome, stable_result


def parent_key(row):
    return str(row.get("entry_source_id", row.get("parent_candidate_id", row["id"])))


def load_report(path):
    return json.loads(Path(path).read_text()) if path else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-bank", required=True)
    parser.add_argument("--stage-a", required=True)
    parser.add_argument("--stage-b", required=True)
    parser.add_argument("--adaptive", default="")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output-bank", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()
    if Path(args.output_bank).exists() or Path(args.output_report).exists():
        raise SystemExit("Stable construction output already exists")

    cfg = load_config(args.config)
    bank = SnapshotBank.load(args.candidate_bank)
    records = bank.records_for_phase("flight", include_training_only=False)
    if len({snapshot_identity(row) for row in records}) != len(records):
        raise SystemExit("Stable construction candidate bank contains duplicate snapshots")
    stage_a, stage_b, adaptive = load_report(args.stage_a), load_report(args.stage_b), load_report(args.adaptive)
    reports = [stage_a, stage_b] + ([adaptive] if adaptive else [])
    common_keys = ("candidate_bank_sha256", "candidate_source_policy_hash", "candidate_source_policy_hashes", "descent_policy_hash",
                   "landing_policy_hash", "landing_entry_set_sha256", "xml_sha256", "config_hash",
                   "runtime_source_fingerprint")
    for key in common_keys:
        if len({str(report.get(key)) for report in reports}) != 1:
            raise SystemExit(f"Stable construction {key} mismatch")
    if stage_a["candidate_bank_sha256"] != file_sha256(args.candidate_bank):
        raise SystemExit("Stable construction candidate hash mismatch")
    if stage_a["descent_policy_hash"] != file_sha256(Path(args.policy) / "params.pkl"):
        raise SystemExit("Stable construction policy hash mismatch")

    by_a = {int(row["candidate_index"]): row for row in stage_a["rows"]}
    by_b = {int(row["candidate_index"]): row for row in stage_b["rows"]}
    by_adaptive = {int(row["candidate_index"]): row for row in adaptive["rows"]} if adaptive else {}
    if sorted(by_a) != list(range(len(records))):
        raise SystemExit("Stage A does not cover the complete candidate pool")
    expected_adaptive = adaptive_indices(stage_a["rows"], stage_b["rows"], cfg)
    if sorted(by_adaptive) != expected_adaptive:
        raise SystemExit("Adaptive construction coverage does not match preregistered rule")
    all_evidence = [item for report in reports for row in report["rows"] for item in row["branch_evidence"]]
    seed_keys = [(str(item["seed_namespace"]), int(item["branch_seed"])) for item in all_evidence]
    raw_seeds = [value for _, value in seed_keys]
    if len(seed_keys) != len(set(seed_keys)) or len(raw_seeds) != len(set(raw_seeds)):
        raise SystemExit("Stable construction Stage-A/B/adaptive branch seeds overlap")

    results = []
    for index, record in enumerate(records):
        result = stable_result(by_a[index], by_b.get(index), by_adaptive.get(index), cfg)
        results.append({
            "id": record["id"], "candidate_index": index, "parent": parent_key(record),
            "layer": record.get("descent_layer"), "previous_label": record["final"]["label"],
            **result,
        })

    digest = hashlib.sha256(json.dumps({
        "policy": stage_a["descent_policy_hash"], "candidate": stage_a["candidate_bank_sha256"],
        "seeds": [report["seed"] for report in reports], "protocol": stage_a["protocol"],
    }, sort_keys=True).encode()).hexdigest()
    tube_version = f"stable-descent-{digest[:12]}"
    work = SnapshotBank(copy.deepcopy(bank.records), copy.deepcopy(bank.metadata))
    by_id = {row["id"]: row for row in results}
    for record in work.records_for_phase("flight", include_training_only=False):
        result = by_id[record["id"]]
        chain = outcome(result["branch_evidence"], cfg, key="chain_success")
        final = copy.deepcopy(result["combined"])
        final["label"] = result["label"]
        record.update({
            "chain": chain, "final": final, "empirical_label": result["label"],
            "stable_safe": bool(result["stable_safe"]),
            "stable_batch_labels": {
                "stage_a": result["stage_a"]["label"],
                "stage_b": result["stage_b"]["label"] if result["stage_b"] else None,
                "adaptive": result["adaptive"]["label"] if result["adaptive"] else None,
            },
            "policy_version": stage_a["descent_policy_version"], "tube_version": tube_version,
            "certification_branches": copy.deepcopy(result["branch_evidence"]),
            "connection_flag": bool(result["stable_safe"]),
        })
    work.metadata.update({
        "bank_role": "stable_descent_construction", "policy_hash": stage_a["descent_policy_hash"],
        "snapshot_source_policy_hashes": stage_a["candidate_source_policy_hashes"],
        "stable_construction_policy_hash": stage_a["descent_policy_hash"],
        "stable_construction_stage_seeds": {report["stage"]: report["seed"] for report in reports},
        "stable_construction_protocol": stage_a["protocol"], "last_tube_version": tube_version,
        "runtime_source_fingerprint": stage_a["runtime_source_fingerprint"],
    })
    work.save(args.output_bank)

    stable = [row for row in results if row["stable_safe"]]
    comparable = [row for row in results if row["stage_b"] is not None]
    stage_consistency = sum(row["stage_a"]["label"] == row["stage_b"]["label"] for row in comparable)
    report = {
        "status": "PASS", "artifact_role": "stable_descent_construction",
        "policy_hash": stage_a["descent_policy_hash"], "candidate_bank_sha256": stage_a["candidate_bank_sha256"],
        "output_bank_sha256": file_sha256(args.output_bank), "xml_sha256": stage_a["xml_sha256"],
        "landing_entry_set_sha256": stage_a["landing_entry_set_sha256"],
        "landing_policy_hash": stage_a["landing_policy_hash"],
        "runtime_source_fingerprint": stage_a["runtime_source_fingerprint"],
        "seed_namespaces": [report["seed_namespace"] for report in reports],
        "seed_namespace": "stable_cross_seed_construction",
        "stage_reports": [{"stage": report["stage"], "seed": report["seed"], "sha256": file_sha256(path)}
                          for report, path in zip(reports, [args.stage_a, args.stage_b] + ([args.adaptive] if adaptive else []))],
        "protocol": stage_a["protocol"], "tube_version": tube_version,
        "states": len(results), "labels": dict(Counter(row["label"] for row in results)),
        "stable_safe_states": len(stable), "stable_safe_parents": len({row["parent"] for row in stable}),
        "stable_safe_layers": dict(Counter(row["layer"] for row in stable)),
        "parent_diversity_pass": len({row["parent"] for row in stable}) >= int(cfg.stable_construction_min_safe_parents),
        "activation_support_pass": len(stable) >= int(cfg.stable_construction_min_safe_states),
        "stage_a_b_consistency": stage_consistency / len(comparable) if comparable else 0.0,
        "stage_a_b_compared_states": len(comparable),
        "safe_to_non_safe": sum(row["previous_label"] == "safe" and not row["stable_safe"] for row in results),
        "boundary_unknown_to_stable_safe": sum(row["previous_label"] in {"boundary", "unknown"} and row["stable_safe"] for row in results),
        "adaptive_states": len(by_adaptive), "branch_cost": len(all_evidence),
        "terminal_summary": detailed_terminal_summary(all_evidence),
        "rows": results,
    }
    save_json(args.output_report, report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
