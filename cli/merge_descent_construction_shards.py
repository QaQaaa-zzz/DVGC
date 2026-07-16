"""Strictly validate and atomically assemble C_D construction shards."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cli.certify_descent_entries import current_label, label_decided
from dvgc.bank import SnapshotBank
from dvgc.certification import branch_seed, summarize_branches
from dvgc.config import file_sha256, load_config
from dvgc.descent_entry import descent_entry_feature
from dvgc.runtime import save_json


COMMON_KEYS = (
    "seed",
    "seed_namespace",
    "candidate_bank_sha256",
    "candidate_source_policy_hash",
    "landing_entry_set_sha256",
    "descent_policy_hash",
    "descent_policy_version",
    "descent_estimator_version",
    "landing_policy_hash",
    "landing_policy_version",
    "xml_sha256",
    "config_hash",
    "runtime_source_fingerprint",
    "protocol",
    "min_branches",
    "max_branches",
    "branch_horizon",
    "total_states",
    "confirm_safe_to_max",
)


def validate_and_merge(shards, source: SnapshotBank, cfg):
    if not shards or any(s.get("status") != "PASS" or not s.get("complete") for s in shards):
        raise ValueError("Every construction shard must be atomically complete and PASS")
    for key in COMMON_KEYS:
        if len({json.dumps(s.get(key), sort_keys=True) for s in shards}) != 1:
            raise ValueError(f"Construction shard {key} mismatch")
    rows = sorted((row for shard in shards for row in shard["rows"]), key=lambda row: row["candidate_index"])
    total = int(shards[0]["total_states"])
    indices = [int(row["candidate_index"]) for row in rows]
    if indices != list(range(total)):
        raise ValueError(f"Construction indices are not complete and unique: {indices}")
    source_rows = source.records_for_phase("flight", include_training_only=False)
    if len(source_rows) != total:
        raise ValueError("Candidate bank state count changed")
    seed_keys = []
    base_seed = int(shards[0]["seed"])
    min_branches = int(shards[0]["min_branches"])
    max_branches = int(shards[0]["max_branches"])
    confirm = bool(shards[0]["confirm_safe_to_max"])
    for row, expected in zip(rows, source_rows):
        i = int(row["candidate_index"])
        if row["id"] != expected["id"]:
            raise ValueError(f"Candidate id mismatch at global index {i}")
        evidence = row["branch_evidence"]
        n = len(evidence)
        if not (min_branches <= n <= max_branches) or int(row["branches"]) != n:
            raise ValueError(f"Invalid branch budget at global index {i}: {n}")
        if [int(ev["branch_index"]) for ev in evidence] != list(range(n)):
            raise ValueError(f"Non-contiguous branch indices at global index {i}")
        for b, ev in enumerate(evidence):
            if int(ev["branch_seed"]) != branch_seed(base_seed, i, b):
                raise ValueError(f"Branch seed mismatch at global={i}, branch={b}")
            seed_keys.append((str(ev["seed_namespace"]), int(ev["branch_seed"])))
        chain = sum(bool(ev["chain_success"]) for ev in evidence)
        final = sum(bool(ev["final_recovery"]) for ev in evidence)
        if chain != int(row["chain"]) or final != int(row["final"]):
            raise ValueError(f"Branch evidence counts mismatch at global index {i}")
        if n < max_branches:
            if not (label_decided(final, n-final, cfg) and label_decided(chain, n-chain, cfg)):
                raise ValueError(f"Premature sequential stop at global index {i}")
            provisional_safe = current_label(final, n-final, cfg) == "safe" or current_label(chain, n-chain, cfg) == "safe"
            if confirm and provisional_safe:
                raise ValueError(f"Safe state was not confirmed to max at global index {i}")
    if len(seed_keys) != len(set(seed_keys)):
        raise ValueError("Construction branch seeds are not globally unique")
    return rows


def assemble(shards, source: SnapshotBank, cfg):
    rows = validate_and_merge(shards, source, cfg)
    first = shards[0]
    digest_payload = {key: first[key] for key in COMMON_KEYS}
    tube_version = "descent-entry-" + hashlib.sha256(json.dumps(digest_payload, sort_keys=True).encode()).hexdigest()[:12]
    work = SnapshotBank(source.records, source.metadata)
    for record in work.records:
        record["entry_feature"] = descent_entry_feature(record["physical_feature"], cfg).astype("float32")
    work.invalidate_phase("flight", reason=f"C_D sharded certification under {first['descent_policy_version']} -> {first['landing_policy_version']}")
    by_id = {row["id"]: row for row in rows}
    namespace = str(first["seed_namespace"])
    for record in work.records_for_phase("flight", include_training_only=False):
        row = by_id[record["id"]]
        n = int(row["branches"])
        work.update_certification(
            record["id"],
            chain_successes=int(row["chain"]),
            chain_failures=n-int(row["chain"]),
            final_successes=int(row["final"]),
            final_failures=n-int(row["final"]),
            policy_version=str(first["descent_policy_version"]),
            estimator_version=str(first["descent_estimator_version"]),
            tube_version=tube_version,
            protocol=first["protocol"],
            seed_namespace=namespace,
            branch_evidence=row["branch_evidence"],
        )
    work.metadata.update(
        {
            "entry_bank_role": "certified_descent_handoff_candidates",
            "last_policy_version": first["descent_policy_version"],
            "last_tube_version": tube_version,
            "construction_seed": first["seed"],
            "construction_seed_namespace": namespace,
            "landing_policy_version": first["landing_policy_version"],
            "landing_policy_hash": first["landing_policy_hash"],
            "landing_entry_set_sha256": first["landing_entry_set_sha256"],
            "runtime_source_fingerprint": first["runtime_source_fingerprint"],
            "xml_sha256": first["xml_sha256"],
            "construction_shards": [{"start_index": s["start_index"], "end_index": s["end_index"]} for s in shards],
        }
    )
    evidence = [ev for row in rows for ev in row["branch_evidence"]]
    terminal = summarize_branches(evidence)
    reasons = {}
    for ev in evidence:
        reasons[str(ev.get("end_reason", "unknown"))] = reasons.get(str(ev.get("end_reason", "unknown")), 0) + 1
    terminal.update(
        {
            "end_reasons": reasons,
            "nonfinite": reasons.get("nonfinite", 0),
            "pitch_failures": reasons.get("pitch_limit", 0),
            "roll_failures": reasons.get("roll_limit", 0),
        }
    )
    report = {key: first[key] for key in COMMON_KEYS}
    report.update(
        {
            "status": "PASS",
            "artifact_role": "merged_descent_entry_construction",
            "states": len(rows),
            "shards": [{"start_index": s["start_index"], "end_index": s["end_index"]} for s in shards],
            "tube_version": tube_version,
            "terminal_summary": terminal,
            "summary": work.summary(),
            "rows": rows,
        }
    )
    return work, report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shard", action="append", required=True)
    p.add_argument("--candidate-bank", required=True)
    p.add_argument("--output-bank", required=True)
    p.add_argument("--output-report", required=True)
    p.add_argument("--config", default="configs/default.json")
    a = p.parse_args()
    bank_out, report_out = Path(a.output_bank), Path(a.output_report)
    if bank_out.exists() or report_out.exists():
        raise SystemExit("Merged construction output already exists")
    source = SnapshotBank.load(a.candidate_bank)
    shards = [json.loads(Path(path).read_text(encoding="utf-8")) for path in a.shard]
    cfg = load_config(a.config)
    try:
        work, report = assemble(shards, source, cfg)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    work.save(bank_out)
    report["bank_sha256"] = file_sha256(bank_out)
    save_json(report_out, report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
