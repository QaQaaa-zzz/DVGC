"""Calibrate Final-safe admission on Landing only, split by reference parent."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from dvgc.bank import SnapshotBank
from dvgc.certifier_calibration import PROTOCOL_VERSION, RULES, batch_lcb, protocol_hash
from dvgc.config import file_sha256, load_config
from dvgc.runtime import save_json


def split_name(parent) -> str:
    digest = hashlib.sha256(str(parent).encode()).digest()
    return "development" if digest[0] % 2 == 0 else "validation"


def metrics(selected, audit_by_id, total_states):
    rows = [audit_by_id[x] for x in selected]
    successes = sum(int(row["terminal_summary"]["final_recoveries"]) for row in rows)
    branches = sum(int(row["terminal_summary"]["branches"]) for row in rows)
    all_successes = sum(int(row["terminal_summary"]["final_recoveries"])
                        for row in audit_by_id.values())
    return {
        "safe_states": len(rows),
        "precision": successes / branches if branches else 1.0,
        "recoverable_recall": successes / all_successes if all_successes else 0.0,
        "state_coverage": len(rows) / total_states if total_states else 0.0,
        "audit_successes": successes, "audit_branches": branches,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--landing-bank", required=True)
    p.add_argument("--independent-audit", required=True)
    p.add_argument("--stage-a-cert", required=True)
    p.add_argument("--stage-b-cert", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--minimum-precision", type=float, default=.95)
    a = p.parse_args(); out = Path(a.output)
    if out.exists(): raise SystemExit(f"Output exists: {out}")
    cfg = load_config(a.config)
    bank = SnapshotBank.load(a.landing_bank)
    audit = json.loads(Path(a.independent_audit).read_text())
    stage_a = json.loads(Path(a.stage_a_cert).read_text())
    stage_b = json.loads(Path(a.stage_b_cert).read_text())
    if not stage_a.get("fixed_branches") or not stage_b.get("fixed_branches"):
        raise SystemExit("Calibration requires fixed-branch Stage A and Stage B evidence")
    if stage_a.get("construction_seed") == stage_b.get("construction_seed"):
        raise SystemExit("Stage A and Stage B seeds must be independent")
    stage_rows = []
    for payload in (stage_a, stage_b):
        mapping = {row["id"]: row["branch_evidence"] for row in payload["results"]}
        stage_rows.append(mapping)
    audit_by_id = {row["id"]: row for row in audit["rows"]}
    records = bank.records_for_phase("landing", include_training_only=False)
    if {row["id"] for row in records} != set(audit_by_id):
        raise SystemExit("Landing bank and independent audit state IDs differ")
    partitions = {name: [] for name in ("development", "validation")}
    evidence = {}
    for row in records:
        parent = row.get("parent") or row.get("reference_index") or row["id"]
        partitions[split_name(parent)].append(row["id"])
        if any(row["id"] not in mapping for mapping in stage_rows):
            raise SystemExit(f"Landing state {row['id']} lacks fresh Stage A/B evidence")
        evidence[row["id"]] = stage_rows[0][row["id"]] + stage_rows[1][row["id"]]
    results = {}
    for rule_name, rule in RULES.items():
        results[rule_name] = {}
        for split, ids in partitions.items():
            if rule_name == "stage_conjunction":
                selected = [state_id for state_id in ids if
                            batch_lcb(stage_rows[0][state_id], float(cfg.safe_threshold)) and
                            batch_lcb(stage_rows[1][state_id], float(cfg.safe_threshold))]
            else:
                selected = [state_id for state_id in ids
                            if rule(evidence[state_id], float(cfg.safe_threshold))]
            local_audit = {state_id: audit_by_id[state_id] for state_id in ids}
            results[rule_name][split] = metrics(selected, local_audit, len(ids))
            results[rule_name][split]["selected_ids"] = selected
    eligible = [name for name in RULES if
                results[name]["development"]["precision"] >= a.minimum_precision]
    if not eligible:
        selected_rule = "stage_conjunction"
    else:
        selected_rule = max(eligible, key=lambda name: (
            results[name]["development"]["recoverable_recall"],
            results[name]["development"]["state_coverage"],
            name == "stage_conjunction"))
    validation_pass = results[selected_rule]["validation"]["precision"] >= a.minimum_precision
    if not validation_pass:
        selected_rule = "stage_conjunction"
        validation_pass = results[selected_rule]["validation"]["precision"] >= a.minimum_precision
    payload = {
        "status": "PASS" if validation_pass else "FAIL",
        "artifact_role": "certifier_protocol_calibration",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_hash": protocol_hash(selected_rule, cfg.safe_threshold),
        "selected_rule": selected_rule,
        "selection_uses_descent_results": False,
        "safe_threshold": float(cfg.safe_threshold),
        "minimum_validation_precision": float(a.minimum_precision),
        "split_unit": "reference_index_parent",
        "partition_states": {k: len(v) for k, v in partitions.items()},
        "results": results,
        "landing_bank_sha256": file_sha256(a.landing_bank),
        "independent_audit_sha256": file_sha256(a.independent_audit),
        "stage_a_cert_sha256": file_sha256(a.stage_a_cert),
        "stage_b_cert_sha256": file_sha256(a.stage_b_cert),
    }
    save_json(out, payload); print(json.dumps({k: v for k, v in payload.items()
                                               if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
