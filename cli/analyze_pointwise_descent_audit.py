"""Analyze independent exact-state C_D audit without distance generalization."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from dvgc.bank import SnapshotBank, beta_posterior, posterior_label
from dvgc.certification import assert_disjoint_branch_seeds
from dvgc.config import file_sha256, load_config
from dvgc.discrete_tube import ExactTubeMembership, snapshot_identity
from dvgc.runtime import save_json


def calibration_metrics(prediction, observation, bins=5):
    p, y = np.asarray(prediction, np.float64), np.asarray(observation, np.float64)
    brier = float(np.mean((p-y)**2))
    ece = 0.0
    edges = np.linspace(0, 1, bins+1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p <= hi if hi == 1 else p < hi)
        if mask.any(): ece += float(mask.mean()) * abs(float(p[mask].mean()-y[mask].mean()))
    return {"brier": brier, "ece": float(ece), "bins": bins}


def audit_label(successes, branches, cfg):
    posterior = beta_posterior(successes, branches-successes, alpha0=cfg.beta_alpha0,
                              beta0=cfg.beta_beta0, q_low=cfg.posterior_q_low,
                              q_high=cfg.posterior_q_high)
    label = posterior_label(posterior, branches, min_branches=cfg.min_branches,
                            safe_threshold=cfg.safe_threshold, dead_threshold=cfg.dead_threshold,
                            boundary_max_width=cfg.boundary_max_width)
    return label, posterior


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-bank", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--construction-report", required=True)
    parser.add_argument("--audit-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists(): raise SystemExit(f"Output exists: {out}")
    cfg = load_config(args.config)
    bank = SnapshotBank.load(args.candidate_bank)
    manifest_data = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    membership = ExactTubeMembership.from_manifest(manifest_data)
    construction = json.loads(Path(args.construction_report).read_text(encoding="utf-8"))
    audit = json.loads(Path(args.audit_report).read_text(encoding="utf-8"))
    if audit["candidate_bank_sha256"] != file_sha256(args.candidate_bank):
        raise SystemExit("Pointwise audit candidate bank hash mismatch")
    construction_evidence = [ev for row in construction["rows"] for ev in row["branch_evidence"]]
    audit_evidence = [ev for row in audit["rows"] for ev in row["branch_evidence"]]
    assert_disjoint_branch_seeds(construction_evidence, [ev["branch_seed"] for ev in audit_evidence])
    rows = bank.records_for_phase("flight", include_training_only=False)
    by_id = {row["id"]: row for row in rows}
    if set(by_id) != {row["id"] for row in audit["rows"]}:
        raise SystemExit("Pointwise audit ids do not exactly cover D_all_unique")
    evaluated = []
    for result in audit["rows"]:
        record = by_id[result["id"]]
        label, posterior = audit_label(int(result["final"]), int(result["branches"]), cfg)
        predicted = membership.contains(record, policy_hash=manifest_data["policy_hash"],
                                        certification_hash=manifest_data["certification_hash"])
        evaluated.append({"id": record["id"], "snapshot_sha256": snapshot_identity(record),
                          "construction_label": record["final"]["label"], "construction_probability": record["final"]["posterior"]["mean"],
                          "parent": record.get("entry_source_id", record.get("parent_candidate_id")),
                          "layer": record.get("descent_layer"), "pointwise_member": predicted,
                          "audit_successes": int(result["final"]), "audit_branches": int(result["branches"]),
                          "audit_rate": float(result["final_rate"]), "audit_label": label,
                          "audit_posterior": posterior})
    truth = [row["audit_label"] == "safe" for row in evaluated]
    pred = [row["pointwise_member"] for row in evaluated]
    tp = sum(p and t for p,t in zip(pred,truth)); fp = sum(p and not t for p,t in zip(pred,truth))
    fn = sum(not p and t for p,t in zip(pred,truth)); tn = len(pred)-tp-fp-fn
    precision = tp/(tp+fp) if tp+fp else 1.0; recall = tp/(tp+fn) if tp+fn else 0.0
    member_rows = [row for row in evaluated if row["pointwise_member"]]
    audit_rates = [row["audit_rate"] for row in evaluated]
    prediction = [row["construction_probability"] for row in evaluated]
    branches = int(audit["terminal_summary"]["branches"])
    physical = int(audit["terminal_summary"]["physical_failures"])
    report = {
        "status": "PASS" if precision >= .95 and len(member_rows) >= 4 and len({r['parent'] for r in member_rows}) >= 2 else "FAIL",
        "gate": {"minimum_precision": .95, "minimum_unique_safe": 4, "minimum_parents": 2},
        "pointwise": {"precision": precision, "recall": recall, "candidate_mass_coverage": sum(pred)/len(pred),
                      "confusion": {"tp":tp,"fp":fp,"fn":fn,"tn":tn}},
        "raw_rate_threshold_precision": sum(row["audit_rate"] >= cfg.safe_threshold for row in member_rows)/len(member_rows),
        "aggregate_final_rate": sum(row["audit_successes"] for row in evaluated)/sum(row["audit_branches"] for row in evaluated),
        "calibration": calibration_metrics(prediction, audit_rates),
        "physical_failure_rate": physical/branches if branches else 0.0,
        "timeout_rate": audit["terminal_summary"]["timeouts"]/branches if branches else 0.0,
        "horizon_rate": audit["terminal_summary"]["horizon_exhaustions"]/branches if branches else 0.0,
        "audit_seed": audit["seed"], "audit_seed_namespace": audit["seed_namespace"],
        "construction_seed_namespace": construction["seed_namespace"], "seed_isolation_pass": True,
        "member_results": member_rows, "audit_label_distribution": dict(Counter(row["audit_label"] for row in evaluated)),
        "rows": evaluated,
    }
    save_json(out, report)
    print(json.dumps({k:v for k,v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__": main()
