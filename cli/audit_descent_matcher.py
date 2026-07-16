"""Evaluate an immutable pre-audit C_D matcher on independent branch evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cli.build_descent_entries import snapshot_identity
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config
from dvgc.descent_entry import matcher_audit
from dvgc.runtime import save_json


def calibration_metrics(prediction, observation, bins=5):
    prediction = np.asarray(prediction, np.float64)
    observation = np.asarray(observation, np.float64)
    brier = float(np.mean((prediction - observation) ** 2))
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        mask = (prediction >= lo) & (prediction <= hi if hi == 1 else prediction < hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(prediction[mask].mean() - observation[mask].mean()))
    return {"brier": brier, "ece": float(ece), "bins": int(bins)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matcher-bank", required=True)
    p.add_argument("--matcher-manifest", required=True)
    p.add_argument("--audit-report", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="configs/default.json")
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise SystemExit(f"Output exists: {out}")
    manifest = json.loads(Path(a.matcher_manifest).read_text(encoding="utf-8"))
    if manifest.get("bank_sha256") != file_sha256(a.matcher_bank):
        raise SystemExit("Frozen matcher bank hash mismatch")
    matcher_bank = SnapshotBank.load(a.matcher_bank)
    matcher = matcher_bank.metadata.get("entry_matcher", {})
    if not matcher.get("frozen_before_independent_audit"):
        raise SystemExit("Matcher was not frozen before independent audit")
    audit = json.loads(Path(a.audit_report).read_text(encoding="utf-8"))
    if audit.get("candidate_bank_sha256") != file_sha256(a.matcher_bank):
        raise SystemExit("Independent audit candidate hash mismatch")
    raw = matcher_bank.records_for_phase("flight", include_training_only=False)
    unique = {}
    for row in raw:
        unique.setdefault(snapshot_identity(row), row)
    rows = list(unique.values())
    audit_by = {row["id"]: row for row in audit["rows"]}
    if set(audit_by) != {row["id"] for row in rows}:
        raise SystemExit("Independent audit does not cover every unique matcher state")
    cfg = load_config(a.config)
    audit_rates = np.asarray([float(audit_by[row["id"]]["final_rate"]) for row in rows])
    truth = list(audit_rates >= float(cfg.safe_threshold))
    safe = [row for row in rows if row["final"]["label"] == "safe"]
    metrics = matcher_audit(rows, safe, matcher, truth)
    construction_probability = [float(row["final"]["posterior"]["mean"]) for row in rows]
    calibration = calibration_metrics(construction_probability, audit_rates)
    safe_parents = {str(row.get("entry_source_id", row.get("parent_candidate_id", row["id"]))) for row in safe}
    physical = int(audit["terminal_summary"].get("physical_failures", 0))
    branches = int(audit["terminal_summary"].get("branches", 0))
    timeout = int(audit["terminal_summary"].get("timeouts", 0))
    horizon = int(audit["terminal_summary"].get("horizon_exhaustions", 0))
    reasons = []
    if metrics["precision"] < float(cfg.descent_entry_minimum_calibration_precision):
        reasons.append("matcher precision below gate")
    if len(safe) < int(cfg.tube_activation_min_safe):
        reasons.append("minimum support below gate")
    if len(safe_parents) < 2:
        reasons.append("safe parent diversity below gate")
    report = {
        "status": "PASS" if not reasons else "FAIL",
        "reasons": reasons,
        "matcher_bank_sha256": file_sha256(a.matcher_bank),
        "matcher_manifest_sha256": file_sha256(a.matcher_manifest),
        "audit_report_sha256": file_sha256(a.audit_report),
        "matcher_radius": matcher["radius"],
        "matcher_frozen_before_audit": True,
        "tube_matcher": metrics,
        "recoverable_recall": metrics["recall"],
        "coverage": metrics["coverage"],
        "calibration": calibration,
        "unique_states": len(rows),
        "unique_final_safe": len(safe),
        "safe_parent_count": len(safe_parents),
        "safe_parents": sorted(safe_parents),
        "audit_final_branch_rate": float(audit_rates.mean()),
        "audit_physical_failure_rate": physical / branches if branches else 0.0,
        "audit_timeout_rate": timeout / branches if branches else 0.0,
        "audit_horizon_rate": horizon / branches if branches else 0.0,
    }
    save_json(out, report)
    print(json.dumps(report, indent=2))
    if reasons:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
