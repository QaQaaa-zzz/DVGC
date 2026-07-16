"""Freeze exact block-1 empirical label sets and the failed global matcher baseline."""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np

from dvgc.bank import LABELS, SnapshotBank
from dvgc.config import file_sha256, load_config
from dvgc.descent_entry import DESCENT_ENTRY_FEATURE_NAMES
from dvgc.discrete_tube import snapshot_identity
from dvgc.entry import normalized_nearest, robust_normalization
from dvgc.runtime import save_json


def branch_summary(rows):
    evidence = [ev for row in rows for ev in row.get("certification_branches", [])]
    total = len(evidence)
    reasons = Counter(ev.get("end_reason", ev.get("terminal_cause", "unknown")) for ev in evidence)
    return {
        "states": len(rows), "branches": total,
        "final_successes": sum(bool(ev.get("final_recovery")) for ev in evidence),
        "final_rate": sum(bool(ev.get("final_recovery")) for ev in evidence) / total if total else 0.0,
        "physical_failures": sum(ev.get("terminal_cause") == "physical_failure" for ev in evidence),
        "physical_failure_rate": sum(ev.get("terminal_cause") == "physical_failure" for ev in evidence) / total if total else 0.0,
        "pitch": reasons.get("pitch_limit", 0), "roll": reasons.get("roll_limit", 0),
        "nonfinite": reasons.get("nonfinite", 0),
        "timeout": sum(ev.get("terminal_cause") == "timeout" for ev in evidence),
        "horizon": sum(ev.get("terminal_cause") == "horizon_exhausted" for ev in evidence),
        "termination_reasons": dict(reasons),
    }


def failed_global_radius(rows, safe, cfg):
    center, scale = robust_normalization([row["entry_feature"] for row in safe], cfg.descent_entry_scale_floors)
    positive = [normalized_nearest(row["entry_feature"], [q["entry_feature"] for q in safe], center, scale, exclude=i)[0]
                for i, row in enumerate(safe)]
    negatives = [row for row in rows if row["final"]["label"] != "safe"]
    negative = [normalized_nearest(row["entry_feature"], [q["entry_feature"] for q in safe], center, scale) for row in negatives]
    candidates = sorted(set(positive + [item[0] for item in negative]))
    scored = []
    for radius in candidates:
        tp = sum(distance <= radius for distance in positive)
        fp = sum(item[0] <= radius for item in negative)
        fn = len(positive) - tp
        tn = len(negative) - fp
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / len(positive) if positive else 0.0
        scored.append((precision, recall, -radius, radius, tp, fp, fn, tn))
    best = max(scored)
    radius, tp, fp, fn, tn = best[3:]
    false_safe = []
    for row, (distance, nearest, contribution) in zip(negatives, negative):
        if distance <= radius:
            ranked = sorted(zip(DESCENT_ENTRY_FEATURE_NAMES, contribution.tolist()), key=lambda item: item[1], reverse=True)
            false_safe.append({"id": row["id"], "snapshot_sha256": snapshot_identity(row),
                               "parent": row.get("entry_source_id", row.get("parent_candidate_id")),
                               "label": row["final"]["label"], "distance": distance,
                               "nearest_safe_id": safe[nearest]["id"], "dimension_contributions": ranked})
    return {
        "status": "FAIL", "matcher_type": "global_isotropic_radius_after_robust_axis_scaling",
        "feature_names": DESCENT_ENTRY_FEATURE_NAMES, "center": center.tolist(), "scale": scale.tolist(),
        "selection": "maximize leave-one-out recall subject to construction precision >= 0.95",
        "minimum_precision": float(cfg.descent_entry_minimum_calibration_precision),
        "best_available": {"radius": radius, "precision": best[0], "recall": best[1],
                           "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}},
        "positive_leave_one_out_distances": positive, "false_safe": false_safe,
        "continuous_matcher_active": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certified-bank", required=True)
    parser.add_argument("--cert-report", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()
    out = Path(args.output_dir)
    if out.exists():
        raise SystemExit(f"Output exists: {out}")
    out.mkdir(parents=True)
    bank = SnapshotBank.load(args.certified_bank)
    unique = {}
    for row in bank.records_for_phase("flight", include_training_only=False):
        unique.setdefault(snapshot_identity(row), row)
    rows = list(unique.values())
    policy_hash = file_sha256(Path(args.policy) / "params.pkl")
    certification_hash = file_sha256(args.cert_report)
    assets = {}
    all_unique = SnapshotBank(copy.deepcopy(rows), {**copy.deepcopy(bank.metadata),
                              "discrete_set_role": "D_all_unique",
                              "membership_type": "exact_snapshot_identity",
                              "policy_hash": policy_hash, "certification_hash": certification_hash,
                              "continuous_matcher_active": False})
    all_path = out / "D_all_unique.pkl"
    all_unique.save(all_path)
    assets["all"] = {"role": "D_all_unique", "path": str(all_path.resolve()),
                     "sha256": file_sha256(all_path), "states": len(rows)}
    for label in LABELS:
        selected = [copy.deepcopy(row) for row in rows if row["final"]["label"] == label]
        role = "D_emp_safe" if label == "safe" else f"D_{label}"
        subset = SnapshotBank(selected, {**copy.deepcopy(bank.metadata), "discrete_set_role": role,
                                        "membership_type": "exact_snapshot_identity",
                                        "policy_hash": policy_hash, "certification_hash": certification_hash,
                                        "continuous_matcher_active": False})
        path = out / f"{role}.pkl"
        subset.save(path)
        assets[label] = {"role": role, "path": str(path.resolve()), "sha256": file_sha256(path), "states": len(selected)}
    safe = [row for row in rows if row["final"]["label"] == "safe"]
    members = [{"id": row["id"], "snapshot_sha256": snapshot_identity(row),
                "parent": row.get("entry_source_id", row.get("parent_candidate_id")),
                "layer": row.get("descent_layer")} for row in safe]
    manifest = {"status": "PASS", "membership_type": "exact_snapshot_identity", "policy_hash": policy_hash,
                "certification_hash": certification_hash, "certified_bank_sha256": file_sha256(args.certified_bank),
                "members": members, "sets": assets, "continuous_matcher_active": False,
                "network_predictions_are_members": False}
    save_json(out / "discrete_tube_manifest.json", manifest)
    baseline = failed_global_radius(rows, safe, load_config(args.config))
    save_json(out / "failed_global_matcher_calibration.json", baseline)
    report = {
        "status": "PASS", "records": len(bank.records_for_phase("flight", include_training_only=False)),
        "byte_state_unique": len(rows), "labels": dict(Counter(row["final"]["label"] for row in rows)),
        "by_label": {label: branch_summary([row for row in rows if row["final"]["label"] == label]) for label in LABELS},
        "safe_parent_count": len({str(row.get("entry_source_id", row.get("parent_candidate_id"))) for row in safe}),
        "safe_layers": dict(Counter(row.get("descent_layer", "unknown") for row in safe)),
        "safe_states": [{"id": row["id"], "snapshot_sha256": snapshot_identity(row),
                         "parent": row.get("entry_source_id", row.get("parent_candidate_id")),
                         "layer": row.get("descent_layer"), "final_successes": row["final"]["successes"],
                         "branches": row["final"]["branches"], "posterior": row["final"]["posterior"]} for row in safe],
        "policy_hash": policy_hash, "certification_hash": certification_hash,
        "assets": assets, "failed_global_matcher": baseline,
    }
    save_json(out / "block1_exact_certification_report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("safe_states", "failed_global_matcher")}, indent=2))


if __name__ == "__main__":
    main()
