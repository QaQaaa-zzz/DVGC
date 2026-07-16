"""Freeze a C_D matcher using construction evidence only, before audit."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from cli.build_descent_entries import snapshot_identity
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config
from dvgc.descent_entry import DESCENT_ENTRY_FEATURE_NAMES
from dvgc.entry import calibrate_radius, robust_normalization
from dvgc.runtime import save_json


def build_matcher(bank: SnapshotBank, cfg):
    raw = bank.records_for_phase("flight", include_training_only=False)
    unique = {}
    for row in raw:
        unique.setdefault(snapshot_identity(row), row)
    rows = list(unique.values())
    safe = [row for row in rows if row["final"]["label"] == "safe"]
    parents = {str(row.get("entry_source_id", row.get("parent_candidate_id", row["id"]))) for row in safe}
    if len(safe) < int(cfg.tube_activation_min_safe):
        raise ValueError(f"Insufficient unique C_D safe entries: {len(safe)}")
    if len(parents) < 2:
        raise ValueError("C_D safe support lacks parent/source diversity")
    center, scale = robust_normalization([row["entry_feature"] for row in safe], cfg.descent_entry_scale_floors)
    labels = [row["final"]["label"] for row in rows]
    calibration = calibrate_radius(
        [row["entry_feature"] for row in rows],
        labels,
        center,
        scale,
        float(cfg.descent_entry_minimum_calibration_precision),
    )
    matcher = {
        "version": "descent_entry_task_relative_v1",
        "feature_names": DESCENT_ENTRY_FEATURE_NAMES,
        "center": center.tolist(),
        "scale": scale.tolist(),
        "scale_floors": list(cfg.descent_entry_scale_floors),
        "radius": calibration["radius"],
        "construction_calibration_precision": calibration["precision"],
        "construction_calibration_recall": calibration["recall"],
        "frozen_before_independent_audit": True,
    }
    result = SnapshotBank(copy.deepcopy(rows), copy.deepcopy(bank.metadata))
    result.metadata.update(
        {
            "entry_bank_role": "frozen_pre_audit_descent_handoff_set",
            "entry_set_version": bank.metadata["last_tube_version"],
            "entry_matcher": matcher,
            "deduplicated_from_records": len(raw),
            "safe_parent_sources": sorted(parents),
        }
    )
    return result, {"unique_states": len(rows), "safe_entries": len(safe), "safe_parent_count": len(parents), "matcher": matcher}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--certified-bank", required=True)
    p.add_argument("--output-bank", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--config", default="configs/default.json")
    a = p.parse_args()
    output, manifest = Path(a.output_bank), Path(a.manifest)
    if output.exists() or manifest.exists():
        raise SystemExit("Frozen matcher output already exists")
    source = SnapshotBank.load(a.certified_bank)
    try:
        result, report = build_matcher(source, load_config(a.config))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    result.metadata["certified_bank_sha256"] = file_sha256(a.certified_bank)
    result.save(output)
    report.update(
        {
            "status": "PASS",
            "artifact_role": "immutable_pre_audit_descent_matcher",
            "certified_bank_sha256": file_sha256(a.certified_bank),
            "bank_sha256": file_sha256(output),
        }
    )
    save_json(manifest, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
