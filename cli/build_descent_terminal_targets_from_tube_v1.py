"""Build provenance-locked proposal targets from a certified Descent Tube."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256
from dvgc.runtime import save_json


FEATURES = ("x", "z", "roll", "pitch", "vx", "vz", "wx", "wy", "wz", "hip", "knee")
INDEX = {"x": 0, "z": 2, "roll": 3, "pitch": 4, "vx": 6, "vz": 8,
         "wx": 9, "wy": 10, "wz": 11, "hip": 13, "knee": 14}
FLOORS = np.asarray((0.05, 0.03, 0.03, 0.03, 0.1, 0.1, 0.2, 0.2, 0.2, 0.04, 0.04), float)


def terminal_matrix(records: list[dict]) -> np.ndarray:
    matrix = np.asarray([[row["physical_feature"][INDEX[name]] for name in FEATURES] for row in records], float)
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURES) or not np.isfinite(matrix).all():
        raise ValueError("invalid terminal target features")
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tube", required=True)
    parser.add_argument("--output-bank", required=True)
    parser.add_argument("--output-report", required=True)
    args = parser.parse_args()
    output_bank, output_report = Path(args.output_bank), Path(args.output_report)
    if output_bank.exists() or output_report.exists():
        raise SystemExit("refusing overwrite terminal target artifact")
    tube = SnapshotBank.load(args.tube)
    if tube.metadata.get("artifact_role") != "certified_tube":
        raise SystemExit("source is not a certified Tube")
    if not tube.records or any(not row.get("certified_safe") or not row.get("safe_claim_allowed") for row in tube.records):
        raise SystemExit("source Tube includes non-certified records")
    raw = terminal_matrix(tube.records)
    center = np.median(raw, axis=0)
    scale = np.maximum(np.percentile(raw, 75, axis=0) - np.percentile(raw, 25, axis=0), FLOORS)
    records = []
    for row in tube.records:
        item = copy.deepcopy(row)
        item.update({"candidate_kind": "certified_tube_terminal_proposal_target",
                     "safe_claim_allowed": False, "tube_metrics_eligible": False})
        records.append(item)
    metadata = {
        "artifact_role": "proposal_terminal_targets_from_certified_tube",
        "certified_tube": False, "safe_claim_allowed": False,
        "source_tube_sha256": file_sha256(args.tube), "source_tube_states": len(records),
        "source_policy_version": tube.metadata.get("last_policy_version"),
        "cluster_features": list(FEATURES), "normalization_center": center.tolist(),
        "normalization_scale": scale.tolist(),
    }
    output_bank.parent.mkdir(parents=True, exist_ok=True)
    SnapshotBank(records, metadata).save(output_bank)
    report = {
        "status": "PASS", "artifact_role": metadata["artifact_role"],
        "source_tube": str(Path(args.tube)), "source_tube_sha256": metadata["source_tube_sha256"],
        "states": len(records), "features": list(FEATURES),
        "normalization_center": center.tolist(), "normalization_scale": scale.tolist(),
        "output_bank": str(output_bank), "output_bank_sha256": file_sha256(output_bank),
        "not_a_tube_or_matcher": True,
    }
    save_json(output_report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
