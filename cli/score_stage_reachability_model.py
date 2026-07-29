"""Score an unseen state bank with a construction-only reachability model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256
from dvgc.runtime import save_json
from cli.train_stage_reachability_model import parent_key, sigmoid


def nearest_distances(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.min(np.linalg.norm(query[:, None, :] - reference[None, :, :], axis=2), axis=1)


def training_support_radius(features: np.ndarray) -> float:
    if len(features) < 2:
        raise ValueError("at least two training states are required")
    distances = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    return float(max(1.0, np.quantile(np.min(distances, axis=1), .95)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--training-bank", required=True)
    parser.add_argument("--candidate-bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing overwrite {output}")
    model = np.load(args.model)
    center, scale = model["center"], model["scale"]
    weights, bias = model["weights"], float(model["bias"])
    training = SnapshotBank.load(args.training_bank)
    candidates = SnapshotBank.load(args.candidate_bank)
    train_x = np.asarray([row["physical_feature"] for row in training.records], float)
    candidate_x = np.asarray([row["physical_feature"] for row in candidates.records], float)
    train_z, candidate_z = (train_x-center)/scale, (candidate_x-center)/scale
    radius = training_support_radius(train_z)
    distances = nearest_distances(candidate_z, train_z)
    predictions = sigmoid(candidate_z @ weights + bias)
    seen_parents = {parent_key(row) for row in training.records}
    records = []
    for row, prediction, distance in zip(candidates.records, predictions, distances):
        parent = parent_key(row)
        unseen = parent not in seen_parents
        within = bool(distance <= radius)
        records.append({
            "candidate_id": row["id"], "parent": parent,
            "predicted_p_next": float(prediction),
            "normalized_training_distance": float(distance),
            "training_support_radius_p95": radius,
            "unseen_parent": unseen, "within_training_support": within,
            "acquisition_eligible": bool(unseen and within),
            "ranking_only": True,
        })
    payload = {
        "status": "PASS", "artifact_role": "stage_reachability_ranked_proposals",
        "stage": args.stage, "not_certified_tube": True, "not_safe_labels": True,
        "model_sha256": file_sha256(args.model),
        "training_bank_sha256": file_sha256(args.training_bank),
        "candidate_bank_sha256": file_sha256(args.candidate_bank),
        "training_support_radius_p95": radius,
        "states": len(records), "eligible_states": sum(row["acquisition_eligible"] for row in records),
        "eligible_parents": len({row["parent"] for row in records if row["acquisition_eligible"]}),
        "records": sorted(records, key=lambda row: (-row["acquisition_eligible"],
                                                      -row["predicted_p_next"], row["candidate_id"])),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, payload)
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
