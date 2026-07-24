"""Cluster current-runtime Final-safe/boundary Descent terminal proposals."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256
from dvgc.runtime import save_json


FEATURES = (
    "x", "z", "roll", "pitch", "vx", "vz", "wx", "wy", "wz", "hip", "knee",
)
INDEX = {"x": 0, "z": 2, "roll": 3, "pitch": 4, "vx": 6, "vz": 8,
         "wx": 9, "wy": 10, "wz": 11, "hip": 13, "knee": 14}


def _deterministic_kmeans(x, count, iterations=40):
    centers = [x[0]]
    while len(centers) < count:
        distance = np.min(
            [np.sum(np.square(x - center), axis=1) for center in centers], axis=0
        )
        centers.append(x[int(np.argmax(distance))])
    centers = np.asarray(centers, float)
    labels = np.zeros(len(x), int)
    for _ in range(iterations):
        updated = np.argmin(
            np.sum(np.square(x[:, None, :] - centers[None, :, :]), axis=2), axis=1
        )
        if np.array_equal(updated, labels):
            break
        labels = updated
        for cluster in range(count):
            if np.any(labels == cluster):
                centers[cluster] = np.mean(x[labels == cluster], axis=0)
    return labels, centers


def _ranges(values):
    return {
        "min": float(np.min(values)), "p50": float(np.median(values)),
        "max": float(np.max(values)),
    }


def _physical_shock(branch):
    return bool(
        branch["five_step_reset_shock_failure"]
        and branch["physical_failure"]
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--support-bank", required=True)
    p.add_argument("--runtime-audit", required=True)
    p.add_argument("--apex-bank", required=True)
    p.add_argument("--output-bank", required=True)
    p.add_argument("--output-report", required=True)
    p.add_argument("--clusters", type=int, default=4)
    a = p.parse_args()

    support = SnapshotBank.load(a.support_bank)
    apex = SnapshotBank.load(a.apex_bank)
    audit = json.loads(Path(a.runtime_audit).read_text())
    audit_rows = {row["candidate_id"]: row for row in audit["rows"]}
    selected = []
    for record in support.records:
        row = audit_rows[record["id"]]
        if row["replay_label"] not in ("final_safe_replay", "boundary_replay"):
            continue
        branches = row["branches"]
        # Exact restore validity is a state property.  A noisy controller branch
        # failing during the first five steps is reported, but must not erase a
        # Final-safe state.  Boundary targets additionally need majority local
        # continuation and less than half reset-shock failures.
        if row["replay_label"] == "boundary_replay" and (
            np.mean([b["descent_controller_success"] for b in branches]) < .5
            or np.mean([_physical_shock(b) for b in branches]) >= .5
        ):
            continue
        selected.append((record, row))
    if not selected:
        raise RuntimeError("no current-runtime positive Descent terminal proposals")
    raw = np.asarray([
        [record["physical_feature"][INDEX[name]] for name in FEATURES]
        for record, _ in selected
    ], float)
    center = np.median(raw, axis=0)
    scale = np.percentile(raw, 75, axis=0) - np.percentile(raw, 25, axis=0)
    scale = np.maximum(scale, np.asarray(
        [.05, .03, .03, .03, .1, .1, .2, .2, .2, .04, .04], float
    ))
    normalized = (raw - center) / scale
    cluster_count = min(int(a.clusters), len(selected))
    labels, centers = _deterministic_kmeans(normalized, cluster_count)
    output_records = []
    clusters = []
    for cluster in range(cluster_count):
        indices = np.where(labels == cluster)[0]
        rows = [selected[i] for i in indices]
        branches = [branch for _, row in rows for branch in row["branches"]]
        values = raw[indices]
        known_parents = {
            str(row["parent"]) for _, row in rows if row["parent"] is not None
        }
        clusters.append({
            "cluster": cluster, "states": len(rows),
            "runtime_labels": dict(Counter(row["replay_label"] for _, row in rows)),
            "regions": dict(Counter(str(row["region"]) for _, row in rows)),
            "independent_parent_count": len(known_parents),
            "states_with_unknown_parent": sum(
                row["parent"] is None for _, row in rows
            ),
            "parents": sorted(known_parents),
            "reset_shock_failure_rate": float(np.mean([
                _physical_shock(branch) for branch in branches
            ])),
            "descent_controller_success_rate": float(np.mean([
                branch["descent_controller_success"] for branch in branches
            ])),
            "landing_final_recovery_rate": float(np.mean([
                branch["final_landing_recovery"] for branch in branches
            ])),
            "feature_ranges": {
                name: _ranges(values[:, fi]) for fi, name in enumerate(FEATURES)
            },
            "source_paths": sorted({
                provenance["source_path"] for _, row in rows
                for provenance in (row.get("source_provenance") or [])
            }),
        })
        for i in indices:
            record = dict(selected[i][0])
            record.update({
                "candidate_kind": "current_runtime_descent_terminal_proposal",
                "descent_terminal_cluster": cluster,
                "runtime_replay_label": selected[i][1]["replay_label"],
                "runtime_audit_candidate_id": selected[i][1]["candidate_id"],
            })
            output_records.append(record)

    apex_rows = [
        row for row in apex.records
        if row.get("candidate_kind") == "apex_dynamically_reached"
    ]
    apex_raw = np.asarray([
        [row["physical_feature"][INDEX[name]] for name in FEATURES]
        for row in apex_rows
    ], float)
    apex_norm = (apex_raw - center) / scale
    distances = np.sqrt(np.sum(
        np.square(apex_norm[:, None, :] - centers[None, :, :]), axis=2
    ))
    nearest = np.argmin(distances, axis=1)
    bridge_alignment = []
    for cluster in range(cluster_count):
        values = distances[:, cluster]
        bridge_alignment.append({
            "cluster": cluster,
            "minimum_normalized_distance": float(np.min(values)),
            "p50_normalized_distance": float(np.median(values)),
            "closest_apex_candidate": apex_rows[int(np.argmin(values))]["id"],
            "apex_states_nearest_to_cluster": int(np.sum(nearest == cluster)),
        })
    preferred_pool = [
        row for row in bridge_alignment if clusters[row["cluster"]]["states"] >= 3
    ] or bridge_alignment
    preferred = min(preferred_pool, key=lambda row: (
        -clusters[row["cluster"]]["landing_final_recovery_rate"],
        row["minimum_normalized_distance"],
    ))["cluster"]
    SnapshotBank(output_records, {
        "artifact_role": "current_runtime_descent_terminal_proposals",
        "certified_tube": False, "safe_claim_allowed": False,
        "source_support_bank_sha256": file_sha256(a.support_bank),
        "runtime_audit_sha256": file_sha256(a.runtime_audit),
        "cluster_features": list(FEATURES),
        "normalization_center": center.tolist(),
        "normalization_scale": scale.tolist(),
    }).save(a.output_bank)
    payload = {
        "status": "PASS",
        "artifact_role": "current_runtime_descent_terminal_cluster_diagnostic",
        "not_a_certified_tube": True,
        "support_bank_sha256": file_sha256(a.support_bank),
        "runtime_audit_sha256": file_sha256(a.runtime_audit),
        "apex_bank_sha256": file_sha256(a.apex_bank),
        "selected_states": len(output_records),
        "excluded_dead_unknown": len(support.records) - len(output_records),
        "runtime_label_counts": dict(Counter(
            row["runtime_replay_label"] for row in output_records
        )),
        "features": list(FEATURES),
        "normalization_center": center.tolist(),
        "normalization_scale": scale.tolist(),
        "clusters": clusters,
        "apex_to_terminal_cluster_alignment": bridge_alignment,
        "preferred_bridge_terminal_cluster": preferred,
        "output_bank": str(Path(a.output_bank).resolve()),
        "output_bank_sha256": file_sha256(a.output_bank),
        "formal_matcher_unchanged": True,
    }
    save_json(a.output_report, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
