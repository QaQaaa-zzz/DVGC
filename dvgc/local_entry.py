"""Construction-only calibration for conservative per-anchor entry balls."""
from __future__ import annotations

import numpy as np


def calibrate_local_radii(
    anchor_features,
    rows,
    scale,
    *,
    minimum_safe_per_anchor: int = 4,
    minimum_precision: float = .95,
):
    """Freeze the largest zero-false-positive radius supported per anchor.

    ``rows`` contain an explicit ``anchor_index``, feature and construction
    safe label.  Assignment is never inferred from outcomes.  An anchor is
    inactive when fewer than ``minimum_safe_per_anchor`` positive local
    perturbations fit strictly inside its nearest negative example.
    """
    anchors = np.asarray(anchor_features, np.float64)
    scale = np.asarray(scale, np.float64)
    if anchors.ndim != 2 or scale.shape != (anchors.shape[1],):
        raise ValueError("Local-entry feature/scale shape mismatch")
    radii = np.zeros(len(anchors), np.float64)
    support = []
    for index in range(len(anchors)):
        local = [row for row in rows if int(row["anchor_index"]) == index]
        positive = sorted(
            float(np.linalg.norm((np.asarray(row["feature"], float) - anchors[index]) / scale))
            for row in local if bool(row["safe"])
        )
        negative = [
            float(np.linalg.norm((np.asarray(row["feature"], float) - anchors[index]) / scale))
            for row in local if not bool(row["safe"])
        ]
        nearest_negative = min(negative, default=np.inf)
        admitted = [distance for distance in positive if distance < nearest_negative]
        if len(admitted) >= int(minimum_safe_per_anchor):
            radii[index] = admitted[-1]
        support.append({
            "anchor_index": index,
            "safe_examples": len(positive),
            "admitted_safe_examples": len(admitted),
            "negative_examples": len(negative),
            "nearest_negative": nearest_negative,
            "radius": float(radii[index]),
        })
    active = np.flatnonzero(radii > 0.0)
    predicted = []
    truth = []
    for row in rows:
        if not len(active):
            match = False
        else:
            feature = np.asarray(row["feature"], np.float64)
            distances = np.linalg.norm((anchors[active] - feature) / scale, axis=1)
            match = bool(np.any(distances <= radii[active]))
        predicted.append(match); truth.append(bool(row["safe"]))
    tp = sum(p and t for p, t in zip(predicted, truth))
    fp = sum(p and not t for p, t in zip(predicted, truth))
    fn = sum((not p) and t for p, t in zip(predicted, truth))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "status": "PASS" if len(active) and precision >= minimum_precision else "FAIL",
        "radii": radii.tolist(),
        "active_anchor_indices": active.tolist(),
        "precision": precision,
        "recall": recall,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "anchor_support": support,
    }
