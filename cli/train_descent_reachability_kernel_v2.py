"""Construction-only nonlinear Descent reachability guide with parent CV.

This model ranks new acquisition proposals.  It is deliberately incapable of
defining C_D or certifying Tube membership, and it never reads independent
audit results.
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from cli.train_descent_history_reachability_v1 import load_proposals
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256
from dvgc.runtime import save_json


CONSTRUCTION = Path("runs/descent_compact_matcher_neighborhood_v1/construction_24_adaptive/construction_bank.pkl")
INDEX = Path("runs/backward_recovery_tube_fast_track_v1/proposal_state_index.json")
PILOT_ROOT = Path("runs/descent_natural_bridge_candidates_v1")
DEFAULT_RUN = Path("runs/descent_reachability_kernel_v2/construction_parent_cv_feature_selected")
H_GRID = (0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0)


def record_dedup_id(record: dict) -> str:
    return str(record.get("state_byte_hash", record["id"]))


def average_precision(target: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(target) > 0
    order = np.argsort(-np.asarray(prediction), kind="stable")
    ranked = truth[order]
    positives = int(ranked.sum())
    if not positives:
        return 0.0
    return float(sum(bool(value) * ranked[: rank + 1].mean() for rank, value in enumerate(ranked)) / positives)


def robust_scale(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(features, axis=0)
    scale = np.maximum(np.median(np.abs(features - center), axis=0) * 1.4826, 0.01)
    return center, scale


def kernel_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, bandwidth: float) -> np.ndarray:
    center, scale = robust_scale(train_x)
    train = (train_x - center) / scale
    test = (np.atleast_2d(test_x) - center) / scale
    distance = np.sqrt(np.mean((test[:, None, :] - train[None, :, :]) ** 2, axis=2))
    weight = np.exp(-0.5 * (distance / bandwidth) ** 2)
    prior = float(np.mean(train_y))
    # One prior pseudo-observation makes isolated points conservative.
    return (weight @ train_y + prior) / (weight.sum(axis=1) + 1.0)


def grouped_cv(features: np.ndarray, target: np.ndarray, parents: np.ndarray) -> tuple[np.ndarray, list[float]]:
    prediction = np.zeros(len(target), dtype=float)
    selected_bandwidths: list[float] = []
    for outer_parent in sorted(set(parents)):
        test = np.flatnonzero(parents == outer_parent)
        train = np.flatnonzero(parents != outer_parent)
        scores = []
        for bandwidth in H_GRID:
            inner_prediction = np.zeros(len(train), dtype=float)
            for inner_parent in sorted(set(parents[train])):
                validation_local = np.flatnonzero(parents[train] == inner_parent)
                fitting_local = np.flatnonzero(parents[train] != inner_parent)
                inner_prediction[validation_local] = kernel_predict(
                    features[train[fitting_local]], target[train[fitting_local]],
                    features[train[validation_local]], bandwidth,
                )
            scores.append((float(np.mean((inner_prediction - target[train]) ** 2)), bandwidth))
        bandwidth = min(scores)[1]
        selected_bandwidths.append(bandwidth)
        prediction[test] = kernel_predict(features[train], target[train], features[test], bandwidth)
    return prediction, selected_bandwidths


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    target = np.asarray(target, float)
    prediction = np.asarray(prediction, float)
    bins = np.minimum((prediction * 10).astype(int), 9)
    ece = 0.0
    for index in range(10):
        mask = bins == index
        if mask.any():
            ece += float(mask.mean()) * abs(float(prediction[mask].mean() - target[mask].mean()))
    return {
        "soft_brier": float(np.mean((prediction - target) ** 2)),
        "ece_10": float(ece),
        "auprc_any_success": average_precision(target, prediction),
        "mean_target": float(target.mean()),
        "mean_prediction": float(prediction.mean()),
    }


def load_rows(construction: Path, pilot_root: Path, index: Path) -> list[dict]:
    rows = []
    seen_keys = set()
    for record in SnapshotBank.load(construction).records:
        parent = record["entry_source_id"]
        rows.append({
            "key": f"construction:{record['id']}", "parent": parent,
            "target": record["final"]["successes"] / record["final"]["branches"],
            "physical": np.asarray(record["entry_feature"], float),
            "history": np.asarray(record["policy_state"]["actor_observation"], float),
            "source": str(construction),
        })
        # Older construction records may predate the explicit byte hash.  The
        # immutable bank-local record id remains a deterministic dedup key.
        seen_keys.add((parent, record_dedup_id(record)))

    index_rows = json.loads(index.read_text())["rows"]
    by_proposal = {row["proposal_id"]: row for row in index_rows}
    record_cache: dict[str, dict] = {}
    for directory in sorted(pilot_root.glob("pilot_12x_p1*")):
        if "independent_audit" in str(directory):
            raise ValueError("independent audit input is forbidden")
        manifest_path = directory / "manifest.json"
        certification_path = directory / "certification.json"
        if not manifest_path.exists() or not certification_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        certification = json.loads(certification_path.read_text())
        by_node = {row["node_id"]: row for row in certification["rows"]}
        for selected in manifest["rows"]:
            proposal = selected["proposal_id"]
            if proposal not in by_proposal or proposal not in by_node:
                raise ValueError(f"incomplete construction label for {proposal}")
            source = by_proposal[proposal]
            if proposal not in record_cache:
                record_cache[proposal] = load_proposals([source])[0]
            record = record_cache[proposal]
            state_hash = source["physical_state_sha256"]
            dedup_key = (selected["candidate_id"], state_hash)
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            p1 = by_node[proposal]["P1"]
            target = p1["successes"] / p1["branches"] if p1["branches"] else 0.0
            rows.append({
                "key": f"pilot:{proposal}", "parent": selected["candidate_id"], "target": target,
                "physical": np.asarray(record["physical_feature"], float)[:16],
                "history": np.asarray(record["policy_state"]["actor_observation"], float),
                "source": str(directory),
            })
    return rows


def choose_full_bandwidth(features: np.ndarray, target: np.ndarray, parents: np.ndarray) -> float:
    prediction, bandwidths = grouped_cv(features, target, parents)
    del prediction
    counts = Counter(bandwidths)
    return min(H_GRID, key=lambda value: (-counts[value], value))


def select_ranked(
    rows: list[dict], excluded_parents: set[str], per_region: int = 4, total: int | None = None,
) -> list[dict]:
    total = total if total is not None else per_region * 3
    selected = []
    used = set(excluded_parents)
    for region in ("early", "middle", "late"):
        candidates = sorted(
            (row for row in rows if row["region"] == region and row["candidate_id"] not in used),
            key=lambda row: (-row["reachability_score"], row["candidate_id"], row["proposal_id"]),
        )
        for row in candidates:
            if row["candidate_id"] in used:
                continue
            selected.append(row)
            used.add(row["candidate_id"])
            if sum(item["region"] == region for item in selected) == per_region:
                break
    # Preserve every region that remains available, then redistribute a
    # depleted stratum's quota without ever reusing a labelled parent.
    remaining = sorted(
        (row for row in rows if row["candidate_id"] not in used),
        key=lambda row: (-row["reachability_score"], row["region"], row["candidate_id"], row["proposal_id"]),
    )
    for row in remaining:
        if len(selected) == total:
            break
        if row["candidate_id"] in used:
            continue
        selected.append(row)
        used.add(row["candidate_id"])
    if len(selected) != total:
        raise ValueError(f"insufficient unseen-parent coverage: {len(selected)}/{total}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=str(DEFAULT_RUN))
    parser.add_argument("--construction", default=str(CONSTRUCTION))
    parser.add_argument("--pilot-root", default=str(PILOT_ROOT))
    parser.add_argument("--index", default=str(INDEX))
    args = parser.parse_args()
    root = Path(args.run)
    if root.exists():
        raise SystemExit(f"refusing overwrite {root}")
    construction, pilot_root, index = Path(args.construction), Path(args.pilot_root), Path(args.index)
    rows = load_rows(construction, pilot_root, index)
    target = np.asarray([row["target"] for row in rows], float)
    parents = np.asarray([row["parent"] for row in rows])
    physical = np.asarray([row["physical"] for row in rows], float)
    history = np.asarray([row["history"] for row in rows], float)
    if physical.shape[1] != 16 or history.shape[1] != 140:
        raise ValueError(f"unexpected feature shapes {physical.shape}, {history.shape}")

    prior_prediction = np.asarray([target[parents != parent].mean() for parent in parents])
    physical_prediction, physical_bandwidths = grouped_cv(physical, target, parents)
    history_prediction, history_bandwidths = grouped_cv(history, target, parents)
    prior_metrics = metrics(target, prior_prediction)
    physical_metrics = metrics(target, physical_prediction)
    history_metrics = metrics(target, history_prediction)
    panels = {
        "physical_16D": (physical, physical_metrics, physical_bandwidths),
        "actor_history_140D": (history, history_metrics, history_bandwidths),
    }
    chosen_name = min(
        panels,
        key=lambda name: (panels[name][1]["soft_brier"], -panels[name][1]["auprc_any_success"], name),
    )
    chosen_features, chosen_metrics, chosen_bandwidths = panels[chosen_name]
    improvement = (prior_metrics["soft_brier"] - chosen_metrics["soft_brier"]) / prior_metrics["soft_brier"]
    positive_parents = len({row["parent"] for row in rows if row["target"] > 0})
    status = "PASS" if (
        positive_parents >= 6 and improvement >= 0.05
        and chosen_metrics["auprc_any_success"] >= prior_metrics["auprc_any_success"] + 0.10
    ) else "FAIL"

    full_bandwidth = choose_full_bandwidth(chosen_features, target, parents)
    index_rows = json.loads(index.read_text())["rows"]
    proposal_records = load_proposals(index_rows)
    proposal_features = (
        np.asarray([record["physical_feature"] for record in proposal_records], float)
        if chosen_name == "physical_16D" else
        np.asarray([record["policy_state"]["actor_observation"] for record in proposal_records], float)
    )
    scores = kernel_predict(chosen_features, target, proposal_features, full_bandwidth)
    excluded_parents = set(parents)
    ranked = []
    for source, score in zip(index_rows, scores, strict=True):
        if source["candidate_id"] in excluded_parents:
            continue
        ranked.append({
            key: source[key] for key in (
                "proposal_id", "candidate_id", "region", "shell_layer",
                "physical_state_sha256", "source_artifact", "source_index",
            )
        } | {"reachability_score": float(score)})
    selected = select_ranked(ranked, excluded_parents) if status == "PASS" else []

    root.mkdir(parents=True)
    np.savez(
        root / "model.npz", training_feature=chosen_features, training_target=target,
        bandwidth=np.asarray(full_bandwidth), parents=parents, feature_name=np.asarray(chosen_name),
    )
    model_hash = file_sha256(root / "model.npz")
    save_json(root / "ranked_proposals.json", {
        "status": status, "artifact_role": "proposal_only_reachability_ranking",
        "formal_tube_or_matcher": False, "independent_audit_labels_used": False,
        "model_sha256": model_hash, "selected": selected,
        "all_ranked": sorted(ranked, key=lambda row: -row["reachability_score"]),
    })
    report = {
        "status": status, "artifact_role": "phase_conditioned_reachability_proposal_guide",
        "formal_tube_or_matcher": False, "cpu_only": True,
        "states": len(rows), "unique_parents": len(set(parents)),
        "positive_parents": positive_parents, "sources": dict(Counter(row["source"] for row in rows)),
        "target": {"min": float(target.min()), "median": float(np.median(target)), "max": float(target.max())},
        "parent_prior": prior_metrics,
        "physical_16D_kernel": physical_metrics | {"outer_bandwidths": dict(Counter(map(str, physical_bandwidths)))},
        "actor_history_140D_kernel": history_metrics | {"outer_bandwidths": dict(Counter(map(str, history_bandwidths)))},
        "relative_brier_improvement_over_parent_prior": float(improvement),
        "feature_selection_rule": "lowest parent-held-out soft Brier; AUPRC tie-break",
        "selected_feature": chosen_name,
        "selected_feature_metrics": chosen_metrics | {"outer_bandwidths": dict(Counter(map(str, chosen_bandwidths)))},
        "full_model_bandwidth": full_bandwidth, "selected_proposals": len(selected),
        "selected_regions": {region: sum(row["region"] == region for row in selected) for region in ("early", "middle", "late")},
        "construction_bank_sha256": file_sha256(construction), "proposal_index_sha256": file_sha256(index),
        "model_sha256": model_hash, "PPO_authorization": False,
        "next": "certify_ranked_unseen_parent_pilot" if status == "PASS" else "nonlinear_estimator_insufficient",
    }
    save_json(root / "DESCENT_REACHABILITY_KERNEL_V2_REPORT.json", report)
    save_json(root / "completed.json", {"status": status, "next": report["next"]})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
