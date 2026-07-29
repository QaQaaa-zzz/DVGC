"""Train a construction-only Descent state-safety probability network.

The network is an acquisition guide.  It predicts branch recoverability for
physically valid Descent snapshots, but its predictions never certify a state
or change the frozen matcher/Tube contracts.  Only construction labels are
accepted; independent-audit paths are rejected explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from cli.train_descent_reachability_kernel_v2 import (
    CONSTRUCTION,
    INDEX,
    PILOT_ROOT,
    average_precision,
    load_rows,
    robust_scale,
)
from dvgc.config import file_sha256
from dvgc.runtime import save_json


SOURCE_A_ROOT = Path("runs/descent_diverse_p1_predecessor_recovery_v1")
DEFAULT_RUN = Path("runs/descent_reachability_network_v3/construction_parent_cv")
SAFE_THRESHOLD = 0.75


def final_branch_success(row: dict) -> bool:
    """Final-Recovery label, intentionally independent of Chain entry."""
    return bool(
        row.get("final_recovery")
        and not row.get("illegal_contact")
        and not row.get("penetration")
        and not row.get("phase_violation")
        and not row.get("timeout")
        and not row.get("nonfinite")
    )


def safety_class(target: float) -> str:
    if target >= SAFE_THRESHOLD:
        return "safe"
    if target > 0.0:
        return "boundary"
    return "dead"


def _forbid_audit_path(path: Path) -> None:
    if "independent_audit" in str(path).lower():
        raise ValueError(f"independent audit input is forbidden: {path}")


def load_source_a_rows(root: Path) -> tuple[list[dict], list[dict], dict]:
    """Load Source-A construction labels and the still-unlabelled pool."""
    snapshots_path = root / "trajectory_harvested_snapshots.pkl"
    labels_path = root / "source_a_certification_results.json"
    prior_path = Path("runs/backward_recovery_tube_fast_track_v1/descent_cem3_tier2/descent_cem_pilot_report.json")
    for path in (root, snapshots_path, labels_path, prior_path):
        _forbid_audit_path(path)
    snapshots = pickle.loads(snapshots_path.read_bytes())
    labels = json.loads(labels_path.read_text())
    if labels.get("heldout_used") is not False or labels.get("status") != "PASS":
        raise ValueError("Source-A is not an eligible construction-label artifact")
    by_hash = {row["physical_state_hash"]: row for row in snapshots}
    if len(by_hash) != len(snapshots):
        raise ValueError("duplicate Source-A physical state hashes")

    prior_nodes = json.loads(prior_path.read_text())["nodes"]
    source_nodes = {row["node_id"]: row for row in prior_nodes + labels.get("nodes", [])}
    rows = []
    labelled_hashes = set()
    for label in labels["rows"]:
        state_hash = label["physical_state_hash"]
        if state_hash not in by_hash:
            raise ValueError(f"Source-A label has no snapshot: {state_hash}")
        snapshot = by_hash[state_hash]
        if snapshot["candidate_id"] != label["candidate_id"]:
            raise ValueError(f"Source-A candidate mismatch: {state_hash}")
        branch_rows = label.get("branches", [])
        branches = len(branch_rows)
        successes = sum(final_branch_success(row) for row in branch_rows)
        if branches <= 0 or not 0 <= successes <= branches:
            raise ValueError(f"invalid Source-A branch label: {state_hash}")
        record = snapshot["snapshot_v4"]
        rows.append({
            "key": f"source_a:{state_hash}",
            "state_hash": state_hash,
            "parent": label["candidate_id"],
            "target": successes / branches,
            "successes": successes,
            "branches": branches,
            "physical": np.asarray(record["physical_feature"], float)[:16],
            "history": np.asarray(record["actor_observation_t"], float),
            "source": str(labels_path),
            "region": label["region"],
            "layer": int(label["layer"]),
        })
        labelled_hashes.add(state_hash)

    pool = []
    for index, snapshot in enumerate(snapshots):
        node = source_nodes.get(snapshot["source_node_id"])
        if node is None:
            raise ValueError(f"Source-A source node missing: {snapshot['source_node_id']}")
        record = snapshot["snapshot_v4"]
        pool.append({
            "proposal_id": f"source_a:{snapshot['physical_state_hash']}",
            "physical_state_hash": snapshot["physical_state_hash"],
            "candidate_id": snapshot["candidate_id"],
            "source_node_id": snapshot["source_node_id"],
            "source_index": index,
            "source_artifact": str(snapshots_path),
            "region": node["region"],
            "shell_layer": int(node["layer"]),
            "relative_tick_to_downstream_entry": int(snapshot["relative_tick_to_downstream_entry"]),
            "physical": np.asarray(record["physical_feature"], float)[:16],
            "history": np.asarray(record["actor_observation_t"], float),
            "already_labelled": snapshot["physical_state_hash"] in labelled_hashes,
        })
    identity = {
        "snapshots": {"path": str(snapshots_path), "sha256": file_sha256(snapshots_path)},
        "labels": {"path": str(labels_path), "sha256": file_sha256(labels_path)},
        "prior_nodes": {"path": str(prior_path), "sha256": file_sha256(prior_path)},
    }
    return rows, pool, identity


def load_ranked_pilot_rows(path: Path) -> list[dict]:
    _forbid_audit_path(path)
    report = json.loads(path.read_text())
    if report.get("artifact_role") != "reachability_ranked_construction_certification_pilot":
        raise ValueError(f"ineligible ranked pilot role: {path}")
    rows = []
    bank_cache: dict[str, list] = {}
    for label in report["rows"]:
        proposal = label["proposal"]
        artifact = proposal["source_artifact"]
        _forbid_audit_path(Path(artifact))
        if artifact not in bank_cache:
            bank_cache[artifact] = pickle.loads(Path(artifact).read_bytes())
        source = bank_cache[artifact][int(proposal["source_index"])]
        if source["physical_state_hash"] != proposal["physical_state_hash"]:
            raise ValueError(f"ranked pilot physical identity mismatch: {path}")
        branches = label["branches"]
        successes = sum(final_branch_success(row) for row in branches)
        record = source["snapshot_v4"]
        rows.append({
            "key": f"ranked_pilot:{proposal['physical_state_hash']}",
            "state_hash": proposal["physical_state_hash"],
            "parent": proposal["candidate_id"],
            "target": successes / len(branches) if branches else 0.0,
            "successes": successes,
            "branches": len(branches),
            "physical": np.asarray(record["physical_feature"], float)[:16],
            "history": np.asarray(record["actor_observation_t"], float),
            "source": str(path),
            "region": proposal["region"],
            "layer": int(proposal["shell_layer"]),
        })
    return rows


def apply_final_semantics_to_old_pilots(rows: list[dict], pilot_root: Path) -> list[dict]:
    """Replace legacy Chain+Final P1 counts with Final-only branch outcomes."""
    by_key = {row["key"]: dict(row) for row in rows}
    for directory in sorted(pilot_root.glob("pilot_12x_p1*")):
        if "independent_audit" in str(directory).lower():
            raise ValueError("independent audit input is forbidden")
        certification_path = directory / "certification.json"
        if not certification_path.exists():
            continue
        certification = json.loads(certification_path.read_text())
        for label in certification["rows"]:
            key = f"pilot:{label['node_id']}"
            if key not in by_key:
                continue
            branches = label.get("micro_branches", [])
            if not branches:
                by_key[key]["target"] = 0.0
                by_key[key]["successes"] = 0
                by_key[key]["branches"] = 0
                continue
            successes = sum(final_branch_success(row) for row in branches)
            by_key[key]["target"] = successes / len(branches)
            by_key[key]["successes"] = successes
            by_key[key]["branches"] = len(branches)
    return [by_key[row["key"]] for row in rows]


def normalize_old_rows(rows: list[dict]) -> list[dict]:
    normalized = []
    for row in rows:
        target = float(row["target"])
        normalized.append({
            **row,
            "state_hash": row["key"],
            "successes": None,
            "branches": None,
            "class": safety_class(target),
        })
    return normalized


def merge_rows(old_rows: list[dict], new_rows: list[dict]) -> list[dict]:
    """Globally deduplicate state identities and reject conflicting labels."""
    merged: dict[str, dict] = {}
    for row in normalize_old_rows(old_rows) + new_rows:
        row = dict(row)
        row["class"] = safety_class(float(row["target"]))
        key = str(row["state_hash"])
        if key in merged:
            previous = merged[key]
            if previous["parent"] != row["parent"] or not np.isclose(previous["target"], row["target"]):
                raise ValueError(f"conflicting duplicate construction label: {key}")
            continue
        merged[key] = row
    return list(merged.values())


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def fit_network(features: np.ndarray, target: np.ndarray, *, seed: int, steps: int = 1600) -> dict:
    """Fit a small deterministic 16-12-1 tanh network with Adam."""
    features = np.asarray(features, np.float64)
    target = np.asarray(target, np.float64)
    center, scale = robust_scale(features)
    x = (features - center) / scale
    rng = np.random.default_rng(seed)
    params = {
        "w1": rng.normal(0.0, 0.08, (x.shape[1], 12)),
        "b1": np.zeros(12),
        "w2": rng.normal(0.0, 0.08, (12, 1)),
        "b2": np.zeros(1),
    }
    first = {key: np.zeros_like(value) for key, value in params.items()}
    second = {key: np.zeros_like(value) for key, value in params.items()}
    lr, l2 = 0.01, 0.003
    for step in range(1, steps + 1):
        hidden = np.tanh(x @ params["w1"] + params["b1"])
        prediction = sigmoid(hidden @ params["w2"] + params["b2"]).reshape(-1)
        delta2 = (prediction - target)[:, None] / len(target)
        gradients = {
            "w2": hidden.T @ delta2 + l2 * params["w2"],
            "b2": delta2.sum(axis=0),
        }
        delta1 = (delta2 @ params["w2"].T) * (1.0 - hidden * hidden)
        gradients["w1"] = x.T @ delta1 + l2 * params["w1"]
        gradients["b1"] = delta1.sum(axis=0)
        for key in params:
            first[key] = 0.9 * first[key] + 0.1 * gradients[key]
            second[key] = 0.999 * second[key] + 0.001 * gradients[key] ** 2
            corrected_first = first[key] / (1.0 - 0.9 ** step)
            corrected_second = second[key] / (1.0 - 0.999 ** step)
            params[key] -= lr * corrected_first / (np.sqrt(corrected_second) + 1e-8)
    return {"center": center, "scale": scale, **params}


def predict_network(model: dict, features: np.ndarray) -> np.ndarray:
    x = (np.atleast_2d(np.asarray(features, float)) - model["center"]) / model["scale"]
    hidden = np.tanh(x @ model["w1"] + model["b1"])
    return sigmoid(hidden @ model["w2"] + model["b2"]).reshape(-1)


def parent_cv(features: np.ndarray, target: np.ndarray, parents: np.ndarray, *, seed: int) -> np.ndarray:
    prediction = np.zeros(len(target), float)
    for fold, parent in enumerate(sorted(set(parents))):
        test = parents == parent
        train = ~test
        if not train.any():
            raise ValueError("parent CV requires at least two parents")
        model = fit_network(features[train], target[train], seed=seed + fold)
        prediction[test] = predict_network(model, features[test])
    return prediction


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    target = np.asarray(target, float)
    prediction = np.asarray(prediction, float)
    bins = np.minimum((prediction * 10).astype(int), 9)
    ece = sum(
        float((bins == index).mean()) * abs(float(prediction[bins == index].mean() - target[bins == index].mean()))
        for index in range(10) if np.any(bins == index)
    )
    safe = target >= SAFE_THRESHOLD
    order = np.argsort(-prediction, kind="stable")
    ranked = safe[order]
    safe_count = int(ranked.sum())
    safe_ap = 0.0 if not safe_count else float(sum(
        bool(value) * ranked[: rank + 1].mean() for rank, value in enumerate(ranked)
    ) / safe_count)
    return {
        "soft_brier": float(np.mean((prediction - target) ** 2)),
        "ece_10": float(ece),
        "auprc_any_success": average_precision(target, prediction),
        "average_precision_safe": safe_ap,
        "mean_target": float(target.mean()),
        "mean_prediction": float(prediction.mean()),
    }


def select_pilot(ranked: list[dict], count: int = 4) -> list[dict]:
    selected, parents = [], set()
    for row in ranked:
        if row["candidate_id"] in parents:
            continue
        selected.append(row)
        parents.add(row["candidate_id"])
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"insufficient unseen parents for pilot: {len(selected)}/{count}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=str(DEFAULT_RUN))
    parser.add_argument("--construction", default=str(CONSTRUCTION))
    parser.add_argument("--pilot-root", default=str(PILOT_ROOT))
    parser.add_argument("--index", default=str(INDEX))
    parser.add_argument("--source-a-root", default=str(SOURCE_A_ROOT))
    parser.add_argument("--additional-pilot", action="append", default=[])
    parser.add_argument("--seed", type=int, default=9_860_000)
    args = parser.parse_args()
    root = Path(args.run)
    if root.exists():
        raise SystemExit(f"refusing overwrite {root}")
    paths = [Path(args.construction), Path(args.pilot_root), Path(args.index), Path(args.source_a_root)]
    for path in paths:
        _forbid_audit_path(path)

    old = apply_final_semantics_to_old_pilots(load_rows(paths[0], paths[1], paths[2]), paths[1])
    additional, pool, source_a_identity = load_source_a_rows(paths[3])
    pilot_rows = [row for pilot in args.additional_pilot for row in load_ranked_pilot_rows(Path(pilot))]
    rows = merge_rows(old, additional + pilot_rows)
    features = np.asarray([row["physical"] for row in rows], float)
    target = np.asarray([row["target"] for row in rows], float)
    parents = np.asarray([row["parent"] for row in rows])
    if features.shape[1] != 16:
        raise ValueError(f"expected physical 16D features, got {features.shape}")

    cv_prediction = parent_cv(features, target, parents, seed=args.seed)
    prior = np.asarray([target[parents != parent].mean() for parent in parents])
    cv_metrics, prior_metrics = metrics(target, cv_prediction), metrics(target, prior)
    improvement = (prior_metrics["soft_brier"] - cv_metrics["soft_brier"]) / prior_metrics["soft_brier"]
    positive_parents = len({row["parent"] for row in rows if row["target"] > 0})
    status = "PASS" if (
        positive_parents >= 6
        and improvement >= 0.05
        and cv_metrics["auprc_any_success"] >= prior_metrics["auprc_any_success"] + 0.10
    ) else "FAIL"

    model = fit_network(features, target, seed=args.seed)
    labelled_hashes = {row["state_hash"] for row in rows}
    labelled_parents = set(parents)
    eligible = [row for row in pool if (
        not row["already_labelled"]
        and row["physical_state_hash"] not in labelled_hashes
        and row["candidate_id"] not in labelled_parents
    )]
    scores = predict_network(model, np.asarray([row["physical"] for row in eligible], float))
    ranked = []
    for source, score in zip(eligible, scores, strict=True):
        ranked.append({key: source[key] for key in (
            "proposal_id", "physical_state_hash", "candidate_id", "source_node_id",
            "source_index", "source_artifact", "region", "shell_layer",
            "relative_tick_to_downstream_entry",
        )} | {"predicted_p_safe": float(score), "ranking_only": True})
    ranked.sort(key=lambda row: (-row["predicted_p_safe"], row["candidate_id"], row["proposal_id"]))
    available_parents = len({row["candidate_id"] for row in ranked})
    selected = select_pilot(ranked, min(4, available_parents)) if status == "PASS" and available_parents else []

    root.mkdir(parents=True)
    np.savez(root / "model.npz", **model)
    model_hash = file_sha256(root / "model.npz")
    dataset_manifest = {
        "status": "PASS",
        "artifact_role": "construction_only_descent_reachability_labels",
        "independent_audit_labels_used": False,
        "formal_tube_labels": False,
        "state_count": len(rows),
        "unique_parents": len(set(parents)),
        "class_counts": dict(Counter(row["class"] for row in rows)),
        "sources": dict(Counter(row["source"] for row in rows)),
        "rows": [{
            "state_hash": row["state_hash"], "parent": row["parent"],
            "target": row["target"], "class": row["class"], "source": row["source"],
        } for row in rows],
        "source_a_identity": source_a_identity,
        "additional_pilots": [
            {"path": str(path), "sha256": file_sha256(path)} for path in map(Path, args.additional_pilot)
        ],
    }
    save_json(root / "dataset_manifest.json", dataset_manifest)
    save_json(root / "ranked_proposals.json", {
        "status": status,
        "artifact_role": "proposal_only_reachability_ranking",
        "formal_tube_or_matcher": False,
        "independent_audit_labels_used": False,
        "model_sha256": model_hash,
        "selected_pilot": selected,
        "all_ranked": ranked,
    })
    report = {
        "status": status,
        "artifact_role": "phase_conditioned_reachability_probability_network",
        "formal_tube_or_matcher": False,
        "network": "deterministic_numpy_mlp_16_12_1",
        "cpu_only": True,
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "states": len(rows), "unique_parents": len(set(parents)),
        "positive_parents": positive_parents,
        "class_counts": dict(Counter(row["class"] for row in rows)),
        "target_definition": "legal frozen-policy Final-Recovery branch success fraction; Chain reported separately",
        "safe_threshold": SAFE_THRESHOLD,
        "split": "leave-one-trajectory-candidate-out",
        "parent_prior": prior_metrics,
        "parent_cv": cv_metrics,
        "relative_brier_improvement_over_parent_prior": float(improvement),
        "eligible_unseen_states": len(ranked),
        "eligible_unseen_parents": len({row["candidate_id"] for row in ranked}),
        "selected_pilot": len(selected),
        "selected_regions": dict(Counter(row["region"] for row in selected)),
        "model_sha256": model_hash,
        "dataset_manifest_sha256": file_sha256(root / "dataset_manifest.json"),
        "PPO_authorization": False,
        "next": "branch_certify_ranked_pilot" if status == "PASS" else "reachability_network_insufficient",
    }
    save_json(root / "DESCENT_REACHABILITY_NETWORK_V3_REPORT.json", report)
    save_json(root / "completed.json", {"status": status, "next": report["next"]})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
