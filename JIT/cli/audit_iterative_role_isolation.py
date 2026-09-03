#!/usr/bin/env python3
"""Audit TRAIN/calibration/acceptance isolation before pi_(k+1) training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jit_dvgc.config import file_sha256
from jit_dvgc.iterative_frontier_protocol import canonical_sha256
from jit_dvgc.soft_tube import load_soft_tube


SCHEMA = "jit_iterative_role_isolation_audit_v1"


def read(path: Path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def role(root: Path, expected: str):
    root = Path(root)
    manifest = read(root / "role_manifest.json")
    labels = read(root / "logical_labels.json")
    if manifest.get("status") != "completed" or manifest.get("role") != expected:
        raise ValueError(f"{expected} role manifest drift")
    base = {k: v for k, v in manifest.items() if k != "role_manifest_sha256"}
    if canonical_sha256(base) != manifest.get("role_manifest_sha256"):
        raise ValueError(f"{expected} role manifest hash drift")
    if file_sha256(root / "logical_labels.json") != manifest["logical_labels_file_sha256"]:
        raise ValueError(f"{expected} logical labels file drift")
    label_base = {k: v for k, v in labels.items() if k != "labels_sha256"}
    if canonical_sha256(label_base) != labels.get("labels_sha256"):
        raise ValueError(f"{expected} logical labels hash drift")
    rows = labels.get("entries")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{expected} rows empty")
    if any(r.get("split") != expected or r.get("logical_role") != expected for r in rows):
        raise ValueError(f"{expected} logical split drift")
    return manifest, [dict(r) for r in rows]


def near_pairs(left, right, atol: float):
    if not left or not right:
        return []
    right_obs = np.asarray([r["actor_observation"] for r in right], dtype=np.float32)
    result = []
    for row in left:
        obs = np.asarray(row["actor_observation"], dtype=np.float32)
        matches = np.where(np.all(np.abs(right_obs - obs) <= float(atol), axis=1))[0]
        for index in matches.tolist():
            result.append((str(row["state_sha256"]), str(right[index]["state_sha256"])))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--acceptance-root", type=Path, required=True)
    parser.add_argument("--target-tube", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observation-atol", type=float, default=0.01)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"isolation audit already exists: {args.output}")

    tm, train = role(args.train_root, "train")
    cm, calibration = role(args.calibration_root, "calibration")
    am, acceptance = role(args.acceptance_root, "acceptance")
    for field in ("iteration", "policy_actor_sha256", "policy_payload_sha256", "source_tube_manifest_sha256", "plan_sha256"):
        if not (tm.get(field) == cm.get(field) == am.get(field)):
            raise ValueError(f"role {field} mismatch")

    sets = {
        "train": {str(r["state_sha256"]) for r in train},
        "calibration": {str(r["state_sha256"]) for r in calibration},
        "acceptance": {str(r["state_sha256"]) for r in acceptance},
    }
    exact_tc = sets["train"] & sets["calibration"]
    exact_ta = sets["train"] & sets["acceptance"]
    exact_ca = sets["calibration"] & sets["acceptance"]
    if exact_tc or exact_ta or exact_ca:
        raise ValueError("logical roles contain exact duplicate physical states")

    parent_audit = {}
    for phase in ("upstream", "downstream"):
        groups = {
            name: {str(r["parent_group_id"]) for r in rows if r["phase"] == phase}
            for name, rows in (("train", train), ("calibration", calibration), ("acceptance", acceptance))
        }
        if groups["train"] & groups["calibration"] or groups["train"] & groups["acceptance"] or groups["calibration"] & groups["acceptance"]:
            raise ValueError(f"{phase} logical roles share a parent group")
        parent_audit[phase] = {name: len(value) for name, value in groups.items()}

    atol = float(args.observation_atol)
    if not np.isfinite(atol) or atol <= 0:
        raise ValueError("observation atol must be positive")
    near_tc = near_pairs(calibration, train, atol)
    near_ta = near_pairs(acceptance, train, atol)
    near_ca = near_pairs(acceptance, calibration, atol)
    if near_tc or near_ta or near_ca:
        raise ValueError(
            "logical roles contain near-duplicate actor observations; automatic iteration stops without replacement"
        )

    tube = load_soft_tube(args.target_tube)
    if int(tube.manifest.get("iteration", -1)) != int(tm["iteration"]) + 1:
        raise ValueError("target Tube iteration drift")
    target_states = {str(r["state_sha256"]) for r in tube.entries}
    calibration_overlap = target_states & sets["calibration"]
    acceptance_overlap = target_states & sets["acceptance"]
    if calibration_overlap or acceptance_overlap:
        raise ValueError("calibration/acceptance state entered the target Tube")

    report = {
        "schema": SCHEMA,
        "status": "independent",
        "iteration": int(tm["iteration"]),
        "target_tube_iteration": int(tube.manifest["iteration"]),
        "plan_sha256": tm["plan_sha256"],
        "train_role_manifest_sha256": tm["role_manifest_sha256"],
        "calibration_role_manifest_sha256": cm["role_manifest_sha256"],
        "acceptance_role_manifest_sha256": am["role_manifest_sha256"],
        "target_tube_manifest_sha256": tube.manifest["manifest_sha256"],
        "parent_group_counts": parent_audit,
        "actor_observation_atol": atol,
        "train_calibration_exact_overlap_count": 0,
        "train_acceptance_exact_overlap_count": 0,
        "calibration_acceptance_exact_overlap_count": 0,
        "train_calibration_near_overlap_count": 0,
        "train_acceptance_near_overlap_count": 0,
        "calibration_acceptance_near_overlap_count": 0,
        "calibration_target_tube_overlap_count": 0,
        "acceptance_target_tube_overlap_count": 0,
        "no_replacement_after_exclusion": True,
        "training_transitions": 0,
        "environment_interactions": 0,
        "test_data_used": False,
        "final_evaluation_data_used": False,
    }
    report["audit_sha256"] = canonical_sha256(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
