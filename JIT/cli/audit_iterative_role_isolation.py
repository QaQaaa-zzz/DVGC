#!/usr/bin/env python3
"""Audit TRAIN/calibration/acceptance isolation before pi_(k+1) training."""
from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path

import numpy as np

from jit_dvgc.config import file_sha256
from jit_dvgc.iterative_frontier_protocol import canonical_sha256
from jit_dvgc.soft_tube import load_soft_tube


SCHEMA = "jit_iterative_role_isolation_audit_v1"
NEAR_DIAGNOSTIC_SCHEMA = "jit_iterative_role_near_overlap_diagnostic_v1"


def read(path: Path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


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


def near_pairs(left, right, atol: float, *, left_role: str, right_role: str):
    if not left or not right:
        return []
    right_obs = np.asarray([r["actor_observation"] for r in right], dtype=np.float32)
    result = []
    for row in left:
        obs = np.asarray(row["actor_observation"], dtype=np.float32)
        diffs = np.abs(right_obs - obs)
        matches = np.where(np.all(diffs <= float(atol), axis=1))[0]
        for index in matches.tolist():
            other = right[index]
            delta = diffs[index]
            result.append(
                {
                    "left_role": str(left_role),
                    "right_role": str(right_role),
                    "left_state_sha256": str(row["state_sha256"]),
                    "right_state_sha256": str(other["state_sha256"]),
                    "left_phase": str(row["phase"]),
                    "right_phase": str(other["phase"]),
                    "left_parent_group_id": str(row["parent_group_id"]),
                    "right_parent_group_id": str(other["parent_group_id"]),
                    "left_candidate_id": str(row.get("candidate_id", "")),
                    "right_candidate_id": str(other.get("candidate_id", "")),
                    "same_phase": bool(row["phase"] == other["phase"]),
                    "same_parent_group": bool(
                        str(row["parent_group_id"]) == str(other["parent_group_id"])
                    ),
                    "max_abs_diff": float(np.max(delta)),
                    "mean_abs_diff": float(np.mean(delta)),
                    "l2_diff": float(np.linalg.norm(delta)),
                }
            )
    return result


def summarize_near_pairs(pairs, *, example_limit: int = 12):
    phase_pairs = Counter(
        f"{row['left_phase']}->{row['right_phase']}" for row in pairs
    )
    ordered = sorted(
        pairs,
        key=lambda row: (
            float(row["max_abs_diff"]),
            float(row["l2_diff"]),
            str(row["left_state_sha256"]),
            str(row["right_state_sha256"]),
        ),
    )
    return {
        "pair_count": len(pairs),
        "same_phase_pair_count": sum(bool(row["same_phase"]) for row in pairs),
        "cross_phase_pair_count": sum(not bool(row["same_phase"]) for row in pairs),
        "same_parent_group_pair_count": sum(
            bool(row["same_parent_group"]) for row in pairs
        ),
        "phase_pair_counts": dict(sorted(phase_pairs.items())),
        "minimum_max_abs_diff": (
            float(min(row["max_abs_diff"] for row in pairs)) if pairs else None
        ),
        "maximum_max_abs_diff": (
            float(max(row["max_abs_diff"] for row in pairs)) if pairs else None
        ),
        "examples": ordered[: int(example_limit)],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--acceptance-root", type=Path, required=True)
    parser.add_argument("--target-tube", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observation-atol", type=float, default=0.01)
    parser.add_argument(
        "--near-duplicate-diagnostics",
        type=Path,
        help=(
            "optional read-only diagnostic artifact written before the audit stops "
            "on near-observation overlap; this never relaxes the isolation gate"
        ),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"isolation audit already exists: {args.output}")
    if args.near_duplicate_diagnostics is not None and args.near_duplicate_diagnostics.exists():
        raise FileExistsError(
            f"near-duplicate diagnostic already exists: {args.near_duplicate_diagnostics}"
        )

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
    near_tc = near_pairs(
        calibration,
        train,
        atol,
        left_role="calibration",
        right_role="train",
    )
    near_ta = near_pairs(
        acceptance,
        train,
        atol,
        left_role="acceptance",
        right_role="train",
    )
    near_ca = near_pairs(
        acceptance,
        calibration,
        atol,
        left_role="acceptance",
        right_role="calibration",
    )
    if near_tc or near_ta or near_ca:
        summaries = {
            "train_calibration": summarize_near_pairs(near_tc),
            "train_acceptance": summarize_near_pairs(near_ta),
            "calibration_acceptance": summarize_near_pairs(near_ca),
        }
        diagnostic = {
            "schema": NEAR_DIAGNOSTIC_SCHEMA,
            "status": "completed_read_only_failure_diagnostic",
            "iteration": int(tm["iteration"]),
            "plan_sha256": tm["plan_sha256"],
            "train_role_manifest_sha256": tm["role_manifest_sha256"],
            "calibration_role_manifest_sha256": cm["role_manifest_sha256"],
            "acceptance_role_manifest_sha256": am["role_manifest_sha256"],
            "actor_observation_atol": atol,
            "parent_group_counts": parent_audit,
            "exact_overlap_counts": {
                "train_calibration": len(exact_tc),
                "train_acceptance": len(exact_ta),
                "calibration_acceptance": len(exact_ca),
            },
            "near_overlap": summaries,
            "decision": "automatic_iteration_remains_stopped",
            "claim_boundary": {
                "diagnostic_only": True,
                "isolation_gate_relaxed": False,
                "observation_atol_changed": False,
                "rows_removed_or_reassigned": False,
                "new_environment_interactions": False,
                "test_or_final_data_used": False,
            },
            "training_transitions": 0,
            "environment_interactions": 0,
            "test_data_used": False,
            "final_evaluation_data_used": False,
        }
        diagnostic["diagnostic_sha256"] = canonical_sha256(diagnostic)
        if args.near_duplicate_diagnostics is not None:
            write(args.near_duplicate_diagnostics, diagnostic)
        concise = {
            name: {
                "pair_count": value["pair_count"],
                "same_phase_pair_count": value["same_phase_pair_count"],
                "cross_phase_pair_count": value["cross_phase_pair_count"],
                "minimum_max_abs_diff": value["minimum_max_abs_diff"],
            }
            for name, value in summaries.items()
        }
        raise ValueError(
            "logical roles contain near-duplicate actor observations; automatic "
            "iteration stops without replacement. diagnostic="
            + json.dumps(concise, sort_keys=True, allow_nan=False)
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
    write(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
