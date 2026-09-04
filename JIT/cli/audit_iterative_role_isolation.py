#!/usr/bin/env python3
"""Audit TRAIN/calibration/acceptance isolation before pi_(k+1) training.

The default contract is strict: exact state overlap, parent-group overlap, or any
near-observation overlap stops the automatic iteration.  After such a failure,
an operator may explicitly authorize an engineering-only continuation from the
preserved diagnostic artifact.  That narrower continuation never rewrites the
strict result and is allowed only when candidate-training isolation remains
clean: TRAIN and ACCEPTANCE must have no exact or near-observation overlap, all
roles must remain parent-disjoint, and CALIBRATION/ACCEPTANCE rows must remain
outside the target Tube.
"""
from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path

import numpy as np

from jit_dvgc.config import file_sha256
from jit_dvgc.iterative_frontier_protocol import (
    canonical_sha256,
    exact_state_disjoint_role_rows,
)
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


def verify_self_hash(payload, field: str) -> None:
    declared = str(payload.get(field, ""))
    base = {k: v for k, v in payload.items() if k != field}
    if len(declared) != 64 or canonical_sha256(base) != declared:
        raise ValueError(f"{field} self-hash drift")


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


def write_disjoint_role_view(
    source_root: Path,
    output_root: Path,
    *,
    role_name: str,
    kept_rows: list[dict],
    excluded_states: list[str],
) -> None:
    """Write a derived logical view; raw acquisition and labels stay immutable."""
    source_root = Path(source_root)
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"disjoint role view already exists: {output_root}")
    source_manifest = read(source_root / "role_manifest.json")
    source_labels = read(source_root / "logical_labels.json")
    output_root.mkdir(parents=True, exist_ok=False)

    logical = {
        **{key: value for key, value in source_labels.items() if key != "labels_sha256"},
        "entries": kept_rows,
        "entry_count": len(kept_rows),
        "derived_from_logical_labels": str(source_root / "logical_labels.json"),
        "exact_state_excluded_count": len(excluded_states),
        "exact_state_partition_outcome_blind": True,
    }
    logical["labels_sha256"] = canonical_sha256(logical)
    logical_path = output_root / "logical_labels.json"
    write(logical_path, logical)

    phase_counts = {}
    for phase in ("upstream", "downstream"):
        rows = [row for row in kept_rows if row["phase"] == phase]
        positives = sum(int(row["label"]) for row in rows)
        phase_counts[phase] = {
            "candidate_count": len(rows),
            "positive_count": positives,
            "negative_count": len(rows) - positives,
            "parent_group_count": len({str(row["parent_group_id"]) for row in rows}),
        }
    manifest = {
        **{
            key: value
            for key, value in source_manifest.items()
            if key != "role_manifest_sha256"
        },
        "logical_labels": str(logical_path),
        "logical_labels_file_sha256": file_sha256(logical_path),
        "phase_counts": phase_counts,
        "source_role_manifest_sha256": source_manifest["role_manifest_sha256"],
        "exact_state_partition": {
            "priority": ["train", "calibration", "acceptance"],
            "role": role_name,
            "outcome_fields_used_for_partition": False,
            "excluded_state_count": len(excluded_states),
            "excluded_state_sha256": sorted(excluded_states),
            "new_environment_interactions": 0,
        },
    }
    manifest["role_manifest_sha256"] = canonical_sha256(manifest)
    write(output_root / "role_manifest.json", manifest)


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


def concise_summaries(summaries):
    return {
        name: {
            "pair_count": value["pair_count"],
            "same_phase_pair_count": value["same_phase_pair_count"],
            "cross_phase_pair_count": value["cross_phase_pair_count"],
            "same_parent_group_pair_count": value["same_parent_group_pair_count"],
            "minimum_max_abs_diff": value["minimum_max_abs_diff"],
        }
        for name, value in summaries.items()
    }


def validate_engineering_override(
    diagnostic_path: Path,
    *,
    tm,
    cm,
    am,
    atol: float,
    summaries,
    exact_counts,
) -> dict:
    diagnostic = read(diagnostic_path)
    if diagnostic.get("schema") != NEAR_DIAGNOSTIC_SCHEMA:
        raise ValueError("engineering override requires a near-overlap diagnostic")
    if diagnostic.get("status") != "completed_read_only_failure_diagnostic":
        raise ValueError("engineering override diagnostic status drift")
    verify_self_hash(diagnostic, "diagnostic_sha256")
    expected_identity = {
        "iteration": int(tm["iteration"]),
        "plan_sha256": tm["plan_sha256"],
        "train_role_manifest_sha256": tm["role_manifest_sha256"],
        "calibration_role_manifest_sha256": cm["role_manifest_sha256"],
        "acceptance_role_manifest_sha256": am["role_manifest_sha256"],
    }
    for field, expected in expected_identity.items():
        if diagnostic.get(field) != expected:
            raise ValueError(f"engineering override diagnostic {field} drift")
    if abs(float(diagnostic.get("actor_observation_atol", -1.0)) - float(atol)) > 1e-12:
        raise ValueError("engineering override observation atol drift")
    if diagnostic.get("exact_overlap_counts") != exact_counts:
        raise ValueError("engineering override exact-overlap evidence drift")

    observed = diagnostic.get("near_overlap")
    if not isinstance(observed, dict):
        raise ValueError("engineering override near-overlap evidence missing")
    for name, summary in summaries.items():
        prior = observed.get(name)
        if not isinstance(prior, dict):
            raise ValueError(f"engineering override missing {name} evidence")
        for field in (
            "pair_count",
            "same_phase_pair_count",
            "cross_phase_pair_count",
            "same_parent_group_pair_count",
        ):
            if int(prior.get(field, -1)) != int(summary[field]):
                raise ValueError(f"engineering override {name} {field} drift")

    # Candidate-training independence is the non-negotiable requirement for this
    # narrower continuation.  pi_(k+1) trains from TRAIN-derived Tube support and
    # is later judged on ACCEPTANCE; therefore those two roles must remain both
    # exactly and geometrically isolated under the declared audit tolerance.
    if exact_counts["train_acceptance"] != 0:
        raise ValueError("engineering override forbids TRAIN/ACCEPTANCE exact overlap")
    if int(summaries["train_acceptance"]["pair_count"]) != 0:
        raise ValueError("engineering override forbids TRAIN/ACCEPTANCE near overlap")
    if any(
        int(summary["same_parent_group_pair_count"]) != 0
        for summary in summaries.values()
    ):
        raise ValueError("engineering override forbids near pairs sharing a parent group")

    claim = diagnostic.get("claim_boundary")
    if not isinstance(claim, dict):
        raise ValueError("engineering override diagnostic claim boundary missing")
    if claim.get("diagnostic_only") is not True:
        raise ValueError("engineering override diagnostic was not read-only")
    if claim.get("isolation_gate_relaxed") is not False:
        raise ValueError("engineering override diagnostic already relaxed the gate")
    if claim.get("new_environment_interactions") is not False:
        raise ValueError("engineering override diagnostic touched environment")
    if claim.get("test_or_final_data_used") is not False:
        raise ValueError("engineering override diagnostic touched TEST/final data")
    return diagnostic


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--acceptance-root", type=Path, required=True)
    parser.add_argument("--target-tube", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observation-atol", type=float, default=0.01)
    parser.add_argument(
        "--exact-state-disjoint-view-root",
        type=Path,
        help=(
            "derive outcome-blind CALIBRATION/ACCEPTANCE logical views using "
            "TRAIN > CALIBRATION > ACCEPTANCE physical-state priority"
        ),
    )
    parser.add_argument(
        "--near-duplicate-diagnostics",
        type=Path,
        help=(
            "optional read-only diagnostic artifact written before the strict audit "
            "stops on near-observation overlap; this never relaxes the isolation gate"
        ),
    )
    parser.add_argument(
        "--engineering-near-overlap-override-from",
        type=Path,
        help=(
            "explicitly continue after a preserved strict near-overlap failure. "
            "Requires the exact diagnostic artifact and zero TRAIN/ACCEPTANCE "
            "exact+near overlap. Produces an engineering-only isolation report, "
            "not a formal all-role geometric-isolation PASS."
        ),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"isolation audit already exists: {args.output}")
    if (
        args.near_duplicate_diagnostics is not None
        and args.near_duplicate_diagnostics.exists()
    ):
        raise FileExistsError(
            f"near-duplicate diagnostic already exists: {args.near_duplicate_diagnostics}"
        )
    if (
        args.near_duplicate_diagnostics is not None
        and args.engineering_near_overlap_override_from is not None
    ):
        raise ValueError("diagnostic creation and engineering override are separate steps")

    tm, train = role(args.train_root, "train")
    cm, calibration = role(args.calibration_root, "calibration")
    am, acceptance = role(args.acceptance_root, "acceptance")
    for field in (
        "iteration",
        "policy_actor_sha256",
        "policy_payload_sha256",
        "source_tube_manifest_sha256",
        "plan_sha256",
    ):
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
        if args.exact_state_disjoint_view_root is None:
            raise ValueError("logical roles contain exact duplicate physical states")
        partitioned, exclusion_counts = exact_state_disjoint_role_rows(
            train=train,
            calibration=calibration,
            acceptance=acceptance,
        )
        view_root = Path(args.exact_state_disjoint_view_root)
        if view_root.exists():
            raise FileExistsError(f"exact-state disjoint view root exists: {view_root}")
        calibration_excluded = sorted(
            sets["calibration"]
            - {str(row["state_sha256"]) for row in partitioned["calibration"]}
        )
        acceptance_excluded = sorted(
            sets["acceptance"]
            - {str(row["state_sha256"]) for row in partitioned["acceptance"]}
        )
        write_disjoint_role_view(
            args.calibration_root,
            view_root / "calibration",
            role_name="calibration",
            kept_rows=partitioned["calibration"],
            excluded_states=calibration_excluded,
        )
        write_disjoint_role_view(
            args.acceptance_root,
            view_root / "acceptance",
            role_name="acceptance",
            kept_rows=partitioned["acceptance"],
            excluded_states=acceptance_excluded,
        )
        cm, calibration = role(view_root / "calibration", "calibration")
        am, acceptance = role(view_root / "acceptance", "acceptance")
        sets = {
            "train": {str(r["state_sha256"]) for r in train},
            "calibration": {str(r["state_sha256"]) for r in calibration},
            "acceptance": {str(r["state_sha256"]) for r in acceptance},
        }
        exact_tc = sets["train"] & sets["calibration"]
        exact_ta = sets["train"] & sets["acceptance"]
        exact_ca = sets["calibration"] & sets["acceptance"]
        if exact_tc or exact_ta or exact_ca:
            raise ValueError("derived logical roles remain exact-state overlapping")
        write(
            view_root / "summary.json",
            {
                "schema": "jit_exact_state_disjoint_role_views_v1",
                "status": "completed",
                "priority": ["train", "calibration", "acceptance"],
                "outcome_fields_used_for_partition": False,
                "source_train_root": str(args.train_root),
                "source_calibration_root": str(args.calibration_root),
                "source_acceptance_root": str(args.acceptance_root),
                "effective_train_root": str(args.train_root),
                "effective_calibration_root": str(view_root / "calibration"),
                "effective_acceptance_root": str(view_root / "acceptance"),
                "exclusion_counts": exclusion_counts,
                "training_transitions": 0,
                "environment_interactions": 0,
                "test_data_used": False,
            },
        )
    exact_counts = {
        "train_calibration": len(exact_tc),
        "train_acceptance": len(exact_ta),
        "calibration_acceptance": len(exact_ca),
    }

    parent_audit = {}
    for phase in ("upstream", "downstream"):
        groups = {
            name: {str(r["parent_group_id"]) for r in rows if r["phase"] == phase}
            for name, rows in (
                ("train", train),
                ("calibration", calibration),
                ("acceptance", acceptance),
            )
        }
        if (
            groups["train"] & groups["calibration"]
            or groups["train"] & groups["acceptance"]
            or groups["calibration"] & groups["acceptance"]
        ):
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
    summaries = {
        "train_calibration": summarize_near_pairs(near_tc),
        "train_acceptance": summarize_near_pairs(near_ta),
        "calibration_acceptance": summarize_near_pairs(near_ca),
    }
    has_near_overlap = bool(near_tc or near_ta or near_ca)

    diagnostic = None
    engineering_override = args.engineering_near_overlap_override_from is not None
    if has_near_overlap and not engineering_override:
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
            "exact_overlap_counts": exact_counts,
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
        raise ValueError(
            "logical roles contain near-duplicate actor observations; automatic "
            "iteration stops without replacement. diagnostic="
            + json.dumps(concise_summaries(summaries), sort_keys=True, allow_nan=False)
        )

    if engineering_override:
        if not has_near_overlap:
            raise ValueError("engineering near-overlap override requested but no overlap exists")
        diagnostic = validate_engineering_override(
            args.engineering_near_overlap_override_from,
            tm=tm,
            cm=cm,
            am=am,
            atol=atol,
            summaries=summaries,
            exact_counts=exact_counts,
        )

    tube = load_soft_tube(args.target_tube)
    if int(tube.manifest.get("iteration", -1)) != int(tm["iteration"]) + 1:
        raise ValueError("target Tube iteration drift")
    target_states = {str(r["state_sha256"]) for r in tube.entries}
    calibration_overlap = target_states & sets["calibration"]
    acceptance_overlap = target_states & sets["acceptance"]
    if calibration_overlap or acceptance_overlap:
        raise ValueError("calibration/acceptance state entered the target Tube")

    if engineering_override:
        report = {
            "schema": SCHEMA,
            "status": "independent_for_candidate_training_engineering",
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
            "train_calibration_near_overlap_count": int(summaries["train_calibration"]["pair_count"]),
            "train_acceptance_near_overlap_count": 0,
            "calibration_acceptance_near_overlap_count": int(summaries["calibration_acceptance"]["pair_count"]),
            "calibration_target_tube_overlap_count": 0,
            "acceptance_target_tube_overlap_count": 0,
            "candidate_training_acceptance_isolation_passed": True,
            "formal_all_role_geometric_isolation_passed": False,
            "engineering_near_overlap_override_used": True,
            "strict_failure_diagnostic": str(args.engineering_near_overlap_override_from),
            "strict_failure_diagnostic_sha256": diagnostic["diagnostic_sha256"],
            "strict_failure_diagnostic_file_sha256": file_sha256(
                args.engineering_near_overlap_override_from
            ),
            "near_overlap_summary": concise_summaries(summaries),
            "claim_boundary": {
                "engineering_mainline_only": True,
                "formal_strict_role_isolation_pass_claim": False,
                "parent_group_disjointness_preserved": True,
                "exact_state_disjointness_preserved": True,
                "train_acceptance_geometric_isolation_preserved": True,
                "calibration_geometric_isolation_claim": False,
                "rows_removed_or_reassigned": False,
                "observation_atol_changed": False,
                "test_or_final_data_used": False,
            },
            "no_replacement_after_exclusion": True,
            "training_transitions": 0,
            "environment_interactions": 0,
            "test_data_used": False,
            "final_evaluation_data_used": False,
        }
    else:
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
            "candidate_training_acceptance_isolation_passed": True,
            "formal_all_role_geometric_isolation_passed": True,
            "engineering_near_overlap_override_used": False,
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
