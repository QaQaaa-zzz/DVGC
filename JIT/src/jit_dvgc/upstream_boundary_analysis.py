"""Post-hoc TRAIN analysis for reachable V_up boundary pilots.

This module performs no environment interaction. It joins an immutable boundary
catalog with its continuation labels and reports whether the *failure anchor
itself* crosses the continuation boundary, rather than relying on a potentially
confounded aggregate success rate across positive guard anchors.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

# Operational gates, not physical constants.  They prevent a dataset from being
# declared lock-ready when boundary evidence exists only as a tiny minority such
# as 2 successes among 42 failure-anchor perturbations.
MIN_LOCAL_MINORITY_COUNT_FOR_LOCK = 4
MIN_LOCAL_MINORITY_FRACTION_FOR_LOCK = 0.15


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _closed_reason(label: Mapping[str, Any]) -> str:
    branches = list(label.get("branches", []))
    if len(branches) != 1:
        raise ValueError("boundary analysis expects one deterministic branch per candidate")
    return str(branches[0]["reason"])


def _summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    successes = sum(int(row["success"]) for row in rows)
    total = len(rows)
    failures = total - successes
    return {
        "count": total,
        "successes": successes,
        "failures": failures,
        "success_rate": (successes / total) if total else None,
        "mixed_outcomes": bool(successes and failures),
        "reasons": dict(sorted(Counter(str(row["reason"]) for row in rows).items())),
    }


def _group(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    result = []
    for values, members in sorted(
        grouped.items(), key=lambda item: tuple(str(x) for x in item[0])
    ):
        result.append({**dict(zip(keys, values)), **_summary(members)})
    return result


def _join(catalog: Mapping[str, Any], labels: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    entries = list(catalog.get("entries", []))
    if not entries:
        raise ValueError("boundary catalog is empty")
    if len(entries) != len(labels):
        raise ValueError("boundary catalog/label candidate counts differ")
    result = []
    seen = set()
    for label in labels:
        index = int(label.get("boundary_candidate_index", -1))
        if index < 0 or index >= len(entries) or index in seen:
            raise ValueError("invalid or duplicate boundary_candidate_index")
        seen.add(index)
        entry = entries[index]
        for key in ("state_sha256", "parent_group_id", "seed"):
            if str(entry.get(key)) != str(label.get(key)):
                raise ValueError(f"catalog/label mismatch for {key} at index {index}")
        if str(entry.get("protocol_sha256")) != str(
            label.get("boundary_protocol_sha256")
        ):
            raise ValueError("boundary protocol hash mismatch")
        perturbation = dict(entry.get("perturbation", {}))
        result.append(
            {
                "index": index,
                "success": bool(int(label.get("success_count", 0))),
                "reason": _closed_reason(label),
                "anchor_kind": str(entry.get("anchor_kind", "")),
                "anchor_parent_group_id": str(
                    entry.get("anchor_parent_group_id", "")
                ),
                "anchor_state_sha256": str(entry.get("anchor_state_sha256", "")),
                "anchor_role": str(entry.get("anchor_role", entry.get("role", ""))),
                "action_name": str(perturbation.get("action_name", "")),
                "sign": int(perturbation.get("sign", 0)),
                "strength": float(perturbation.get("strength", 0.0)),
                "duration": int(perturbation.get("duration", 0)),
            }
        )
    return result


def _audit_sensitivity(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Separate exact, strict-threshold, and practical physical-near duplication.

    The practical profile is descriptive only; it never changes anchor selection.
    It deliberately uses loose-but-small TRAIN diagnostics to expose seed-level
    pseudo-diversity that a 1e-5 all-field threshold can miss.
    """
    pairs = list(audit.get("pairwise_distances", []))
    practical = {
        "qpos_linf": 5.0e-4,
        "qvel_linf": 2.0e-3,
        "observation_linf": 1.0e-2,
    }
    practical_pairs = [
        row
        for row in pairs
        if float(row.get("qpos_linf", float("inf"))) <= practical["qpos_linf"]
        and float(row.get("qvel_linf", float("inf"))) <= practical["qvel_linf"]
        and float(row.get("observation_linf", float("inf")))
        <= practical["observation_linf"]
    ]
    exact_pairs = sum(int(bool(row.get("exact", False))) for row in pairs)
    strict_pairs = sum(int(bool(row.get("near_duplicate", False))) for row in pairs)
    return {
        "pair_count": len(pairs),
        "exact_duplicate_pair_count": exact_pairs,
        "strict_near_duplicate_pair_count": strict_pairs,
        "strict_tolerances": audit.get("near_duplicate_tolerances"),
        "practical_profile": practical,
        "practical_near_duplicate_pair_count": len(practical_pairs),
        "all_pairs_practically_near_duplicate": bool(pairs)
        and len(practical_pairs) == len(pairs),
        "interpretation": (
            "seed-level pseudo-diversity remains physically concentrated"
            if pairs and len(practical_pairs) == len(pairs)
            else "failure states show more than one practical physical pocket"
        ),
    }


def _strength_brackets(
    rows: list[dict[str, Any]], keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Report observed failure-to-success strength brackets for focused refinement."""
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)

    result = []
    for values, members in sorted(
        grouped.items(), key=lambda item: tuple(str(x) for x in item[0])
    ):
        by_strength: dict[float, list[dict[str, Any]]] = defaultdict(list)
        for row in members:
            by_strength[float(row["strength"])].append(row)
        summaries = {
            strength: _summary(items) for strength, items in by_strength.items()
        }
        success_strengths = sorted(
            strength
            for strength, summary in summaries.items()
            if summary["successes"] > 0
        )
        failure_strengths = sorted(
            strength
            for strength, summary in summaries.items()
            if summary["failures"] > 0
        )
        bracket = None
        if success_strengths and failure_strengths:
            for upper in success_strengths:
                lower_candidates = [x for x in failure_strengths if x < upper]
                if lower_candidates:
                    bracket = (max(lower_candidates), upper)
                    break
        if bracket is not None:
            result.append(
                {
                    **dict(zip(keys, values)),
                    "lower_failure_strength": bracket[0],
                    "upper_success_strength": bracket[1],
                    "width": bracket[1] - bracket[0],
                }
            )
    return result


def analyze_boundary_pilot(
    catalog_path: Path,
    labels_path: Path,
    *,
    audit_path: Path | None = None,
) -> dict[str, Any]:
    catalog = _load_json(catalog_path)
    labels = _load_json(labels_path)
    if not isinstance(labels, list):
        raise ValueError("boundary labels must be a list")
    rows = _join(catalog, labels)
    failure_rows = [row for row in rows if row["anchor_kind"] == "failure_anchor"]
    guard_rows = [row for row in rows if row["anchor_kind"] == "positive_guard"]
    if not failure_rows:
        raise ValueError("boundary pilot contains no failure_anchor candidates")

    failure_summary = _summary(failure_rows)
    guard_summary = _summary(guard_rows)
    by_direction = _group(failure_rows, ("action_name", "sign"))
    mixed_directions = [row for row in by_direction if row["mixed_outcomes"]]
    minority_count = min(failure_summary["successes"], failure_summary["failures"])
    failure_side_fraction = (
        minority_count / failure_summary["count"] if failure_summary["count"] else 0.0
    )

    boundary_found = bool(failure_summary["mixed_outcomes"])
    dataset_lock_ready = bool(
        boundary_found
        and mixed_directions
        and minority_count >= MIN_LOCAL_MINORITY_COUNT_FOR_LOCK
        and failure_side_fraction >= MIN_LOCAL_MINORITY_FRACTION_FOR_LOCK
    )

    if boundary_found:
        decision = "BOUNDARY_FOUND"
        next_step = (
            "lock_train_protocol_then_validate_without_retuning"
            if dataset_lock_ready
            else "refine_discovered_crossing_direction_train_only"
        )
    else:
        decision = "REFINE_FAILURE_ANCHOR_ONLY"
        next_step = "adjust_strength_duration_or_direction_using_train_failure_anchor_only"

    report = {
        "schema": "jit_upstream_boundary_analysis_v1",
        "status": "completed",
        "target": "V_up",
        "split": "train",
        "candidate_count": len(rows),
        "aggregate": _summary(rows),
        "failure_anchor": failure_summary,
        "positive_guards": guard_summary,
        "failure_anchor_minority_side_count": minority_count,
        "failure_anchor_minority_side_fraction": failure_side_fraction,
        "failure_anchor_by_action_axis": _group(failure_rows, ("action_name",)),
        "failure_anchor_by_direction": by_direction,
        "failure_anchor_by_strength": _group(failure_rows, ("strength",)),
        "failure_anchor_by_duration": _group(failure_rows, ("duration",)),
        "failure_anchor_by_direction_strength": _group(
            failure_rows, ("action_name", "sign", "strength")
        ),
        "failure_anchor_by_direction_duration": _group(
            failure_rows, ("action_name", "sign", "duration")
        ),
        "failure_anchor_crossing_brackets": _strength_brackets(
            failure_rows, ("action_name", "sign")
        ),
        "failure_anchor_crossing_brackets_by_duration": _strength_brackets(
            failure_rows, ("action_name", "sign", "duration")
        ),
        "mixed_failure_directions": [
            {
                "action_name": row["action_name"],
                "sign": row["sign"],
                "count": row["count"],
                "success_rate": row["success_rate"],
            }
            for row in mixed_directions
        ],
        "mixed_failure_direction_count": len(mixed_directions),
        "boundary_evidence": boundary_found,
        "dataset_lock_ready": dataset_lock_ready,
        "dataset_lock_gate": {
            "min_local_minority_count": MIN_LOCAL_MINORITY_COUNT_FOR_LOCK,
            "min_local_minority_fraction": MIN_LOCAL_MINORITY_FRACTION_FOR_LOCK,
            "note": "operational anti-sparsity gate, not a physical constant",
        },
        "decision": decision,
        "next_step": next_step,
    }
    if audit_path is not None:
        report["pseudo_diversity_audit"] = _audit_sensitivity(_load_json(audit_path))
    return report


def write_boundary_analysis(
    catalog_path: Path,
    labels_path: Path,
    output_path: Path,
    *,
    audit_path: Path | None = None,
) -> dict[str, Any]:
    report = analyze_boundary_pilot(
        catalog_path, labels_path, audit_path=audit_path
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
