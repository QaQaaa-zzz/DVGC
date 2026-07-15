"""Pure construction and merge helpers for independent Tube audit reports."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .certification import summarize_branches


def build_audit_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy_version: str,
    phase: str,
    seed_namespace: str,
    branches_per_state: int,
    safe_threshold: float,
    dynamics_variants: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute audit metrics from complete per-state branch evidence."""
    audit = [dict(row) for row in rows]
    expected = int(branches_per_state)
    if expected <= 0:
        raise ValueError("branches_per_state must be positive")
    if any(len(row.get("branches", [])) != expected for row in audit):
        raise ValueError("Every audited state must contain branches_per_state evidence rows")
    all_branches = [branch for row in audit for branch in row["branches"]]
    if any(str(branch.get("seed_namespace")) != str(seed_namespace) for branch in all_branches):
        raise ValueError("Audit branch namespace mismatch")

    pred_safe = np.asarray([row["predicted_label"] == "safe" for row in audit])
    recoverable = np.asarray([float(row["audit_final"]) >= float(safe_threshold) for row in audit])
    probs = np.asarray([row["predicted_mean"] for row in audit], float)
    obs = np.asarray([row["audit_final"] for row in audit], float)
    precision = float(recoverable[pred_safe].mean()) if pred_safe.any() else float("nan")
    recall = float(pred_safe[recoverable].mean()) if recoverable.any() else float("nan")
    coverage = float(pred_safe.mean()) if len(pred_safe) else 0.0
    brier = float(np.mean((probs - obs) ** 2)) if len(obs) else float("nan")
    bins = np.linspace(0, 1, 6)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < (hi if hi < 1 else hi + 1e-9))
        if mask.any():
            ece += float(mask.mean() * abs(probs[mask].mean() - obs[mask].mean()))
    terminal = summarize_branches(all_branches)
    return {
        "policy_version": str(policy_version),
        "phase": str(phase),
        "seed_namespace": str(seed_namespace),
        "states": len(audit),
        "branches_per_state": expected,
        "safe_threshold": float(safe_threshold),
        "tube_precision": precision,
        "recoverable_recall": recall,
        "candidate_mass_coverage": coverage,
        "brier": brier,
        "ece_5bin": ece,
        "false_progress_rate": terminal["false_progress_rate"],
        "missed_success_rate": terminal["missed_success_rate"],
        "physical_failure_rate": terminal["physical_failure_rate"],
        "timeout_rate": terminal["timeout_rate"],
        "horizon_exhaustion_rate": terminal["horizon_exhaustion_rate"],
        "terminal_summary": terminal,
        "dynamics_variants": [dict(spec) for spec in dynamics_variants],
        "rows": audit,
    }


def merge_audit_reports(parts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Merge contiguous policy-identical audit parts without changing seeds."""
    reports = [dict(part) for part in parts]
    if not reports:
        raise ValueError("At least one audit part is required")
    common_keys = (
        "policy_version",
        "phase",
        "seed_namespace",
        "branches_per_state",
        "safe_threshold",
        "total_bank_states",
        "dynamics_variants",
    )
    expected = {key: reports[0].get(key) for key in common_keys}
    for report in reports[1:]:
        for key, value in expected.items():
            if report.get(key) != value:
                raise ValueError(f"Audit parts disagree on {key}")

    total = int(expected["total_bank_states"])
    for report in reports:
        start = int(report["state_index_start"])
        stop = int(report["state_index_end_exclusive"])
        part_rows = report.get("rows", [])
        part_indices = [int(row["state_index"]) for row in part_rows]
        if start < 0 or stop > total or start >= stop:
            raise ValueError("Audit part has an invalid global state range")
        if part_indices != list(range(start, stop)):
            raise ValueError("Audit part rows do not match its declared global state range")
        if int(report.get("states", -1)) != len(part_rows):
            raise ValueError("Audit part state count does not match its evidence rows")

    rows = [dict(row) for report in reports for row in report.get("rows", [])]
    rows.sort(key=lambda row: int(row["state_index"]))
    indices = [int(row["state_index"]) for row in rows]
    if indices != list(range(total)):
        raise ValueError("Audit parts must cover each global state index exactly once")
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Audit parts contain duplicate state ids")
    branch_seeds = [
        int(branch["branch_seed"])
        for row in rows
        for branch in row.get("branches", [])
    ]
    if len(branch_seeds) != len(set(branch_seeds)):
        raise ValueError("Audit parts reuse branch seeds")

    merged = build_audit_report(
        rows,
        policy_version=str(expected["policy_version"]),
        phase=str(expected["phase"]),
        seed_namespace=str(expected["seed_namespace"]),
        branches_per_state=int(expected["branches_per_state"]),
        safe_threshold=float(expected["safe_threshold"]),
        dynamics_variants=expected["dynamics_variants"],
    )
    merged.update(
        {
            "state_index_start": 0,
            "state_index_end_exclusive": total,
            "total_bank_states": total,
            "merged_part_ranges": [
                [int(report["state_index_start"]), int(report["state_index_end_exclusive"])]
                for report in reports
            ],
        }
    )
    merged["grouped_candidate_metrics"] = grouped_audit_metrics(
        rows, policy_version=str(expected["policy_version"]), phase=str(expected["phase"]),
        seed_namespace=str(expected["seed_namespace"]), branches_per_state=int(expected["branches_per_state"]),
        safe_threshold=float(expected["safe_threshold"]), dynamics_variants=expected["dynamics_variants"],
    )
    return merged


def grouped_audit_metrics(rows, **kwargs) -> dict[str, Any]:
    groups={}
    for kind in sorted({str(row.get("candidate_kind","unknown")) for row in rows}):
        subset=[row for row in rows if str(row.get("candidate_kind","unknown"))==kind]
        report=build_audit_report(subset,**kwargs)
        groups[kind]={key:value for key,value in report.items() if key not in ("rows","dynamics_variants")}
    return groups
