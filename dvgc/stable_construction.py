"""Pure rules for cross-seed stable-safe construction."""
from __future__ import annotations

from typing import Mapping, Sequence

from dvgc.bank import beta_posterior, posterior_label


def protocol_from_config(cfg) -> dict:
    return {
        "alpha0": float(cfg.beta_alpha0),
        "beta0": float(cfg.beta_beta0),
        "q_low": float(cfg.posterior_q_low),
        "q_high": float(cfg.posterior_q_high),
        "min_branches": int(cfg.min_branches),
        "safe_threshold": float(cfg.safe_threshold),
        "dead_threshold": float(cfg.dead_threshold),
        "boundary_max_width": float(cfg.boundary_max_width),
        "stage_branches": int(cfg.stable_construction_stage_branches),
        "adaptive_max_branches": int(cfg.stable_construction_adaptive_max_branches),
        "near_safe_lcb_margin": float(cfg.stable_construction_near_safe_lcb_margin),
    }


def outcome(evidence: Sequence[Mapping], cfg, key: str = "final_recovery") -> dict:
    successes = sum(bool(row.get(key)) for row in evidence)
    branches = len(evidence)
    posterior = beta_posterior(
        successes, branches - successes,
        alpha0=cfg.beta_alpha0, beta0=cfg.beta_beta0,
        q_low=cfg.posterior_q_low, q_high=cfg.posterior_q_high,
    )
    label = posterior_label(
        posterior, branches, min_branches=cfg.min_branches,
        safe_threshold=cfg.safe_threshold, dead_threshold=cfg.dead_threshold,
        boundary_max_width=cfg.boundary_max_width,
    )
    return {
        "successes": successes, "failures": branches - successes,
        "branches": branches, "posterior": posterior, "label": label,
    }


def stage_b_indices(stage_a_rows: Sequence[Mapping], cfg) -> list[int]:
    """Confirm every non-dead Stage-A state; dead states cannot be stable-safe."""
    selected = []
    for row in stage_a_rows:
        label = outcome(row["branch_evidence"], cfg)["label"]
        if label != "dead":
            selected.append(int(row["candidate_index"]))
    return selected


def needs_adaptive(stage_a: Mapping, stage_b: Mapping, cfg) -> bool:
    a = outcome(stage_a["branch_evidence"], cfg)
    b = outcome(stage_b["branch_evidence"], cfg)
    combined = outcome(stage_a["branch_evidence"] + stage_b["branch_evidence"], cfg)
    near = abs(float(combined["posterior"]["lower"]) - float(cfg.safe_threshold)) <= float(
        cfg.stable_construction_near_safe_lcb_margin
    )
    return a["label"] != b["label"] or combined["label"] == "unknown" or near


def adaptive_indices(stage_a_rows: Sequence[Mapping], stage_b_rows: Sequence[Mapping], cfg) -> list[int]:
    by_b = {int(row["candidate_index"]): row for row in stage_b_rows}
    return [
        int(row["candidate_index"]) for row in stage_a_rows
        if int(row["candidate_index"]) in by_b and needs_adaptive(row, by_b[int(row["candidate_index"])], cfg)
    ]


def stable_result(stage_a: Mapping, stage_b: Mapping | None, adaptive: Mapping | None, cfg) -> dict:
    a = outcome(stage_a["branch_evidence"], cfg)
    b = outcome(stage_b["branch_evidence"], cfg) if stage_b is not None else None
    evidence = list(stage_a["branch_evidence"])
    if stage_b is not None:
        evidence.extend(stage_b["branch_evidence"])
    if adaptive is not None:
        evidence.extend(adaptive["branch_evidence"])
    combined = outcome(evidence, cfg)
    unhealthy = any(
        row.get("terminal_cause") == "timeout" or row.get("end_reason") == "nonfinite"
        for row in evidence
    )
    stable_safe = bool(
        b is not None and a["label"] == "safe" and b["label"] == "safe"
        and combined["label"] == "safe" and not unhealthy
    )
    if stable_safe:
        label = "safe"
    elif combined["label"] == "dead":
        label = "dead"
    elif combined["label"] == "boundary":
        label = "boundary"
    else:
        label = "unknown"
    return {
        "label": label, "stable_safe": stable_safe, "stage_a": a, "stage_b": b,
        "adaptive": outcome(adaptive["branch_evidence"], cfg) if adaptive is not None else None,
        "combined": combined, "branch_evidence": evidence,
        "timeout_or_nonfinite": unhealthy,
    }
