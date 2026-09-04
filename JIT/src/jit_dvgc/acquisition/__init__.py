"""Stable real-dynamics acquisition API for envelope iterations."""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from ..unified_boundary import (
    DEFAULT_ANCHORS_PER_PHASE,
    DEFAULT_FRONTIER_SCORE_CEILING,
    DEFAULT_UNIFIED_BOUNDARY_DURATIONS,
    DEFAULT_UNIFIED_BOUNDARY_STRENGTHS,
    TubeBoundaryAnchor,
    action_sparse_directions,
    collect_unified_boundary_candidates,
    select_tube_boundary_anchors,
)
from ..unified_transition_band_search import (
    load_transition_band_search_config,
    search_unified_transition_band,
)
from .fresh_parent import (
    DEFAULT_BOUNDARY_ACTIVE_DIMENSIONS,
    DEFAULT_BOUNDARY_DURATIONS as DEFAULT_FRESH_PARENT_BOUNDARY_DURATIONS,
    DEFAULT_BOUNDARY_STRENGTHS as DEFAULT_FRESH_PARENT_BOUNDARY_STRENGTHS,
    DEFAULT_DOWNSTREAM_APEX_OFFSET,
    DEFAULT_MINIMUM_PARENT_GROUPS,
    DEFAULT_PARENT_EXCITATION_DURATION,
    DEFAULT_PARENT_EXCITATION_STRENGTH,
    DEFAULT_PARENT_NEAR_DUPLICATE_ATOL,
    DEFAULT_UPSTREAM_APEX_OFFSET,
    FRESH_PARENT_SOURCE_SCHEMA,
    FRESH_PARENT_TYPE,
    FreshParentAnchor,
    audit_fresh_parent_predeclaration,
    baseline_probe_declaration,
    collect_fresh_parent_anchors,
    collect_snapshot_anchor_boundary_candidates,
    consumed_baseline_probe_exclusions,
    planned_parent_groups,
    prepare_fresh_parent_acceptance_predeclaration,
)
from .resolution_frontier import (
    DEFAULT_MAX_PARENT_CELLS_PER_PHASE,
    revise_frontier_plan_for_resolution_cells,
    select_resolution_distinct_anchors,
)


def select_disjoint_tube_boundary_anchors(
    artifact: Any,
    *,
    max_per_phase: int = DEFAULT_ANCHORS_PER_PHASE,
    minimum_per_phase: int = 1,
    frontier_score_ceiling: float = DEFAULT_FRONTIER_SCORE_CEILING,
    excluded_state_sha256: Sequence[str] = (),
    excluded_parent_groups: Mapping[str, Sequence[str]] | None = None,
) -> tuple[tuple[TubeBoundaryAnchor, ...], dict[str, Any]]:
    """Select frontier anchors after excluding consumed audit neighborhoods."""
    if int(max_per_phase) <= 0:
        raise ValueError("max_per_phase must be positive")
    if int(minimum_per_phase) <= 0 or int(minimum_per_phase) > int(max_per_phase):
        raise ValueError("minimum_per_phase must lie in [1, max_per_phase]")

    excluded_states = {str(value) for value in excluded_state_sha256}
    excluded_groups = {
        phase: {str(value) for value in values}
        for phase, values in (excluded_parent_groups or {}).items()
    }
    if any(phase not in {"upstream", "downstream"} for phase in excluded_groups):
        raise ValueError("excluded_parent_groups contains an unsupported phase")

    all_anchors, base_audit = select_tube_boundary_anchors(
        artifact,
        max_per_phase=max(len(artifact.entries), int(max_per_phase)),
        frontier_score_ceiling=frontier_score_ceiling,
    )

    selected: list[TubeBoundaryAnchor] = []
    by_phase: dict[str, Any] = {}
    for phase in ("upstream", "downstream"):
        phase_anchors = [anchor for anchor in all_anchors if anchor.phase == phase]
        state_excluded = [
            anchor for anchor in phase_anchors if anchor.state_sha256 in excluded_states
        ]
        parent_excluded = [
            anchor
            for anchor in phase_anchors
            if anchor.state_sha256 not in excluded_states
            and anchor.parent_group_id in excluded_groups.get(phase, set())
        ]
        available = [
            anchor
            for anchor in phase_anchors
            if anchor.state_sha256 not in excluded_states
            and anchor.parent_group_id not in excluded_groups.get(phase, set())
        ]
        chosen = available[: int(max_per_phase)]
        if len(chosen) < int(minimum_per_phase):
            raise ValueError(
                f"disjoint frontier has only {len(chosen)} {phase} anchors; "
                f"minimum is {int(minimum_per_phase)}"
            )
        selected.extend(chosen)
        base_phase = dict(base_audit["by_phase"][phase])
        base_phase.pop("selected", None)
        base_phase["pre_exclusion_parent_unique_count"] = len(phase_anchors)
        base_phase["excluded_consumed_state_count"] = len(state_excluded)
        base_phase["excluded_consumed_parent_group_count"] = len(parent_excluded)
        base_phase["available_after_exclusion_count"] = len(available)
        base_phase["selected_count"] = len(chosen)
        base_phase["selected"] = [
            {
                "entry_index": int(anchor.entry_index),
                "global_index": int(anchor.global_index),
                "state_sha256": anchor.state_sha256,
                "parent_group_id": anchor.parent_group_id,
                "value_score": anchor.value_score,
                "sampling_weight": float(anchor.row["sampling_weight"]),
                "role": str(anchor.row.get("role", "")),
                "source_bank": str(anchor.row.get("source_bank", "")),
            }
            for anchor in chosen
        ]
        by_phase[phase] = base_phase

    phase_counts = Counter(anchor.phase for anchor in selected)
    audit = {
        "schema": "jit_unified_boundary_anchor_audit_v1",
        "status": "completed",
        "split": "train_audit_candidate_generation",
        "selection": (
            "bootstrap_score_at_or_below_ceiling_parent_group_unique_state_unique_"
            "excluding_consumed_audit_state_and_parent_groups"
        ),
        "anchor_semantics": "weak_bootstrap_frontier_probe_not_certified_boundary",
        "frontier_score_ceiling": float(frontier_score_ceiling),
        "max_per_phase": int(max_per_phase),
        "minimum_per_phase": int(minimum_per_phase),
        "selected_anchor_count": len(selected),
        "selected_phase_counts": dict(sorted(phase_counts.items())),
        "source_tube_manifest_sha256": artifact.manifest["manifest_sha256"],
        "excluded_state_sha256_count": len(excluded_states),
        "excluded_parent_group_counts": {
            phase: len(excluded_groups.get(phase, set()))
            for phase in ("upstream", "downstream")
        },
        "test_data_used": False,
        "validation_data_used": False,
        "final_evaluation_data_used": False,
        "by_phase": by_phase,
    }
    return tuple(selected), audit


__all__ = [
    "DEFAULT_ANCHORS_PER_PHASE",
    "DEFAULT_FRONTIER_SCORE_CEILING",
    "DEFAULT_UNIFIED_BOUNDARY_DURATIONS",
    "DEFAULT_UNIFIED_BOUNDARY_STRENGTHS",
    "action_sparse_directions",
    "collect_unified_boundary_candidates",
    "select_tube_boundary_anchors",
    "select_disjoint_tube_boundary_anchors",
    "load_transition_band_search_config",
    "search_unified_transition_band",
    "FRESH_PARENT_SOURCE_SCHEMA",
    "FRESH_PARENT_TYPE",
    "FreshParentAnchor",
    "DEFAULT_PARENT_EXCITATION_STRENGTH",
    "DEFAULT_PARENT_EXCITATION_DURATION",
    "DEFAULT_UPSTREAM_APEX_OFFSET",
    "DEFAULT_DOWNSTREAM_APEX_OFFSET",
    "DEFAULT_PARENT_NEAR_DUPLICATE_ATOL",
    "DEFAULT_MINIMUM_PARENT_GROUPS",
    "DEFAULT_FRESH_PARENT_BOUNDARY_STRENGTHS",
    "DEFAULT_FRESH_PARENT_BOUNDARY_DURATIONS",
    "DEFAULT_BOUNDARY_ACTIVE_DIMENSIONS",
    "baseline_probe_declaration",
    "consumed_baseline_probe_exclusions",
    "prepare_fresh_parent_acceptance_predeclaration",
    "planned_parent_groups",
    "audit_fresh_parent_predeclaration",
    "collect_fresh_parent_anchors",
    "collect_snapshot_anchor_boundary_candidates",
    "DEFAULT_MAX_PARENT_CELLS_PER_PHASE",
    "revise_frontier_plan_for_resolution_cells",
    "select_resolution_distinct_anchors",
]
