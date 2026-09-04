"""Resolution-aware parent selection for JIT frontier acquisition.

Exact snapshot hashes are reproducibility identities, not geometric diversity
metrics.  This module revises a still-outcome-blind frontier plan so selected
parents are unique in both parent group and declared root-geometry capability
cell.  It never changes policy, Tube, reward, physics, probe outcomes, or TEST.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..analysis.capability_tube import (
    SCHEMA as CAPABILITY_TUBE_SCHEMA,
    load_projected_capability_entries,
)
from ..config import file_sha256
from ..soft_tube import load_soft_tube


SCHEMA = "jit_resolution_aware_frontier_plan_revision_v1"
ROLE_PATTERN = ("train", "train", "train", "calibration", "acceptance")
ROLES = ("train", "calibration", "acceptance")
DEFAULT_MAX_PARENT_CELLS_PER_PHASE = 25


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _verify_hash(payload: Mapping[str, Any], field: str) -> None:
    declared = str(payload.get(field, ""))
    base = {key: value for key, value in payload.items() if key != field}
    if len(declared) != 64 or _canonical_sha256(base) != declared:
        raise ValueError(f"{field} self-hash drift")


def select_resolution_distinct_anchors(
    *,
    tube_entries: Sequence[Mapping[str, Any]],
    projected_entries: Sequence[Mapping[str, Any]],
    core_retained_count: int,
    max_parent_cells_per_phase: int = DEFAULT_MAX_PARENT_CELLS_PER_PHASE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(tube_entries) != len(projected_entries):
        raise ValueError("Tube and capability projection entry counts differ")
    if int(core_retained_count) <= 0 or int(core_retained_count) >= len(tube_entries):
        raise ValueError("resolution-aware frontier requires a nonempty newest Tube shell")
    projected_by_index = {int(row["global_index"]): row for row in projected_entries}
    if set(projected_by_index) != set(range(len(tube_entries))):
        raise ValueError("capability projection global-index coverage drift")

    selected: list[dict[str, Any]] = []
    audit: dict[str, Any] = {}
    for phase, phase_index in (("upstream", 0), ("downstream", 1)):
        candidates = []
        local_index = 0
        for global_index, row in enumerate(tube_entries):
            row_phase = str(row["phase"])
            current_local = local_index if row_phase == phase else None
            if row_phase == phase:
                local_index += 1
            if row_phase != phase or global_index < int(core_retained_count):
                continue
            projection = projected_by_index[global_index]
            if projection["phase"] != phase or projection["state_sha256"] != row["state_sha256"]:
                raise ValueError("capability projection identity drift during parent selection")
            candidates.append((global_index, int(current_local), row, projection))

        ordered = sorted(
            candidates,
            key=lambda item: (
                float(item[2]["value_score"]),
                str(item[2]["parent_group_id"]),
                str(item[3]["root_geometry_cell_id"]),
                str(item[2]["state_sha256"]),
            ),
        )
        chosen = []
        seen_groups: set[str] = set()
        seen_states: set[str] = set()
        seen_cells: set[str] = set()
        excluded_same_cell = 0
        excluded_same_group = 0
        for global_index, local, row, projection in ordered:
            group = str(row["parent_group_id"])
            state = str(row["state_sha256"])
            cell = str(projection["root_geometry_cell_id"])
            if group in seen_groups:
                excluded_same_group += 1
                continue
            if state in seen_states:
                continue
            if cell in seen_cells:
                excluded_same_cell += 1
                continue
            seen_groups.add(group)
            seen_states.add(state)
            seen_cells.add(cell)
            chosen.append((global_index, local, row, projection))
            if len(chosen) >= int(max_parent_cells_per_phase):
                break

        if len(chosen) < 5:
            raise ValueError(
                f"resolution-aware frontier needs >=5 distinct newest-shell root geometry cells "
                f"in {phase}; found {len(chosen)}"
            )
        counts = Counter()
        for index, (global_index, local, row, projection) in enumerate(chosen):
            role = ROLE_PATTERN[index % len(ROLE_PATTERN)]
            counts[role] += 1
            selected.append(
                {
                    "role": role,
                    "phase": phase,
                    "phase_index": phase_index,
                    "entry_index": int(local),
                    "global_index": int(global_index),
                    "state_sha256": str(row["state_sha256"]),
                    "parent_group_id": str(row["parent_group_id"]),
                    "value_score": float(row["value_score"]),
                    "sampling_weight": float(row["sampling_weight"]),
                    "root_geometry_cell_id": str(projection["root_geometry_cell_id"]),
                    "root_geometry_bins": dict(projection["root_geometry_bins"]),
                }
            )
        if counts["train"] < 3 or counts["calibration"] < 1 or counts["acceptance"] < 1:
            raise ValueError(f"resolution-aware role split insufficient in {phase}")
        audit[phase] = {
            "newest_shell_raw_candidate_count": len(candidates),
            "distinct_selected_root_geometry_cell_count": len(chosen),
            "excluded_same_root_geometry_cell_count": excluded_same_cell,
            "excluded_same_parent_group_count": excluded_same_group,
            "role_counts": {role: int(counts[role]) for role in ROLES},
        }
    return selected, audit


def revise_frontier_plan_for_resolution_cells(
    *,
    source_plan: Path,
    source_tube: Path,
    capability_geometry_summary: Path,
    output: Path,
    max_parent_cells_per_phase: int = DEFAULT_MAX_PARENT_CELLS_PER_PHASE,
) -> dict[str, Any]:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"resolution-aware frontier plan already exists: {output}")
    plan = json.loads(Path(source_plan).read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or plan.get("schema") != "jit_iterative_frontier_plan_v1":
        raise ValueError("resolution-aware revision requires an iterative frontier plan")
    if plan.get("status") != "predeclared_before_frontier_outcomes":
        raise ValueError("frontier plan is no longer outcome-blind/predeclared")
    _verify_hash(plan, "plan_sha256")
    if tuple(plan.get("role_pattern", ())) != ROLE_PATTERN:
        raise ValueError("frontier role pattern drift before resolution-aware revision")

    tube = load_soft_tube(Path(source_tube))
    if str(tube.manifest["manifest_sha256"]) != str(plan["source_tube_manifest_sha256"]):
        raise ValueError("resolution-aware revision source Tube identity drift")
    if len(tube.entries) != int(plan["source_tube_entry_count"]):
        raise ValueError("resolution-aware revision source Tube entry-count drift")

    geometry_summary, projected = load_projected_capability_entries(
        Path(capability_geometry_summary)
    )
    if geometry_summary.get("schema") != CAPABILITY_TUBE_SCHEMA:
        raise ValueError("capability geometry schema drift")
    if geometry_summary.get("tube_manifest_sha256") != tube.manifest["manifest_sha256"]:
        raise ValueError("capability geometry was not built from the frontier source Tube")

    anchors, audit = select_resolution_distinct_anchors(
        tube_entries=tube.entries,
        projected_entries=projected,
        core_retained_count=int(plan["source_tube_core_retained_count"]),
        max_parent_cells_per_phase=int(max_parent_cells_per_phase),
    )
    revised = {key: value for key, value in plan.items() if key != "plan_sha256"}
    revised["anchors"] = anchors
    revised["role_parent_group_counts"] = {
        phase: dict(audit[phase]["role_counts"])
        for phase in ("upstream", "downstream")
    }
    revised["frontier_definition"] = (
        "newest_expansion_shell_lowest_score_parent_group_and_root_geometry_cell_unique"
    )
    revised["capability_geometry"] = {
        "summary": str(capability_geometry_summary),
        "summary_file_sha256": file_sha256(capability_geometry_summary),
        "summary_sha256": str(geometry_summary["summary_sha256"]),
        "resolution_sha256": str(
            geometry_summary["resolution_contract"]["resolution_sha256"]
        ),
        "parent_diversity_profile": "root_geometry_v1",
        "max_parent_cells_per_phase": int(max_parent_cells_per_phase),
        "selection_audit": audit,
    }
    revised["protocol_revision"] = {
        "name": "resolution_aware_parent_cells_v1",
        "purpose": (
            "replace exact-SHA-only parent diversity with physically resolved root-geometry "
            "cell diversity before any frontier outcomes are observed"
        ),
        "supersedes_plan": str(source_plan),
        "supersedes_plan_sha256": str(plan["plan_sha256"]),
        "changed_fields": [
            "anchors",
            "role_parent_group_counts",
            "frontier_definition",
            "capability_geometry",
        ],
        "unchanged_contracts": [
            "selected_policy_identity",
            "source_tube_identity",
            "newest_shell_only_parent_pool",
            "outcome_blind_role_assignment",
            "real_dynamics_only",
            "probe_panel",
            "role_seeds",
            "reward_physics_action_semantics",
            "test_isolation",
        ],
        "outcomes_observed_before_revision": False,
    }
    revised["plan_sha256"] = _canonical_sha256(revised)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(revised, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report = {
        "schema": SCHEMA,
        "status": "completed_pre_outcome_revision",
        "source_plan": str(source_plan),
        "source_plan_sha256": str(plan["plan_sha256"]),
        "revised_plan": str(output),
        "revised_plan_sha256": str(revised["plan_sha256"]),
        "source_tube_manifest_sha256": str(tube.manifest["manifest_sha256"]),
        "capability_geometry_summary": str(capability_geometry_summary),
        "capability_resolution_sha256": str(
            geometry_summary["resolution_contract"]["resolution_sha256"]
        ),
        "selection_audit": audit,
        "test_data_used": False,
        "final_evaluation_data_used": False,
    }
    report["report_sha256"] = _canonical_sha256(report)
    sidecar = output.with_suffix(output.suffix + ".resolution.json")
    sidecar.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
