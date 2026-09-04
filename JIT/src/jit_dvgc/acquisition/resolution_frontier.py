"""Trajectory-centered, resolution-aware JIT frontier parent selection.

Prospective Jump-Tube expansion is no longer defined by a global low-score shell.
Parents must lie on the longitudinal support of one successful nominal jump and
must represent distinct physical root-geometry cells. Selection is balanced over
0.1 m x slices. Downstream support must be post-apex/descending and late recovery
is excluded by the centerline terminal at first valid landing (never beyond 4.2 m).

The source Soft Tube remains immutable replay/provenance. This module only
revises still-outcome-blind frontier plans and never changes reward, physics,
policy, probe outcomes, or TEST.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..analysis.capability_tube import (
    SCHEMA as CAPABILITY_TUBE_SCHEMA,
    load_projected_capability_entries,
)
from ..analysis.nominal_jump_centerline import load_nominal_jump_centerline
from ..config import file_sha256
from ..soft_tube import load_soft_tube


SCHEMA = "jit_trajectory_centered_frontier_plan_revision_v1"
ROLE_PATTERN = ("train", "train", "train", "calibration", "acceptance")
ROLES = ("train", "calibration", "acceptance")
DEFAULT_MAX_PARENT_CELLS_PER_PHASE = 25
JUMP_X_MIN_M = 2.5
JUMP_X_HARD_MAX_M = 4.2
JUMP_X_STEP_M = 0.1


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _verify_hash(payload: Mapping[str, Any], field: str) -> None:
    declared = str(payload.get(field, ""))
    base = {key: value for key, value in payload.items() if key != field}
    if len(declared) != 64 or _canonical_sha256(base) != declared:
        raise ValueError(f"{field} self-hash drift")


def _centerline_bins(centerline: Mapping[str, Any] | None) -> dict[str, set[int]]:
    if centerline is None:
        lo = int(round(JUMP_X_MIN_M / JUMP_X_STEP_M))
        hi = int(round(JUMP_X_HARD_MAX_M / JUMP_X_STEP_M))
        all_bins = set(range(lo, hi + 1))
        return {"upstream": set(all_bins), "downstream": set(all_bins)}
    result = {"upstream": set(), "downstream": set()}
    for point in centerline["points"]:
        x_bin = int(round(float(point["x_target_m"]) / JUMP_X_STEP_M))
        semantics = str(point["phase_semantics"])
        if semantics in {"upstream", "apex"}:
            result["upstream"].add(x_bin)
        if semantics in {"downstream", "apex"}:
            result["downstream"].add(x_bin)
    return result


def _semantic_eligible(
    phase: str,
    projection: Mapping[str, Any],
    *,
    allowed_bins: Mapping[str, set[int]],
    x_max_m: float,
) -> bool:
    coords = projection["coordinates"]
    x = float(coords["root_x_m"])
    if x < JUMP_X_MIN_M - 1.0e-9 or x > float(x_max_m) + 0.05 + 1.0e-9:
        return False
    if int(projection["x_bin"]) not in allowed_bins[phase]:
        return False
    if phase == "downstream" and not float(coords["root_vz_mps"]) < 0.0:
        return False
    return True


def select_resolution_distinct_anchors(
    *,
    tube_entries: Sequence[Mapping[str, Any]],
    projected_entries: Sequence[Mapping[str, Any]],
    core_retained_count: int,
    max_parent_cells_per_phase: int = DEFAULT_MAX_PARENT_CELLS_PER_PHASE,
    nominal_centerline: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select x-balanced physically distinct newest-shell Jump-Tube parents."""
    if len(tube_entries) != len(projected_entries):
        raise ValueError("Tube and capability projection entry counts differ")
    if int(core_retained_count) <= 0 or int(core_retained_count) >= len(tube_entries):
        raise ValueError("trajectory-centered frontier requires a nonempty newest Tube shell")
    projected_by_index = {int(row["global_index"]): row for row in projected_entries}
    if set(projected_by_index) != set(range(len(tube_entries))):
        raise ValueError("capability projection global-index coverage drift")

    allowed_bins = _centerline_bins(nominal_centerline)
    x_max_m = (
        float(nominal_centerline["effective_centerline_max_x_m"])
        if nominal_centerline is not None
        else JUMP_X_HARD_MAX_M
    )
    selected: list[dict[str, Any]] = []
    audit: dict[str, Any] = {}

    for phase, phase_index in (("upstream", 0), ("downstream", 1)):
        candidates = []
        local_index = 0
        newest_shell_count = 0
        semantic_rejected = 0
        for global_index, row in enumerate(tube_entries):
            row_phase = str(row["phase"])
            current_local = local_index if row_phase == phase else None
            if row_phase == phase:
                local_index += 1
            if row_phase != phase or global_index < int(core_retained_count):
                continue
            newest_shell_count += 1
            projection = projected_by_index[global_index]
            if projection["phase"] != phase or projection["state_sha256"] != row["state_sha256"]:
                raise ValueError("capability projection identity drift during parent selection")
            if not _semantic_eligible(
                phase,
                projection,
                allowed_bins=allowed_bins,
                x_max_m=x_max_m,
            ):
                semantic_rejected += 1
                continue
            candidates.append((global_index, int(current_local), row, projection))

        # One weakest-score representative per resolved root cell.
        best_by_cell: dict[str, tuple[int, int, Mapping[str, Any], Mapping[str, Any]]] = {}
        duplicate_cell_count = 0
        for item in candidates:
            cell = str(item[3]["root_geometry_cell_id"])
            current = best_by_cell.get(cell)
            if current is None:
                best_by_cell[cell] = item
                continue
            duplicate_cell_count += 1
            row, crow = item[2], current[2]
            key = (float(row["value_score"]), str(row["parent_group_id"]), str(row["state_sha256"]))
            ckey = (float(crow["value_score"]), str(crow["parent_group_id"]), str(crow["state_sha256"]))
            if key < ckey:
                best_by_cell[cell] = item

        # Rank locally, then round-robin x slices so dense regions cannot monopolize budget.
        by_x: dict[int, list[tuple[int, int, Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
        for item in best_by_cell.values():
            by_x[int(item[3]["x_bin"])].append(item)
        for x_bin in by_x:
            by_x[x_bin].sort(
                key=lambda item: (
                    float(item[2]["value_score"]),
                    str(item[2]["parent_group_id"]),
                    str(item[3]["root_geometry_cell_id"]),
                    str(item[2]["state_sha256"]),
                )
            )

        chosen = []
        seen_groups: set[str] = set()
        seen_states: set[str] = set()
        seen_cells: set[str] = set()
        excluded_same_group = 0
        cursors = {x_bin: 0 for x_bin in by_x}
        active_bins = sorted(by_x)
        while active_bins and len(chosen) < int(max_parent_cells_per_phase):
            next_active = []
            added = False
            for x_bin in active_bins:
                rows = by_x[x_bin]
                while cursors[x_bin] < len(rows):
                    item = rows[cursors[x_bin]]
                    cursors[x_bin] += 1
                    _global, _local, row, projection = item
                    group = str(row["parent_group_id"])
                    state = str(row["state_sha256"])
                    cell = str(projection["root_geometry_cell_id"])
                    if group in seen_groups:
                        excluded_same_group += 1
                        continue
                    if state in seen_states or cell in seen_cells:
                        continue
                    seen_groups.add(group)
                    seen_states.add(state)
                    seen_cells.add(cell)
                    chosen.append(item)
                    added = True
                    break
                if cursors[x_bin] < len(rows):
                    next_active.append(x_bin)
                if len(chosen) >= int(max_parent_cells_per_phase):
                    break
            if not added:
                break
            active_bins = next_active

        if len(chosen) < 5:
            raise ValueError(
                f"trajectory-centered frontier needs >=5 distinct newest-shell root geometry cells "
                f"in {phase}; found {len(chosen)} after centerline/corridor filtering"
            )

        counts = Counter()
        selected_x_bins = Counter()
        for index, (global_index, local, row, projection) in enumerate(chosen):
            role = ROLE_PATTERN[index % len(ROLE_PATTERN)]
            counts[role] += 1
            selected_x_bins[int(projection["x_bin"])] += 1
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
                    "x_bin": int(projection["x_bin"]),
                    "x_center_m": float(projection["x_center_m"]),
                    "root_vz_mps": float(projection["coordinates"]["root_vz_mps"]),
                    "jump_frontier_semantics": (
                        "pre_apex_centerline_slice" if phase == "upstream"
                        else "post_apex_descending_centerline_slice"
                    ),
                }
            )
        if counts["train"] < 3 or counts["calibration"] < 1 or counts["acceptance"] < 1:
            raise ValueError(f"trajectory-centered role split insufficient in {phase}")
        audit[phase] = {
            "newest_shell_raw_candidate_count": int(newest_shell_count),
            "semantic_rejected_count": int(semantic_rejected),
            "eligible_jump_corridor_candidate_count": len(candidates),
            "eligible_distinct_root_geometry_cell_count": len(best_by_cell),
            "distinct_selected_root_geometry_cell_count": len(chosen),
            "selected_x_bin_count": len(selected_x_bins),
            "selected_x_bins": {str(k): int(v) for k, v in sorted(selected_x_bins.items())},
            "excluded_same_root_geometry_cell_count": int(duplicate_cell_count),
            "excluded_same_parent_group_count": int(excluded_same_group),
            "role_counts": {role: int(counts[role]) for role in ROLES},
        }
    return selected, audit


def revise_frontier_plan_for_resolution_cells(
    *,
    source_plan: Path,
    source_tube: Path,
    capability_geometry_summary: Path,
    nominal_centerline: Path,
    output: Path,
    max_parent_cells_per_phase: int = DEFAULT_MAX_PARENT_CELLS_PER_PHASE,
) -> dict[str, Any]:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"trajectory-centered frontier plan already exists: {output}")
    plan = json.loads(Path(source_plan).read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or plan.get("schema") != "jit_iterative_frontier_plan_v1":
        raise ValueError("trajectory-centered revision requires an iterative frontier plan")
    if plan.get("status") != "predeclared_before_frontier_outcomes":
        raise ValueError("frontier plan is no longer outcome-blind/predeclared")
    _verify_hash(plan, "plan_sha256")
    if tuple(plan.get("role_pattern", ())) != ROLE_PATTERN:
        raise ValueError("frontier role pattern drift before trajectory-centered revision")

    tube = load_soft_tube(Path(source_tube))
    if str(tube.manifest["manifest_sha256"]) != str(plan["source_tube_manifest_sha256"]):
        raise ValueError("trajectory-centered revision source Tube identity drift")
    if len(tube.entries) != int(plan["source_tube_entry_count"]):
        raise ValueError("trajectory-centered revision source Tube entry-count drift")

    geometry_summary, projected = load_projected_capability_entries(Path(capability_geometry_summary))
    if geometry_summary.get("schema") != CAPABILITY_TUBE_SCHEMA:
        raise ValueError("capability geometry schema drift")
    if geometry_summary.get("tube_manifest_sha256") != tube.manifest["manifest_sha256"]:
        raise ValueError("capability geometry was not built from the frontier source Tube")
    if abs(float(geometry_summary["resolution_contract"]["x_slice_width_m"]) - JUMP_X_STEP_M) > 1.0e-12:
        raise ValueError("trajectory-centered frontier requires 0.1 m x slices")

    centerline = load_nominal_jump_centerline(Path(nominal_centerline))
    anchors, audit = select_resolution_distinct_anchors(
        tube_entries=tube.entries,
        projected_entries=projected,
        core_retained_count=int(plan["source_tube_core_retained_count"]),
        max_parent_cells_per_phase=int(max_parent_cells_per_phase),
        nominal_centerline=centerline,
    )

    revised = {key: value for key, value in plan.items() if key != "plan_sha256"}
    revised["anchors"] = anchors
    revised["role_parent_group_counts"] = {
        phase: dict(audit[phase]["role_counts"])
        for phase in ("upstream", "downstream")
    }
    revised["frontier_definition"] = (
        "nominal_trajectory_x_slice_local_frontier_root_cell_unique_v1"
    )
    revised["jump_tube_contract"] = {
        "profile": "trajectory_centered_jump_tube_v1",
        "nominal_centerline": str(nominal_centerline),
        "nominal_centerline_sha256": str(centerline["centerline_sha256"]),
        "x_min_m": float(centerline["x_min_m"]),
        "x_hard_max_m": float(centerline["x_hard_max_m"]),
        "effective_x_max_m": float(centerline["effective_centerline_max_x_m"]),
        "x_step_m": float(centerline["x_step_m"]),
        "centerline_point_count": int(centerline["point_count"]),
        "upstream_semantics": "pre-apex centerline-supported x slices",
        "downstream_semantics": "post-apex centerline-supported x slices AND root_vz_mps < 0",
        "post_landing_recovery_frontier_eligible": False,
        "selection_scope": "local_cross_section_per_x_slice_not_global_lowest_score",
    }
    revised["capability_geometry"] = {
        "summary": str(capability_geometry_summary),
        "summary_file_sha256": file_sha256(capability_geometry_summary),
        "summary_sha256": str(geometry_summary["summary_sha256"]),
        "resolution_sha256": str(geometry_summary["resolution_contract"]["resolution_sha256"]),
        "parent_diversity_profile": "root_geometry_v1",
        "max_parent_cells_per_phase": int(max_parent_cells_per_phase),
        "selection_audit": audit,
    }
    revised["protocol_revision"] = {
        "name": "trajectory_centered_x_balanced_frontier_v1",
        "purpose": (
            "widen one successful real jump trajectory slice-by-slice; replace global low-score "
            "newest-shell expansion and exclude post-landing/late downstream states before any "
            "new frontier outcomes are observed"
        ),
        "supersedes_plan": str(source_plan),
        "supersedes_plan_sha256": str(plan["plan_sha256"]),
        "changed_fields": [
            "anchors",
            "role_parent_group_counts",
            "frontier_definition",
            "jump_tube_contract",
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
    output.write_text(json.dumps(revised, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    report = {
        "schema": SCHEMA,
        "status": "completed_pre_outcome_revision",
        "source_plan": str(source_plan),
        "source_plan_sha256": str(plan["plan_sha256"]),
        "revised_plan": str(output),
        "revised_plan_sha256": str(revised["plan_sha256"]),
        "source_tube_manifest_sha256": str(tube.manifest["manifest_sha256"]),
        "capability_geometry_summary": str(capability_geometry_summary),
        "capability_resolution_sha256": str(geometry_summary["resolution_contract"]["resolution_sha256"]),
        "nominal_centerline": str(nominal_centerline),
        "nominal_centerline_sha256": str(centerline["centerline_sha256"]),
        "jump_tube_contract": dict(revised["jump_tube_contract"]),
        "selection_audit": audit,
        "test_data_used": False,
        "final_evaluation_data_used": False,
    }
    report["report_sha256"] = _canonical_sha256(report)
    sidecar = output.with_suffix(output.suffix + ".resolution.json")
    sidecar.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return report
