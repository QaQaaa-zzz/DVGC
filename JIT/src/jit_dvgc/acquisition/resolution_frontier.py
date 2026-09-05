"""Trajectory-centered frontier-plan revision for causal JIT.

The historical frontier used low-score states from the newest Soft-Tube shell as
physical RSI reset anchors. That is no longer the prospective Jump-Capability
method. A continuation-successful RSI state does not prove that the robot can
reach that state from the ground.

The active revision uses the locked successful nominal centerline as a complete
0.1 m longitudinal scaffold. Every centerline slice is predeclared before
outcomes. Five deterministic proposal families are created per slice following
TRAIN/TRAIN/TRAIN/CALIBRATION/ACCEPTANCE. These proposal anchors are geometric
identities only; they are never used as physical reset states. The causal role
runner starts every acquisition attempt at the authoritative fixed ground
jump-start reset and reaches the target slice only through env.step.

The source Soft Tube and its resolution-aware geometry remain useful as replay
support and occupancy context, but they do not establish forward reachability.
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
from ..analysis.nominal_jump_centerline import load_nominal_jump_centerline
from ..config import file_sha256
from ..soft_tube import load_soft_tube


SCHEMA = "jit_jump_start_trajectory_frontier_plan_revision_v3"
ROLE_PATTERN = ("train", "train", "train", "calibration", "acceptance")
ROLES = ("train", "calibration", "acceptance")
JUMP_X_STEP_M = 0.1
CAUSAL_LOOKBACKS_M = (0.1, 0.2, 0.3)
CAUSAL_FORWARD_MAX_TICKS = 400
VARIANT_PARTITION_MODULUS = len(ROLE_PATTERN)


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def configure_causal_probe_panel(
    source_panel: Mapping[str, Any],
    *,
    causal_lookbacks_m: Sequence[float] = CAUSAL_LOOKBACKS_M,
    strengths: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Lock a bounded perturbation grid and its per-role batch size."""
    lookbacks = tuple(float(value) for value in causal_lookbacks_m)
    strength_values = tuple(
        float(value)
        for value in (
            source_panel.get("strengths", ()) if strengths is None else strengths
        )
    )
    if (
        not lookbacks
        or len(set(lookbacks)) != len(lookbacks)
        or any(value <= 0.0 or value > 1.0 for value in lookbacks)
    ):
        raise ValueError("causal lookbacks must be unique and within (0, 1] m")
    if (
        not strength_values
        or len(set(strength_values)) != len(strength_values)
        or any(value <= 0.0 or value > 1.0 for value in strength_values)
    ):
        raise ValueError("causal action strengths must be unique and within (0, 1]")
    action_names = tuple(str(value) for value in source_panel.get("action_names", ()))
    signs = tuple(int(value) for value in source_panel.get("signs", ()))
    if not action_names or not signs:
        raise ValueError("causal perturbation action directions are missing")
    variant_count = len(lookbacks) * len(strength_values) * len(action_names) * len(signs)
    if variant_count % VARIANT_PARTITION_MODULUS:
        raise ValueError("causal variant grid must partition equally across role families")

    panel = dict(source_panel)
    panel["acquisition_mode"] = "jump_start_connected_causal_rollout_v1"
    panel["causal_lookbacks_m"] = list(lookbacks)
    panel["strengths"] = list(strength_values)
    panel["causal_forward_max_ticks"] = CAUSAL_FORWARD_MAX_TICKS
    panel["variant_partition_modulus"] = VARIANT_PARTITION_MODULUS
    panel["variant_count"] = variant_count
    panel["variants_per_role_family"] = variant_count // VARIANT_PARTITION_MODULUS
    panel["legacy_duration_field_used_by_causal_acquisition"] = False
    return panel


def _verify_hash(payload: Mapping[str, Any], field: str) -> None:
    declared = str(payload.get(field, ""))
    base = {key: value for key, value in payload.items() if key != field}
    if len(declared) != 64 or _canonical_sha256(base) != declared:
        raise ValueError(f"{field} self-hash drift")


def _x_bin(value: float) -> int:
    return int(round(float(value) / JUMP_X_STEP_M))


def _phase_for_centerline_point(point: Mapping[str, Any]) -> str | None:
    semantics = str(point["phase_semantics"])
    if semantics == "upstream":
        return "upstream"
    if semantics == "downstream":
        return "downstream"
    if semantics == "apex":
        # The exact Apex scaffold point is treated as the terminal upstream
        # slice. The first strictly post-Apex descending slice begins downstream.
        return "upstream"
    return None


def _source_slice_context(
    projected: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], dict[str, int]]:
    raw = Counter((str(row["phase"]), int(row["x_bin"])) for row in projected)
    cells: dict[tuple[str, int], set[str]] = {}
    for row in projected:
        key = (str(row["phase"]), int(row["x_bin"]))
        cells.setdefault(key, set()).add(str(row["root_geometry_cell_id"]))
    result: dict[tuple[str, int], dict[str, int]] = {}
    for key, count in raw.items():
        result[key] = {
            "source_raw_snapshot_count": int(count),
            "source_unique_root_geometry_cell_count": len(cells.get(key, set())),
        }
    return result


def build_centerline_slice_anchors(
    *,
    centerline: Mapping[str, Any],
    projected_entries: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create five pre-outcome proposal families for every usable centerline slice."""
    context = _source_slice_context(projected_entries)
    anchors: list[dict[str, Any]] = []
    by_phase = {
        "upstream": Counter(),
        "downstream": Counter(),
    }
    slice_counts = Counter()

    for point in centerline["points"]:
        phase = _phase_for_centerline_point(point)
        if phase is None:
            continue
        if phase == "downstream" and float(point["root_vz_mps"]) >= 0.0:
            raise ValueError("locked downstream centerline contains a non-descending point")
        phase_index = 0 if phase == "upstream" else 1
        x_target = float(point["x_target_m"])
        x_bin = _x_bin(x_target)
        slice_counts[phase] += 1
        source_context = context.get(
            (phase, x_bin),
            {
                "source_raw_snapshot_count": 0,
                "source_unique_root_geometry_cell_count": 0,
            },
        )
        for family, role in enumerate(ROLE_PATTERN):
            group = (
                f"causal_centerline_{phase}_x{x_bin:04d}_family{family}_{role}"
            )
            anchors.append(
                {
                    "role": role,
                    "phase": phase,
                    "phase_index": phase_index,
                    # These indices are deliberately sentinel values: causal
                    # acquisition MUST NOT reset a Soft-Tube entry.
                    "entry_index": -1,
                    "global_index": -1,
                    "state_sha256": str(point["physical_state_sha256"]),
                    "parent_group_id": group,
                    "value_score": 0.0,
                    "sampling_weight": 1.0,
                    "proposal_family_index": int(family),
                    "x_target_m": x_target,
                    "x_bin": int(x_bin),
                    "x_center_m": x_target,
                    "centerline_frame_index": int(point["frame_index"]),
                    "centerline_phase_semantics": str(point["phase_semantics"]),
                    "centerline_state_sha256": str(point["physical_state_sha256"]),
                    "proposal_anchor_is_physical_reset": False,
                    **source_context,
                }
            )
            by_phase[phase][role] += 1

    for phase in ("upstream", "downstream"):
        if int(slice_counts[phase]) <= 0:
            raise ValueError(f"locked centerline has no {phase} slices")
        if by_phase[phase]["train"] < 3:
            raise ValueError(f"causal centerline role split has insufficient TRAIN support in {phase}")
        if by_phase[phase]["calibration"] < 1 or by_phase[phase]["acceptance"] < 1:
            raise ValueError(f"causal centerline role split missing holdout roles in {phase}")

    audit = {
        "selection": "every_locked_centerline_x_slice_times_five_disjoint_proposal_families",
        "all_centerline_slices_probed": True,
        "proposal_families_per_slice": len(ROLE_PATTERN),
        "role_pattern": list(ROLE_PATTERN),
        "source_tube_states_used_as_physical_resets": False,
        "centerline_slice_counts": {
            phase: int(slice_counts[phase]) for phase in ("upstream", "downstream")
        },
        "role_counts": {
            phase: {role: int(by_phase[phase][role]) for role in ROLES}
            for phase in ("upstream", "downstream")
        },
        "anchor_count": len(anchors),
    }
    return anchors, audit


def select_resolution_distinct_anchors(
    *,
    tube_entries: Sequence[Mapping[str, Any]],
    projected_entries: Sequence[Mapping[str, Any]],
    core_retained_count: int,
    max_parent_cells_per_phase: int = 25,
    nominal_centerline: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compatibility wrapper retained for tests and old callers.

    Under the active method a locked centerline is mandatory and source-Tube
    entries are not physical acquisition anchors. The legacy arguments are
    validated only for identity/shape compatibility, then ignored as reset
    sources.
    """
    if nominal_centerline is None:
        raise ValueError("causal trajectory-centered frontier requires a locked centerline")
    if len(tube_entries) != len(projected_entries):
        raise ValueError("Tube and capability projection entry counts differ")
    if int(core_retained_count) <= 0 or int(core_retained_count) > len(tube_entries):
        raise ValueError("invalid source Tube core-retained count")
    if int(max_parent_cells_per_phase) <= 0:
        raise ValueError("max_parent_cells_per_phase must be positive")
    return build_centerline_slice_anchors(
        centerline=nominal_centerline,
        projected_entries=projected_entries,
    )


def revise_frontier_plan_for_resolution_cells(
    *,
    source_plan: Path,
    source_tube: Path,
    capability_geometry_summary: Path,
    nominal_centerline: Path,
    output: Path,
    max_parent_cells_per_phase: int = 25,
    proposal_frozen_policy: Path | None = None,
    continuation_frozen_policies: Sequence[Path] = (),
    causal_lookbacks_m: Sequence[float] = CAUSAL_LOOKBACKS_M,
    strengths: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Replace an unrevised iterative plan with the causal centerline plan."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"causal trajectory frontier plan already exists: {output}")

    plan = json.loads(Path(source_plan).read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or plan.get("schema") != "jit_iterative_frontier_plan_v1":
        raise ValueError("causal revision requires an iterative frontier plan")
    if plan.get("status") != "predeclared_before_frontier_outcomes":
        raise ValueError("frontier plan is no longer outcome-blind/predeclared")
    _verify_hash(plan, "plan_sha256")
    if tuple(plan.get("role_pattern", ())) != ROLE_PATTERN:
        raise ValueError("frontier role pattern drift before causal revision")

    tube = load_soft_tube(Path(source_tube))
    if str(tube.manifest["manifest_sha256"]) != str(plan["source_tube_manifest_sha256"]):
        raise ValueError("causal revision source Tube identity drift")
    if len(tube.entries) != int(plan["source_tube_entry_count"]):
        raise ValueError("causal revision source Tube entry-count drift")

    geometry_summary, projected = load_projected_capability_entries(
        Path(capability_geometry_summary)
    )
    if geometry_summary.get("schema") != CAPABILITY_TUBE_SCHEMA:
        raise ValueError("capability geometry schema drift")
    if geometry_summary.get("tube_manifest_sha256") != tube.manifest["manifest_sha256"]:
        raise ValueError("capability geometry was not built from the frontier source Tube")
    if abs(
        float(geometry_summary["resolution_contract"]["x_slice_width_m"])
        - JUMP_X_STEP_M
    ) > 1.0e-12:
        raise ValueError("causal trajectory frontier requires 0.1 m x slices")

    centerline = load_nominal_jump_centerline(Path(nominal_centerline))
    anchors, audit = build_centerline_slice_anchors(
        centerline=centerline,
        projected_entries=projected,
    )

    revised = {key: value for key, value in plan.items() if key != "plan_sha256"}
    revised["anchors"] = anchors
    revised["role_parent_group_counts"] = {
        phase: dict(audit["role_counts"][phase])
        for phase in ("upstream", "downstream")
    }
    revised["frontier_definition"] = (
        "locked_centerline_every_0p1m_slice_jump_start_connected_causal_forward_expansion_v3"
    )
    panel = configure_causal_probe_panel(
        revised["fixed_probe_panel"],
        causal_lookbacks_m=causal_lookbacks_m,
        strengths=strengths,
    )
    revised["fixed_probe_panel"] = panel
    revised["jump_tube_contract"] = {
        "profile": "conditional_jump_start_trajectory_tube_v3",
        "nominal_centerline": str(nominal_centerline),
        "nominal_centerline_sha256": str(centerline["centerline_sha256"]),
        "jump_start_state_sha256": str(centerline["jump_start_state_sha256"]),
        "natural_start_connected": False,
        "x_min_m": float(centerline["x_min_m"]),
        "x_hard_max_m": float(centerline["x_hard_max_m"]),
        "effective_x_max_m": float(centerline["effective_centerline_max_x_m"]),
        "x_step_m": float(centerline["x_step_m"]),
        "centerline_point_count": int(centerline["point_count"]),
        "centerline_recomputed_each_iteration": False,
        "all_centerline_slices_probed": True,
        "source_tube_states_used_as_physical_resets": False,
        "acquisition_mode": "jump_start_connected_causal_rollout_v1",
        "reachability_requirement": "fixed_jump_start_connected_forward_env_step_only",
        "rsi_may_establish_forward_reachability": False,
        "rsi_role": "continuation_evaluation_only_after_candidate_is_forward_reached",
        "upstream_semantics": "pre-Apex target slice reached from the fixed jump start",
        "downstream_semantics": (
            "post-Apex target slice reached from the fixed jump start AND root_vz_mps < 0 AND pre-contact"
        ),
        "post_landing_recovery_frontier_eligible": False,
        "selection_scope": "every_locked_centerline_slice_not_global_lowest_score",
    }
    if proposal_frozen_policy is not None or continuation_frozen_policies:
        if proposal_frozen_policy is None:
            raise ValueError("policy-family frontier requires a proposal policy")
        if len(continuation_frozen_policies) != 3:
            raise ValueError("policy-family frontier requires pi_0/pi_1/pi_2 evaluators")
        revised["jump_tube_contract"].update(
            {
                "proposal_frozen_policy": str(proposal_frozen_policy),
                "continuation_frozen_policies": [
                    str(path) for path in continuation_frozen_policies
                ],
                "continuation_success_criterion": (
                    "first_valid_landing_before_physical_failure"
                ),
                "post_landing_recovery_required": False,
                "continuation_policy_aggregation": "positive_if_any_policy_succeeds",
            }
        )
    revised["capability_geometry"] = {
        "summary": str(capability_geometry_summary),
        "summary_file_sha256": file_sha256(capability_geometry_summary),
        "summary_sha256": str(geometry_summary["summary_sha256"]),
        "resolution_sha256": str(
            geometry_summary["resolution_contract"]["resolution_sha256"]
        ),
        "source_control_tube_is_reachability_proof": False,
        "selection_audit": audit,
    }
    revised["protocol_revision"] = {
        "name": "conditional_jump_start_trajectory_centered_frontier_v3",
        "purpose": (
            "identify conditional Jump capability as fixed-jump-start-forward-reachable AND continuation-viable; "
            "probe every locked 0.1 m centerline slice with disjoint pre-outcome proposal "
            "families; never use RSI to manufacture reachability"
        ),
        "supersedes_plan": str(source_plan),
        "supersedes_plan_sha256": str(plan["plan_sha256"]),
        "changed_fields": [
            "anchors",
            "role_parent_group_counts",
            "frontier_definition",
            "fixed_probe_panel",
            "jump_tube_contract",
            "capability_geometry",
        ],
        "unchanged_contracts": [
            "selected_policy_identity",
            "source_tube_identity",
            "outcome_blind_role_assignment",
            "role_seeds",
            "reward_physics_action_semantics",
            "test_isolation",
        ],
        "outcomes_observed_before_revision": False,
    }
    if proposal_frozen_policy is not None:
        revised["protocol_revision"]["proposal_controller"] = "pi_0"
        revised["protocol_revision"]["continuation_controller_family"] = [
            "pi_0",
            "pi_1",
            "pi_2",
        ]
        revised["protocol_revision"]["continuation_success"] = (
            "first_valid_landing_by_any_controller"
        )
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
        "nominal_centerline": str(nominal_centerline),
        "nominal_centerline_sha256": str(centerline["centerline_sha256"]),
        "jump_tube_contract": dict(revised["jump_tube_contract"]),
        "selection_audit": audit,
        "legacy_max_parent_cells_per_phase_argument": int(max_parent_cells_per_phase),
        "legacy_max_parent_cells_per_phase_controls_causal_slice_count": False,
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
