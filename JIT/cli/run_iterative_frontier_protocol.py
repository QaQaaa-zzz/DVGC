#!/usr/bin/env python3
"""Prepare/run outcome-blind frontier roles for automatic envelope iterations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading

import jax

from jit_dvgc.config import file_sha256
from jit_dvgc.iterative_frontier_protocol import (
    canonical_sha256,
    prepare_frontier_plan,
    run_frontier_role,
)
from jit_dvgc.phase_specific_frontier import (
    V3_REVISION_NAME,
    panel_variant_count,
    phase_probe_panels,
    run_phase_specific_frontier_role,
)


LOCAL_HORIZON_V2_DURATIONS = (1, 2, 4, 8)
LEGACY_V1_DURATIONS = (4, 8, 16, 32)
V3_DOWNSTREAM_STRENGTHS = (0.15, 0.30, 0.50)
V3_DOWNSTREAM_DURATIONS = (2, 4, 8)


def _read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _verify_plan_hash(plan: dict) -> None:
    declared = str(plan.get("plan_sha256", ""))
    base = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if len(declared) != 64 or canonical_sha256(base) != declared:
        raise ValueError("source frontier plan self-hash drift")


def _revise_local_horizon_v2(source: Path, output: Path) -> dict:
    """Create a new pre-outcome plan with a more local real-dynamics probe horizon.

    This is an explicit protocol revision after the v1 TRAIN role demonstrated
    zero downstream candidate support. It preserves parent/role assignment,
    policy identity, source Tube, action directions, perturbation strengths,
    acquisition/labeling seeds, and the 400-tick continuation horizon. Only the
    action-perturbation duration grid changes from 4/8/16/32 to 1/2/4/8 ticks.
    The old plan remains immutable and is referenced by SHA-256.
    """
    source = Path(source)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"revised frontier plan already exists: {output}")
    plan = _read_json(source)
    if not isinstance(plan, dict) or plan.get("schema") != "jit_iterative_frontier_plan_v1":
        raise ValueError("local-horizon revision requires a v1 iterative frontier plan")
    if plan.get("status") != "predeclared_before_frontier_outcomes":
        raise ValueError("source frontier plan status drift")
    _verify_plan_hash(plan)
    panel = plan.get("fixed_probe_panel")
    if not isinstance(panel, dict):
        raise ValueError("source frontier plan probe panel missing")
    durations = tuple(int(value) for value in panel.get("durations", ()))
    if durations != LEGACY_V1_DURATIONS:
        raise ValueError(
            "local-horizon-v2 revision is defined only from the legacy 4/8/16/32-tick panel"
        )
    if int(panel.get("max_label_ticks", -1)) != 400:
        raise ValueError("continuation-label horizon drift before local-horizon revision")

    revised = {key: value for key, value in plan.items() if key != "plan_sha256"}
    revised_panel = dict(panel)
    revised_panel["durations"] = list(LOCAL_HORIZON_V2_DURATIONS)
    revised["fixed_probe_panel"] = revised_panel
    revised["frontier_definition"] = (
        "newest_expansion_shell_only_lowest_score_parent_unique_local_horizon_v2"
    )
    revised["protocol_revision"] = {
        "name": "local_horizon_v2",
        "reason": (
            "legacy v1 TRAIN acquisition/labeling completed with zero downstream candidates; "
            "at 50 Hz the previous shortest perturbation duration was 4 ticks (80 ms), so the "
            "replacement panel resolves the local downstream frontier at 1/2/4/8 ticks "
            "(20/40/80/160 ms) before new v2 outcomes are observed"
        ),
        "supersedes_plan": str(source),
        "supersedes_plan_sha256": str(plan["plan_sha256"]),
        "changed_fields": ["fixed_probe_panel.durations", "frontier_definition"],
        "unchanged_contracts": [
            "selected_policy_identity",
            "source_tube_identity",
            "newest_shell_only_parent_pool",
            "parent_group_role_assignment",
            "train_calibration_acceptance_parent_disjointness",
            "action_names",
            "action_signs",
            "perturbation_strengths",
            "acquisition_seeds",
            "labeling_seeds",
            "continuation_max_label_ticks_400",
            "real_dynamics_only",
            "test_isolation",
        ],
        "automatic_repair": False,
        "revision_predeclared_before_v2_outcomes": True,
    }
    revised["plan_sha256"] = canonical_sha256(revised)
    _write_json(output, revised)
    return revised


def _validate_v2_diagnostic(path: Path, plan: dict) -> dict:
    diagnostic = _read_json(path)
    if (
        not isinstance(diagnostic, dict)
        or diagnostic.get("schema") != "jit_frontier_support_diagnostics_v1"
        or diagnostic.get("status") != "completed"
    ):
        raise ValueError("v3 revision requires a completed v2 support diagnostic")
    if diagnostic.get("artifact_role") != "post_failure_read_only_probe_support_diagnostic":
        raise ValueError("v2 diagnostic artifact role drift")
    if int(diagnostic.get("iteration", -1)) != int(plan["iteration"]):
        raise ValueError("v2 diagnostic iteration drift")
    if diagnostic.get("policy_actor_sha256") != plan.get("policy_actor_sha256"):
        raise ValueError("v2 diagnostic actor identity drift")
    if diagnostic.get("policy_payload_sha256") != plan.get("policy_payload_sha256"):
        raise ValueError("v2 diagnostic payload identity drift")
    by_phase = diagnostic.get("by_phase")
    if not isinstance(by_phase, dict):
        raise ValueError("v2 diagnostic phase summary missing")
    upstream = by_phase.get("upstream")
    downstream = by_phase.get("downstream")
    if not isinstance(upstream, dict) or not isinstance(downstream, dict):
        raise ValueError("v2 diagnostic requires both phase summaries")
    if int(upstream.get("positive_count", 0)) < 20 or int(upstream.get("negative_count", 0)) < 20:
        raise ValueError("v3 phase-specific revision expects already-bracketed upstream v2 support")
    if int(downstream.get("positive_count", 0)) < 20:
        raise ValueError("v3 revision requires meaningful downstream positive support in v2")
    if int(downstream.get("negative_count", -1)) != 0:
        raise ValueError("v3 two-axis revision is defined for the v2 downstream all-positive failure")
    return diagnostic


def _revise_phase_specific_two_axis_v3(
    source: Path,
    diagnostic_path: Path,
    output: Path,
) -> dict:
    """Create v3 after the failed v2 TRAIN diagnostic, before any v3 outcomes.

    Upstream keeps the exact v2 single-axis panel because it already brackets both
    continuation classes. Downstream alone adopts the historically successful
    two-axis sparse acquisition family with 0.15/0.30/0.50 strengths and 2/4/8
    tick durations. The parent-role split, pi_1, Tube_1, seeds, 400-tick label
    definition, and all TRAIN/CALIBRATION/ACCEPTANCE semantics remain unchanged.
    """
    source = Path(source)
    diagnostic_path = Path(diagnostic_path)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"v3 frontier plan already exists: {output}")
    plan = _read_json(source)
    if not isinstance(plan, dict) or plan.get("schema") != "jit_iterative_frontier_plan_v1":
        raise ValueError("v3 revision requires an iterative frontier plan")
    if plan.get("status") != "predeclared_before_frontier_outcomes":
        raise ValueError("source v2 frontier plan status drift")
    _verify_plan_hash(plan)
    prior = plan.get("protocol_revision")
    if not isinstance(prior, dict) or prior.get("name") != "local_horizon_v2":
        raise ValueError("phase-specific v3 revision must supersede local_horizon_v2")
    panel = plan.get("fixed_probe_panel")
    if not isinstance(panel, dict):
        raise ValueError("v2 fixed probe panel missing")
    if tuple(int(value) for value in panel.get("durations", ())) != LOCAL_HORIZON_V2_DURATIONS:
        raise ValueError("v2 duration panel drift before v3 revision")
    if tuple(float(value) for value in panel.get("strengths", ())) != (0.025, 0.05, 0.10):
        raise ValueError("v2 strength panel drift before v3 revision")
    if int(panel.get("max_label_ticks", -1)) != 400:
        raise ValueError("v2 continuation horizon drift before v3 revision")
    diagnostic = _validate_v2_diagnostic(diagnostic_path, plan)

    action_names = [str(value) for value in panel["action_names"]]
    signs = [int(value) for value in panel["signs"]]
    upstream_panel = {
        "action_names": action_names,
        "signs": signs,
        "strengths": [float(value) for value in panel["strengths"]],
        "durations": [int(value) for value in panel["durations"]],
        "active_action_dimensions": 1,
    }
    downstream_panel = {
        "action_names": action_names,
        "signs": signs,
        "strengths": list(V3_DOWNSTREAM_STRENGTHS),
        "durations": list(V3_DOWNSTREAM_DURATIONS),
        "active_action_dimensions": 2,
    }

    revised = {key: value for key, value in plan.items() if key != "plan_sha256"}
    revised["frontier_definition"] = (
        "newest_expansion_shell_parent_unique_phase_specific_boundary_bracketing_v3"
    )
    revised["phase_probe_panels"] = {
        "upstream": upstream_panel,
        "downstream": downstream_panel,
    }
    revised["label_execution"] = {
        "mode": "independent_process_shards_above_candidate_limit",
        "max_candidates_per_independent_process": 930,
        "historical_basis": (
            "the previous 3720-candidate continuation-label job OOMed monolithically "
            "and completed as four independent 930-candidate shards"
        ),
        "execution_only": True,
        "logical_candidate_order_unchanged": True,
        "logical_label_protocol_unchanged": True,
        "global_candidate_index_prng_identity_preserved": True,
    }
    downstream_summary = diagnostic["by_phase"]["downstream"]
    upstream_summary = diagnostic["by_phase"]["upstream"]
    revised["protocol_revision"] = {
        "name": V3_REVISION_NAME,
        "reason": (
            "v2 TRAIN acquired downstream support but produced zero downstream continuation "
            "negatives, while upstream already had both classes. Historical repair acceptance "
            "showed that stronger single-axis probing could still fail to expose useful failure "
            "support and that sparse two-axis probing successfully opened the frontier. Therefore "
            "v3 changes only downstream acquisition dimensionality/magnitude while retaining the "
            "successful upstream v2 panel."
        ),
        "evidence": {
            "v2_support_diagnostic": str(diagnostic_path),
            "v2_support_diagnostic_file_sha256": file_sha256(diagnostic_path),
            "v2_upstream": {
                "candidate_count": int(upstream_summary["candidate_count"]),
                "positive_count": int(upstream_summary["positive_count"]),
                "negative_count": int(upstream_summary["negative_count"]),
                "parent_group_count": int(upstream_summary["parent_group_count"]),
            },
            "v2_downstream": {
                "candidate_count": int(downstream_summary["candidate_count"]),
                "positive_count": int(downstream_summary["positive_count"]),
                "negative_count": int(downstream_summary["negative_count"]),
                "parent_group_count": int(downstream_summary["parent_group_count"]),
            },
        },
        "supersedes_plan": str(source),
        "supersedes_plan_sha256": str(plan["plan_sha256"]),
        "supersedes_revision_name": "local_horizon_v2",
        "changed_fields": [
            "frontier_definition",
            "phase_probe_panels",
            "label_execution",
        ],
        "upstream_change": "none_relative_to_v2_probe_panel",
        "downstream_change": (
            "single_axis strengths=0.025/0.05/0.10 durations=1/2/4/8 -> "
            "two_axis_sparse strengths=0.15/0.30/0.50 durations=2/4/8"
        ),
        "unchanged_contracts": [
            "selected_pi1_identity",
            "Tube1_identity",
            "newest_shell_only_parent_pool",
            "parent_group_role_assignment",
            "train_calibration_acceptance_parent_disjointness",
            "role_level_acquisition_seeds",
            "labeling_seeds",
            "continuation_max_label_ticks_400",
            "continuation_success_definition",
            "real_dynamics_only",
            "C1_support_thresholds",
            "Tube2_retains_all_Tube1",
            "pi2_75_25_retained_newest_replay",
            "test_jce_jel_isolation",
        ],
        "automatic_repair": False,
        "revision_predeclared_before_v3_outcomes": True,
    }
    revised["plan_sha256"] = canonical_sha256(revised)
    _write_json(output, revised)
    return revised


def _role_scale(plan_path: Path, role: str) -> tuple[int, int, dict]:
    plan = _read_json(plan_path)
    role_anchors = [row for row in plan.get("anchors", []) if row.get("role") == role]
    revision = plan.get("protocol_revision", {})
    if isinstance(revision, dict) and revision.get("name") == V3_REVISION_NAME:
        panels = phase_probe_panels(plan)
        by_phase = {}
        maximum = 0
        for phase in ("upstream", "downstream"):
            anchors = sum(1 for row in role_anchors if row.get("phase") == phase)
            variants = panel_variant_count(panels[phase])
            by_phase[phase] = {
                "anchors": anchors,
                "variants_per_anchor": variants,
                "maximum_probe_variants": anchors * variants,
                "panel": panels[phase],
            }
            maximum += anchors * variants
        return len(role_anchors), maximum, by_phase

    panel = plan.get("fixed_probe_panel", {})
    variants = (
        len(panel.get("action_names", []))
        * len(panel.get("signs", []))
        * len(panel.get("strengths", []))
        * len(panel.get("durations", []))
    )
    return len(role_anchors), len(role_anchors) * variants, {"fixed": panel}


def _heartbeat(stop: threading.Event, output_dir: Path, role: str) -> None:
    elapsed = 0
    while not stop.wait(30.0):
        elapsed += 30
        acquisition = output_dir / "acquisition"
        labels = output_dir / "labels"
        manifest = output_dir / "role_manifest.json"
        phase = "startup"
        detail = ""
        if manifest.is_file():
            phase = "finalizing"
        elif labels.exists():
            phase = "labeling"
            catalog = acquisition / "catalog.json"
            if catalog.is_file():
                try:
                    catalog_payload = _read_json(catalog)
                    candidate_count = int(catalog_payload.get("candidate_count", 0))
                    entries = catalog_payload.get("entries", [])
                    up = sum(1 for row in entries if row.get("phase") == "upstream")
                    down = sum(1 for row in entries if row.get("phase") == "downstream")
                    detail = (
                        f" candidates={candidate_count} upstream={up} downstream={down} "
                        "max_ticks_each=400"
                    )
                except Exception:
                    pass
        elif acquisition.exists():
            phase = "acquisition"
            catalog = acquisition / "catalog.json"
            if catalog.is_file():
                try:
                    payload = _read_json(catalog)
                    detail = (
                        f" candidates={payload.get('candidate_count', 0)} "
                        f"upstream={payload.get('phase_candidate_counts', {}).get('upstream', 0)} "
                        f"downstream={payload.get('phase_candidate_counts', {}).get('downstream', 0)}"
                    )
                except Exception:
                    pass
            else:
                completed_phases = []
                for name in ("upstream", "downstream"):
                    if (acquisition / f"phase_{name}" / "catalog.json").is_file():
                        completed_phases.append(name)
                if completed_phases:
                    detail = f" completed_phase_acquisitions={','.join(completed_phases)}"
        print(
            f"[frontier:{role}] heartbeat elapsed_s={elapsed} phase={phase}{detail}",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="command", required=True)

    prepare = subs.add_parser("prepare-plan")
    prepare.add_argument("--selected-policy", type=Path, required=True)
    prepare.add_argument("--source-tube", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--max-parent-groups-per-phase", type=int, default=15)

    revise = subs.add_parser(
        "revise-plan-local-horizon-v2",
        help="predeclare a new local-duration frontier plan while preserving the failed v1 plan",
    )
    revise.add_argument("--source-plan", type=Path, required=True)
    revise.add_argument("--output", type=Path, required=True)

    revise_v3 = subs.add_parser(
        "revise-plan-phase-specific-two-axis-v3",
        help=(
            "predeclare upstream-v2/downstream-two-axis v3 from a completed read-only "
            "v2 support diagnostic"
        ),
    )
    revise_v3.add_argument("--source-plan", type=Path, required=True)
    revise_v3.add_argument("--v2-support-diagnostic", type=Path, required=True)
    revise_v3.add_argument("--output", type=Path, required=True)

    role = subs.add_parser("run-role")
    role.add_argument("--plan", type=Path, required=True)
    role.add_argument(
        "--role",
        choices=("train", "calibration", "acceptance"),
        required=True,
    )
    role.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "prepare-plan":
        result = prepare_frontier_plan(
            selected_policy=args.selected_policy,
            source_tube=args.source_tube,
            output=args.output,
            max_parent_groups_per_phase=args.max_parent_groups_per_phase,
        )
    elif args.command == "revise-plan-local-horizon-v2":
        result = _revise_local_horizon_v2(args.source_plan, args.output)
    elif args.command == "revise-plan-phase-specific-two-axis-v3":
        result = _revise_phase_specific_two_axis_v3(
            args.source_plan,
            args.v2_support_diagnostic,
            args.output,
        )
    else:
        if jax.default_backend() != "gpu":
            raise RuntimeError("iterative frontier rollout requires the visible JAX GPU")
        anchors, maximum_probe_variants, panel_description = _role_scale(
            args.plan, args.role
        )
        print(
            f"[frontier:{args.role}] start anchors={anchors} "
            f"maximum_probe_variants={maximum_probe_variants} "
            f"probe={json.dumps(panel_description, sort_keys=True)}",
            flush=True,
        )
        stop = threading.Event()
        heartbeat = threading.Thread(
            target=_heartbeat,
            args=(stop, args.output_dir, args.role),
            daemon=True,
        )
        heartbeat.start()
        try:
            plan = _read_json(args.plan)
            revision = plan.get("protocol_revision")
            if isinstance(revision, dict) and revision.get("name") == V3_REVISION_NAME:
                result = run_phase_specific_frontier_role(
                    plan_path=args.plan,
                    role=args.role,
                    output_dir=args.output_dir,
                )
            else:
                result = run_frontier_role(
                    plan_path=args.plan,
                    role=args.role,
                    output_dir=args.output_dir,
                )
        finally:
            stop.set()
            heartbeat.join(timeout=2.0)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
