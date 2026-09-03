#!/usr/bin/env python3
"""Prepare/run outcome-blind frontier roles for automatic envelope iterations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading

import jax

from jit_dvgc.iterative_frontier_protocol import (
    canonical_sha256,
    prepare_frontier_plan,
    run_frontier_role,
)


LOCAL_HORIZON_V2_DURATIONS = (1, 2, 4, 8)
LEGACY_V1_DURATIONS = (4, 8, 16, 32)


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
    zero downstream candidate support.  It preserves parent/role assignment,
    policy identity, source Tube, action directions, perturbation strengths,
    acquisition/labeling seeds, and the 400-tick continuation horizon.  Only the
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


def _role_scale(plan_path: Path, role: str) -> tuple[int, int]:
    plan = _read_json(plan_path)
    anchors = sum(1 for row in plan.get("anchors", []) if row.get("role") == role)
    panel = plan.get("fixed_probe_panel", {})
    variants = (
        len(panel.get("action_names", []))
        * len(panel.get("signs", []))
        * len(panel.get("strengths", []))
        * len(panel.get("durations", []))
    )
    return anchors, anchors * variants


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
            snapshots = acquisition / "boundary_bank" / "snapshots"
            if snapshots.is_dir():
                try:
                    count = sum(1 for path in snapshots.iterdir() if path.is_dir())
                    detail = f" snapshots_written={count}"
                except OSError:
                    pass
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
    else:
        if jax.default_backend() != "gpu":
            raise RuntimeError("iterative frontier rollout requires the visible JAX GPU")
        anchors, maximum_probe_variants = _role_scale(args.plan, args.role)
        plan = _read_json(args.plan)
        panel = plan.get("fixed_probe_panel", {})
        print(
            f"[frontier:{args.role}] start anchors={anchors} "
            f"maximum_probe_variants={maximum_probe_variants} "
            f"durations={panel.get('durations')} strengths={panel.get('strengths')}",
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
