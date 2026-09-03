#!/usr/bin/env python3
"""Prepare/run outcome-blind frontier roles for automatic envelope iterations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading

import jax

from jit_dvgc.iterative_frontier_protocol import prepare_frontier_plan, run_frontier_role


def _read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
    while not stop.wait(30.0):
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
                    candidate_count = int(_read_json(catalog).get("candidate_count", 0))
                    detail = f" candidates={candidate_count} max_ticks_each=400"
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
            f"[frontier:{role}] heartbeat phase={phase}{detail}",
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
    else:
        if jax.default_backend() != "gpu":
            raise RuntimeError("iterative frontier rollout requires the visible JAX GPU")
        anchors, maximum_probe_variants = _role_scale(args.plan, args.role)
        print(
            f"[frontier:{args.role}] start anchors={anchors} "
            f"maximum_probe_variants={maximum_probe_variants}",
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
