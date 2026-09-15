#!/usr/bin/env python3
"""Prepare generic fresh-bank and stronger-core replacement configs for pi_k -> pi_(k+1)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.repair_acceptance import (
    prepare_core_replay_repair_config,
    prepare_repair_acceptance_predeclaration,
)


def _write_new(path: Path, payload) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    bank = sub.add_parser(
        "prepare-bank",
        help="predeclare a fresh baseline-only acceptance bank before replacement training",
    )
    bank.add_argument("--baseline-frozen-policy", type=Path, required=True)
    bank.add_argument("--target-tube", type=Path, required=True)
    bank.add_argument(
        "--consumed-gate",
        type=Path,
        action="append",
        required=True,
        help="previously exposed paired gate root; repeat for every consumed gate",
    )
    bank.add_argument("--output", type=Path, required=True)
    bank.add_argument("--acquisition-seed", type=int, required=True)
    bank.add_argument("--labeling-seed", type=int, required=True)
    bank.add_argument("--anchors-per-phase", type=int, default=10)
    bank.add_argument(
        "--minimum-anchors-per-phase",
        type=int,
        default=None,
        help=(
            "minimum fresh parent-unique anchors required per phase; when omitted, "
            "defaults to --minimum-negative-parent-groups-per-phase rather than "
            "the negative-state count"
        ),
    )
    bank.add_argument("--frontier-score-ceiling", type=float, default=1.0)
    bank.add_argument("--strengths", type=float, nargs="+", default=[0.15, 0.30, 0.50])
    bank.add_argument("--durations", type=int, nargs="+", default=[2, 4, 8])
    bank.add_argument(
        "--action-names",
        nargs="+",
        default=["steer", "rear_wheel_drive", "hip", "knee"],
    )
    bank.add_argument("--signs", type=int, nargs="+", default=[-1, 1])
    bank.add_argument("--active-action-dimensions", type=int, default=2)
    bank.add_argument("--minimum-negative-states-per-phase", type=int, default=10)
    bank.add_argument(
        "--minimum-negative-parent-groups-per-phase", type=int, default=3
    )

    training = sub.add_parser(
        "prepare-training",
        help=(
            "bind a newly locked fresh bank and strengthen retained-core replay "
            "without changing PPO, initialization, reset mixture, or Tube support"
        ),
    )
    training.add_argument("--base-config", type=Path, required=True)
    training.add_argument("--locked-acceptance-bank", type=Path, required=True)
    training.add_argument("--failed-gate-root", type=Path, required=True)
    training.add_argument("--core-probability", type=float, required=True)
    training.add_argument("--run-id", required=True)
    training.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "prepare-bank":
        minimum_anchors = args.minimum_anchors_per_phase
        if minimum_anchors is None:
            minimum_anchors = args.minimum_negative_parent_groups_per_phase
        if minimum_anchors < args.minimum_negative_parent_groups_per_phase:
            parser.error(
                "--minimum-anchors-per-phase cannot be smaller than "
                "--minimum-negative-parent-groups-per-phase"
            )
        payload = prepare_repair_acceptance_predeclaration(
            baseline_frozen_policy=args.baseline_frozen_policy,
            target_tube=args.target_tube,
            consumed_gate_roots=args.consumed_gate,
            acquisition_seed=args.acquisition_seed,
            labeling_seed=args.labeling_seed,
            anchors_per_phase=args.anchors_per_phase,
            minimum_anchors_per_phase=minimum_anchors,
            frontier_score_ceiling=args.frontier_score_ceiling,
            strengths=args.strengths,
            durations=args.durations,
            action_names=args.action_names,
            signs=args.signs,
            active_action_dimensions=args.active_action_dimensions,
            minimum_negative_states_per_phase=args.minimum_negative_states_per_phase,
            minimum_negative_parent_groups_per_phase=(
                args.minimum_negative_parent_groups_per_phase
            ),
        )
    else:
        payload = prepare_core_replay_repair_config(
            base_config_path=args.base_config,
            locked_acceptance_bank=args.locked_acceptance_bank,
            failed_gate_root=args.failed_gate_root,
            core_probability=args.core_probability,
            run_id=args.run_id,
        )
    _write_new(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
