#!/usr/bin/env python3
"""Stable, authorization-gated entrypoint for two-phase expert smoke runs."""
from __future__ import annotations

import argparse
import json
from typing import Any

from dvgc.phase_expert_training import (
    PHASE_EXPERT_PHASES,
    PhaseExpertRunSpec,
    run_phase_expert,
    validate_phase_expert_run_spec,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=PHASE_EXPERT_PHASES)
    parser.add_argument("--experiment-level", default="smoke", choices=("smoke",))
    parser.add_argument("--requested-total-transitions", type=int, required=True)
    parser.add_argument("--seed", type=int, default=710001)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--training-config", default="configs/phase_expert_smoke.json")
    parser.add_argument("--threshold-manifest", required=True)
    parser.add_argument("--authorization-manifest")
    parser.add_argument("--run", required=True)
    parser.add_argument("--descent-seed-bank")
    parser.add_argument("--descent-seed-manifest")
    parser.add_argument("--resume-run")
    parser.add_argument("--restore-checkpoint")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def spec_from_args(args: argparse.Namespace) -> PhaseExpertRunSpec:
    if bool(args.resume_run) != bool(args.restore_checkpoint):
        raise ValueError("--resume-run and --restore-checkpoint must be paired")
    return PhaseExpertRunSpec(
        phase=args.phase,
        experiment_level=args.experiment_level,
        requested_total_transitions=args.requested_total_transitions,
        seed=args.seed,
        config_path=args.config,
        training_config_path=args.training_config,
        threshold_manifest_path=args.threshold_manifest,
        authorization_manifest_path=args.authorization_manifest,
        output_dir=args.run,
        descent_seed_bank=args.descent_seed_bank,
        descent_seed_manifest=args.descent_seed_manifest,
        resume_run=args.resume_run,
        restore_checkpoint=args.restore_checkpoint,
    )


def _preflight_report(validated: Any) -> dict[str, Any]:
    budget = validated.interaction_budget
    return {
        "status": "preflight_pass",
        "phase": validated.spec.phase,
        "experiment_level": validated.spec.experiment_level,
        "requested_total_transitions": budget.training.requested_total_transitions,
        "effective_total_transitions": budget.training.effective_total_transitions,
        "brax_evaluation_transition_ceiling": budget.brax_evaluation_transition_ceiling,
        "fixed_evaluation_transition_ceiling": budget.fixed_evaluation_transition_ceiling,
        "combined_transition_ceiling": budget.combined_transition_ceiling,
        "threshold_manifest_canonical_hash": validated.thresholds.canonical_manifest_hash,
        "training_transitions_executed": 0,
        "evaluation_transitions_executed": 0,
    }


def main() -> int:
    args = build_parser().parse_args()
    spec = spec_from_args(args)
    validated = validate_phase_expert_run_spec(spec, preflight_only=args.preflight_only)
    report = _preflight_report(validated) if args.preflight_only else run_phase_expert(validated)
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
