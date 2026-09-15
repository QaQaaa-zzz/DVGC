#!/usr/bin/env python3
"""Prepare or execute the generic paired pi_k -> pi_(k+1) acceptance gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.analysis import prepare_locked_bank_gate_config, run_paired_policy_gate


def _write_json(path: Path, payload) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"paired gate config already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help=(
            "bind frozen pi_k/pi_(k+1), source core Tube, and a pre-training "
            "locked negative bank into an immutable v2 gate config"
        ),
    )
    prepare.add_argument("--baseline-frozen-policy", type=Path, required=True)
    prepare.add_argument("--candidate-frozen-policy", type=Path, required=True)
    prepare.add_argument("--core-tube", type=Path, required=True)
    prepare.add_argument("--locked-boundary-bank", type=Path, required=True)
    prepare.add_argument("--gate-output-dir", type=Path, required=True)
    prepare.add_argument("--config-out", type=Path, required=True)
    prepare.add_argument(
        "--minimum-candidate-success-parent-groups",
        type=int,
        required=True,
        help="predeclared boundary-gain parent-group minimum; no post-hoc default",
    )
    prepare.add_argument("--protocol-seed", type=int, required=True)

    run = subparsers.add_parser("run", help="execute one already-predeclared gate")
    run.add_argument("--config", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "prepare":
        config = prepare_locked_bank_gate_config(
            baseline_frozen_manifest=args.baseline_frozen_policy,
            candidate_frozen_manifest=args.candidate_frozen_policy,
            core_tube_path=args.core_tube,
            locked_bank_path=args.locked_boundary_bank,
            output_dir=args.gate_output_dir,
            minimum_candidate_success_parent_groups=(
                args.minimum_candidate_success_parent_groups
            ),
            protocol_seed=args.protocol_seed,
        )
        _write_json(args.config_out, config)
        print(json.dumps(config, indent=2, sort_keys=True, allow_nan=False))
        return 0

    report = run_paired_policy_gate(args.config)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
