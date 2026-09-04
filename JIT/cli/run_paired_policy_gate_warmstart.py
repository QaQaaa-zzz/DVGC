#!/usr/bin/env python3
"""Prepare or run the paired gate for fresh or unified warm-start candidates.

This wrapper changes only config verification.  Rollout/evaluation logic is the
existing paired-policy gate unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jit_dvgc.analysis.paired_policy_gate as gate_mod
import jit_dvgc.unified_policy_freeze as freeze_mod

import train_unified_from_pi0 as actor_only
import train_unified_from_pi0_full as full_warm


_ORIGINAL_GATE_LOADER = gate_mod.load_unified_formal_config
_ORIGINAL_FREEZE_LOADER = freeze_mod.load_unified_formal_config


def _read_json(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _load_for_gate(path: Path):
    raw = _read_json(Path(path))
    initialization = raw.get("initialization", {})
    actor = initialization.get("actor")
    critic = initialization.get("critic")
    if actor in {"warm_start_pi_0", "warm_start_frozen_unified"} and critic == "fresh":
        return actor_only._load_warm_target_config(Path(path))
    if actor == "warm_start_pi_0" and critic == "warm_start_pi_0":
        return full_warm._load_warm_target_config(Path(path))
    return _ORIGINAL_GATE_LOADER(Path(path))


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

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--baseline-frozen-policy", type=Path, required=True)
    prepare.add_argument("--candidate-frozen-policy", type=Path, required=True)
    prepare.add_argument("--core-tube", type=Path, required=True)
    prepare.add_argument("--locked-boundary-bank", type=Path, required=True)
    prepare.add_argument("--gate-output-dir", type=Path, required=True)
    prepare.add_argument("--config-out", type=Path, required=True)
    prepare.add_argument("--minimum-candidate-success-parent-groups", type=int, required=True)
    prepare.add_argument("--protocol-seed", type=int, required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--config", type=Path, required=True)

    args = parser.parse_args()

    previous_gate = gate_mod.load_unified_formal_config
    previous_freeze = freeze_mod.load_unified_formal_config
    gate_mod.load_unified_formal_config = _load_for_gate
    freeze_mod.load_unified_formal_config = _load_for_gate
    try:
        if args.command == "prepare":
            config = gate_mod.prepare_locked_bank_gate_config(
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

        report = gate_mod.run_paired_policy_gate(args.config)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0
    finally:
        gate_mod.load_unified_formal_config = previous_gate
        freeze_mod.load_unified_formal_config = previous_freeze


if __name__ == "__main__":
    raise SystemExit(main())
