#!/usr/bin/env python3
"""Freeze a completed pi0-warm-start unified policy.

This is only a compatibility wrapper for the two warm-start experiment configs.
It does not retrain or evaluate anything.  The standard freeze implementation is
reused after selecting the same config loader that was used for the matching
training run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jit_dvgc.unified_policy_freeze as freeze_mod

import train_unified_from_pi0 as actor_only
import train_unified_from_pi0_full as full_warm


def _read_json(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--formal-report", type=Path)
    args = parser.parse_args()

    raw = _read_json(args.config)
    initialization = raw.get("initialization", {})
    actor_mode = initialization.get("actor")
    critic_mode = initialization.get("critic")
    if actor_mode != "warm_start_pi_0":
        raise ValueError("this wrapper only freezes pi0 warm-start runs")
    if critic_mode == "fresh":
        loader = actor_only._load_warm_target_config
    elif critic_mode == "warm_start_pi_0":
        loader = full_warm._load_warm_target_config
    else:
        raise ValueError("unsupported warm-start critic mode")

    previous = freeze_mod.load_unified_formal_config
    freeze_mod.load_unified_formal_config = loader
    try:
        manifest = freeze_mod.freeze_unified_policy(
            args.output_dir,
            config_path=args.config,
            checkpoint=args.checkpoint,
            iteration=args.iteration,
            formal_report=args.formal_report,
        )
    finally:
        freeze_mod.load_unified_formal_config = previous

    print(args.output_dir / "frozen_unified_policy.json")
    print(manifest["policy"]["name"])
    print(manifest["policy"]["payload_sha256"])
    print(manifest["policy"]["actor_sha256"])
    print(manifest["policy"]["critic_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
