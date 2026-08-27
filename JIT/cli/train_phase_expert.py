#!/usr/bin/env python3
"""Stable CLI for implemented Propulsion-Ascent smoke and formal training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.ppo import run_phase_u_smoke


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--formal", action="store_true")
    parser.add_argument("--restore-checkpoint", type=Path)
    parser.add_argument("--snapshot-bank", type=Path)
    parser.add_argument("--snapshot-catalog", type=Path)
    parser.add_argument("--actor-init-checkpoint", type=Path)
    parser.add_argument("--actor-init-config", type=Path)
    parser.add_argument("--eval-seeds", type=int, nargs="+")
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.phase not in {"propulsion_ascent", "descent_recovery"}:
        parser.error("unsupported phase")
    if args.restore_checkpoint is not None and not args.formal:
        parser.error("--restore-checkpoint is only valid with --formal")
    if args.smoke:
        if args.phase == "descent_recovery":
            if args.snapshot_bank is None and args.snapshot_catalog is None:
                parser.error("descent_recovery smoke requires --snapshot-bank or --snapshot-catalog")
            if args.actor_init_checkpoint is None or args.actor_init_config is None:
                parser.error("descent_recovery smoke requires actor initialization arguments")
            if not args.eval_seeds:
                parser.error("descent_recovery smoke requires --eval-seeds")
            from jit_dvgc.phase_d_smoke import run_phase_d_smoke

            report = run_phase_d_smoke(
                args.config,
                args.run_id,
                snapshot_bank=args.snapshot_bank,
                snapshot_catalog=args.snapshot_catalog,
                actor_init_checkpoint=args.actor_init_checkpoint,
                actor_init_config=args.actor_init_config,
                eval_seeds=tuple(args.eval_seeds),
            )
        else:
            report = run_phase_u_smoke(args.config, args.run_id)
    else:
        if args.phase != "propulsion_ascent":
            parser.error("formal descent_recovery training is not implemented")
        from jit_dvgc.formal_training import run_phase_u_formal

        report = run_phase_u_formal(
            args.config,
            args.run_id,
            restore_checkpoint=args.restore_checkpoint,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
