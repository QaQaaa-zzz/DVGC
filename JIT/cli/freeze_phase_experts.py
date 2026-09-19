#!/usr/bin/env python3
"""Freeze the selected Phase U and Phase D checkpoints by identity and hash."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.expert_freeze import freeze_phase_experts


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pi-up-config", type=Path, required=True)
    p.add_argument("--pi-up-checkpoint", type=Path, required=True)
    p.add_argument("--pi-down-config", type=Path, required=True)
    p.add_argument("--pi-down-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    report = freeze_phase_experts(
        args.output_dir,
        pi_up_config=args.pi_up_config,
        pi_up_checkpoint=args.pi_up_checkpoint,
        pi_down_config=args.pi_down_config,
        pi_down_checkpoint=args.pi_down_checkpoint,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
