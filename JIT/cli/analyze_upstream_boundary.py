#!/usr/bin/env python3
"""Analyze one completed TRAIN V_up boundary pilot without new rollouts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.upstream_boundary_analysis import write_boundary_analysis


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--audit", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = write_boundary_analysis(args.catalog, args.labels, args.output, audit_path=args.audit)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
