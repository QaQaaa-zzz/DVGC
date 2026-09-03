#!/usr/bin/env python3
"""Execute large frontier continuation-label banks in independent GPU processes.

The supervisor path intentionally imports no JAX code. Each shard is a separate
Python process so CUDA/Warp/JAX allocations are released between shards. The
logical label protocol still uses the original global candidate indices and one
unchanged role-level protocol seed, and merge restores exact catalog order.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys


def _read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolved_shard_count(plan: Path, role_root: Path) -> tuple[int, int, int]:
    plan_payload = _read(plan)
    catalog = _read(Path(role_root) / "acquisition" / "catalog.json")
    candidate_count = int(catalog["candidate_count"])
    execution = plan_payload.get("label_execution", {})
    maximum = int(execution.get("max_candidates_per_independent_process", 930))
    if candidate_count <= 0 or maximum <= 0:
        raise ValueError("invalid candidate/shard execution count")
    shard_count = max(1, math.ceil(candidate_count / maximum))
    return candidate_count, maximum, shard_count


def _run_child(args: list[str]) -> None:
    print("[frontier-shards] " + " ".join(args), flush=True)
    subprocess.run(args, check=True)


def _run_all(plan: Path, role_root: Path, role: str) -> int:
    candidate_count, maximum, shard_count = _resolved_shard_count(plan, role_root)
    print(
        f"[frontier-shards] candidate_count={candidate_count} "
        f"max_per_process={maximum} shard_count={shard_count}",
        flush=True,
    )
    if shard_count <= 1:
        raise ValueError(
            "run-all is reserved for banks above the independent-process shard threshold"
        )

    script = Path(__file__).resolve()
    for index in range(shard_count):
        _run_child(
            [
                sys.executable,
                str(script),
                "run-shard",
                "--plan",
                str(plan),
                "--role-root",
                str(role_root),
                "--role",
                role,
                "--shard-index",
                str(index),
                "--shard-count",
                str(shard_count),
            ]
        )
    _run_child(
        [
            sys.executable,
            str(script),
            "merge",
            "--plan",
            str(plan),
            "--role-root",
            str(role_root),
            "--role",
            role,
            "--shard-count",
            str(shard_count),
        ]
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)

    all_parser = subs.add_parser("run-all")
    all_parser.add_argument("--plan", type=Path, required=True)
    all_parser.add_argument("--role-root", type=Path, required=True)
    all_parser.add_argument(
        "--role", choices=("train", "calibration", "acceptance"), required=True
    )

    shard = subs.add_parser("run-shard")
    shard.add_argument("--plan", type=Path, required=True)
    shard.add_argument("--role-root", type=Path, required=True)
    shard.add_argument(
        "--role", choices=("train", "calibration", "acceptance"), required=True
    )
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--shard-count", type=int, required=True)

    merge = subs.add_parser("merge")
    merge.add_argument("--plan", type=Path, required=True)
    merge.add_argument("--role-root", type=Path, required=True)
    merge.add_argument(
        "--role", choices=("train", "calibration", "acceptance"), required=True
    )
    merge.add_argument("--shard-count", type=int, required=True)

    args = parser.parse_args()

    if args.command == "run-all":
        return _run_all(args.plan, args.role_root, args.role)

    # Heavy imports are deliberately delayed so the run-all supervisor never
    # initializes a GPU runtime.
    from jit_dvgc.phase_specific_frontier import (
        merge_frontier_label_shards,
        run_frontier_label_shard,
    )

    if args.command == "run-shard":
        result = run_frontier_label_shard(
            plan_path=args.plan,
            role_root=args.role_root,
            role=args.role,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
    else:
        result = merge_frontier_label_shards(
            plan_path=args.plan,
            role_root=args.role_root,
            role=args.role,
            shard_count=args.shard_count,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
