#!/usr/bin/env python3
"""Complete shared-panel first-landing labels and plot pi_0..pi_3 envelopes."""
from __future__ import annotations
import argparse
from pathlib import Path


def main():
    from jit_dvgc.policy_comparison import DEFAULT_OUTPUT, run_comparison
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--scan-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--shard-size", type=int, default=200)
    parser.add_argument("--interaction-budget", type=int, default=5_000_000)
    parser.add_argument("--frozen-policy", type=Path, action="append", default=[])
    for flag, kind, default in (("worker", str, None), ("worker-output", Path, None),
                                ("job-index", int, 0), ("shard-index", int, 0), ("role", str, "train")):
        parser.add_argument("--" + flag, type=kind, default=default, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        from jit_dvgc.policy_comparison_runtime import run_worker
        return run_worker(args.output_dir, args.worker_output, kind=args.worker,
                          job_index=args.job_index, shard_index=args.shard_index, role=args.role)
    repo = Path(__file__).resolve().parents[2]
    report = run_comparison(repo, args.output_dir or repo / DEFAULT_OUTPUT, scan_root=args.scan_root,
        gpu=args.gpu, shard_size=args.shard_size, interaction_budget=args.interaction_budget,
        frozen_policies=args.frozen_policy)
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
