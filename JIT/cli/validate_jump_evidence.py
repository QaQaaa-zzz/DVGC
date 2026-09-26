#!/usr/bin/env python3
"""Run a small GPU evidence audit and package the results for review. No training."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-policy", type=Path, help="default: uniquely identified Round1 pi_0")
    parser.add_argument("--evaluator-frozen-policy", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--per-phase", type=int, default=3)
    parser.add_argument("--shard-size", type=int, default=2)
    parser.add_argument("--prefix-seed", type=int, default=9400001)
    parser.add_argument("--label-seed", type=int, default=9521602)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--interaction-budget", type=int, default=20000)
    for flag, kind, default in (("worker", str, None), ("plan", Path, None), ("worker-output", Path, None),
                                ("evaluator-index", int, 0), ("sample-index", int, 0),
                                ("shard-index", int, 0), ("shard-count", int, 1)):
        parser.add_argument("--" + flag, type=kind, default=default, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        from jit_dvgc.jump_evidence_runtime import run_worker
        if args.worker not in {"capture", "replay", "serial", "shard", "merge"}:
            parser.error("unknown audit worker")
        return run_worker(args.plan, args.worker_output, kind=args.worker, evaluator_index=args.evaluator_index,
                          sample_index=args.sample_index, shard_index=args.shard_index, shard_count=args.shard_count)
    from jit_dvgc.jump_evidence_validation import run_validation
    repo = Path(__file__).resolve().parents[2]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = args.output_dir or repo / f"JIT/runs/evidence_validation/jump_evidence_{stamp}_{os.getpid()}"
    report = run_validation(repo, output, frozen_policy=args.frozen_policy,
        evaluators=args.evaluator_frozen_policy, per_phase=args.per_phase, shard_size=args.shard_size,
        prefix_seed=args.prefix_seed, label_seed=args.label_seed, atol=args.atol, rtol=args.rtol,
        interaction_budget=args.interaction_budget, gpu=args.gpu)
    return 0 if report["status"] == "passed_on_sampled_states" else 2


if __name__ == "__main__":
    raise SystemExit(main())
