#!/usr/bin/env python3
"""Measure serial and batched continuation on existing TRAIN candidates, no PPO."""
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('catalog', 'acquisition', 'evaluator', 'output-dir'):
        p.add_argument('--'+flag, type=Path, required=True)
    p.add_argument('--batch-sizes', type=int, nargs='+')
    p.add_argument('--backend', choices=['device', 'vectorized'], default='vectorized')
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--shard-count', type=int, default=1)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--horizon', type=int, default=400)
    p.add_argument('--budget', type=int, required=True)
    p.add_argument('--timeout-seconds', type=int, default=900)
    p.add_argument('--gpu', default='0')
    p.add_argument('--capacity', action='store_true')
    p.add_argument('--capacity-worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--worlds', type=int, default=4096)
    p.add_argument('--ticks', type=int, default=64)
    p.add_argument('--memory-limit-mib', type=int, default=22000)
    a = p.parse_args()
    if a.batch_sizes is None:
        a.batch_sizes = [4096, 8192, 16384] if a.capacity else [8, 32, 128]
    if a.capacity_worker:
        from jit_dvgc.continuation_benchmark import capacity_worker
        return capacity_worker(a)
    if a.capacity:
        from jit_dvgc.continuation_benchmark import run_capacity
        result = run_capacity(Path(__file__).resolve().parents[2], a.output_dir, a)
        return 0 if result['status']=='completed' else 2
    from jit_dvgc.continuation_benchmark import run
    result = run(Path(__file__).resolve().parents[2], a.output_dir, catalog=a.catalog,
                 acquisition=a.acquisition, evaluator=a.evaluator, sizes=a.batch_sizes,
                 backend=a.backend, shard_index=a.shard_index, shard_count=a.shard_count,
                 seed=a.seed, horizon=a.horizon, budget=a.budget, timeout_seconds=a.timeout_seconds, gpu=a.gpu)
    return 0 if result['status']=='completed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
