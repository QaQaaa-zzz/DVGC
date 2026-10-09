"""Bounded, fresh-process execution comparison on an existing TRAIN catalog.

This produces engineering evidence, never a new acquisition or policy admission.
Every attempt is reserved before launch; timeout/failed attempts retain the ceiling.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .jump_evidence_validation import read, write, file_sha


def compare_labels(reference, candidate):
    if len(reference) != len(candidate):
        raise ValueError('label count mismatch')
    identities = ('candidate_index', 'state_sha256', 'snapshot_context_sha256', 'label_protocol_sha256')
    endpoints = ('label', 'outcome_class', 'physical_failure', 'timeout', 'final_active_phase')
    differences = []
    for a, b in zip(reference, candidate, strict=True):
        if any(k not in a or k not in b or a[k] != b[k] for k in identities):
            raise ValueError('label identity mismatch')
        for k in (*endpoints, 'environment_interactions'):
            if k not in a or k not in b:
                raise ValueError('missing outcome field')
        changes = {k: [a[k], b[k]] for k in (*endpoints, 'environment_interactions') if a[k] != b[k]}
        if changes:
            differences.append({'candidate_index': a['candidate_index'], 'changes': changes})
    return dict(endpoint_equal=not any(set(d['changes']) & set(endpoints) for d in differences),
                exact_equal=not differences,
                step_difference_count=sum('environment_interactions' in d['changes'] for d in differences),
                differences=differences)


def validate_request(count, horizon, sizes, budget):
    if not sizes or any(type(s) is not int or s < 1 for s in sizes) or len(set(sizes)) != len(sizes):
        raise ValueError('positive distinct batch sizes required')
    if count <= 0 or horizon <= 0:
        raise ValueError('positive count/horizon required')
    maximum = count * horizon * (len(sizes) + 1)
    if budget < maximum:
        raise ValueError('budget must cover every attempt reservation')
    return maximum


def gpu_sample(gpu):
    try:
        command = ['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits']
        memory, utilization = subprocess.check_output(command, text=True, timeout=3).strip().split(',')
        return {'memory_mib': int(memory), 'utilization_percent': int(utilization)}
    except (OSError, subprocess.SubprocessError, ValueError):
        return {'memory_mib': None, 'utilization_percent': None}


def render(output, records):
    fields = ['backend', 'batch_size', 'status', 'wall_seconds', 'label_seconds', 'compile_seconds',
              'restore_seconds', 'execution_seconds', 'charged_interactions', 'useful_interactions',
              'peak_device_memory_mib', 'endpoint_equal', 'exact_equal', 'candidates_per_second',
              'initialization_seconds', 'worlds', 'unique_source_candidates', 'replicated_capacity_test',
              'simulated_steps_per_second', 'useful_steps_per_second', 'substep_capacity_exceeded']
    with (output / 'timings.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(records)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = [(f"{r['batch_size']:,}\n" + ('capacity checked' if 'substep_capacity_exceeded' in r else 'initial'))
              if r['backend']=='vectorized_capacity' else f"{r['backend']}\nB={r['batch_size']}" for r in records]
    labels = [label if r['status']=='completed' else f"{r['batch_size']:,}\n{r['status']}"
              for label, r in zip(labels, records)]
    positions = list(range(len(records)))
    colors = ['tab:blue' if r['status']=='completed' else 'tab:red' for r in records]
    axes[0].bar(positions, [r['wall_seconds'] for r in records], color=colors)
    axes[0].set(ylabel='Full process wall seconds', title=('Replicated TRAIN capacity test'
        if any(r['backend']=='vectorized_capacity' for r in records) else 'Existing TRAIN execution comparison'))
    axes[1].bar(positions, [r.get('peak_device_memory_mib') or 0 for r in records], color=colors)
    for ax in axes:
        ax.set_xticks(positions, labels, fontsize=8)
    axes[1].set(ylabel='Observed whole-device memory (MiB)', title='Includes other processes; 1 second sampling')
    fig.tight_layout()
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(output / f'timings.{ext}', dpi=160)
    plt.close(fig)


def finalize_label_attempt(attempt, record, reference, ceiling, count, wall):
    """Validation failures retain the entire reservation and a durable error."""
    try:
        summary = read(attempt/'result/summary.json'); rows = read(attempt/'result/labels.json')
        cost = summary['environment_interactions']
        if type(cost) is not int or not 0 <= cost <= ceiling:
            raise ValueError('invalid measured cost')
        comparison = compare_labels(rows if reference is None else reference, rows)
        write(attempt/'comparison.json', comparison)
        record.update(status='completed' if comparison['endpoint_equal'] else 'endpoint_mismatch',
                      charged_interactions=cost, useful_interactions=summary['useful_label_interactions'],
                      label_seconds=summary['elapsed_seconds'],
                      compile_seconds=sum(summary['device_compile_seconds']) if record['backend']!='serial' else None,
                      restore_seconds=summary['snapshot_restore_seconds'],
                      execution_seconds=sum(summary['device_batch_seconds']) if record['backend']!='serial' else None,
                      candidates_per_second=count/wall, endpoint_equal=comparison['endpoint_equal'],
                      exact_equal=comparison['exact_equal'])
        return record, rows if reference is None else reference
    except (OSError, ValueError, KeyError, TypeError) as exc:
        record.update(status='engineering_error', charged_interactions=ceiling,
                      error=f'{type(exc).__name__}: {exc}')
        return record, reference


def finalize_capacity_attempt(attempt, record, returncode, stop_reason):
    ceiling = record['charged_interactions']
    try:
        report = read(attempt/'worker_report.json')
        cost = report['environment_interactions']
        if type(cost) is not int or not 0 <= cost <= ceiling:
            raise ValueError('invalid measured capacity cost')
        record.update(report)
        if returncode==0 and not stop_reason and report['status']=='completed':
            record['charged_interactions']=cost
        else:
            record['status']=stop_reason or (report['status'] if report['status']!='completed' else 'engineering_error')
            record['charged_interactions']=ceiling
    except (OSError, ValueError, KeyError, TypeError) as exc:
        record.update(status=stop_reason or 'engineering_error', charged_interactions=ceiling,
                      error=f'{type(exc).__name__}: {exc}')
    return record


def run(repo, output, *, catalog, acquisition, evaluator, sizes, shard_index, shard_count,
        seed, horizon, budget, timeout_seconds, gpu='0', backend='vectorized'):
    if backend not in {'device', 'vectorized'} or timeout_seconds <= 0:
        raise ValueError('invalid benchmark backend/timeout')
    repo, output = Path(repo).resolve(), Path(output).resolve()
    catalog, acquisition, evaluator = map(lambda p: Path(p).resolve(), (catalog, acquisition, evaluator))
    raw = read(catalog)
    protocol = read(catalog.parent / 'protocol.json')
    if protocol.get('logical_role') != 'train':
        raise ValueError('existing TRAIN acquisition required; no final TEST')
    from .unified_continuation_shards import contiguous_shard_bounds
    start, stop = contiguous_shard_bounds(raw['candidate_count'], shard_index, shard_count)
    maximum = validate_request(stop-start, horizon, sizes, budget)
    output.mkdir(parents=True, exist_ok=False)
    request = dict(schema='jit_continuation_execution_benchmark_v1', role='engineering_validation',
                   catalog=str(catalog), acquisition=str(acquisition), evaluator=str(evaluator), sizes=sizes,
                   shard_index=shard_index, shard_count=shard_count, protocol_seed=seed, horizon=horizon,
                   budget=budget, maximum_interactions=maximum, timeout_seconds=timeout_seconds, backend=backend,
                   candidate_start=start, candidate_stop=stop, final_test_used=False, training_transitions=0,
                   input_sha256={str(p): file_sha(p) for p in (catalog, catalog.parent/'protocol.json', acquisition, evaluator)},
                   source_sha256={str(p.relative_to(repo)): file_sha(p) for p in (repo/'JIT/src/jit_dvgc').rglob('*.py')},
                   started_utc=datetime.now(timezone.utc).isoformat(), gpu=str(gpu))
    write(output/'request.json', request)
    records = []
    reference = None
    for index, (kind, size) in enumerate([('serial', 1), *((backend, s) for s in sizes)]):
        attempt = output/f'attempt_{index:03d}_{kind}_{size}'
        attempt.mkdir()
        ceiling = (stop-start)*horizon
        write(attempt/'reservation.json', dict(maximum_interactions=ceiling, backend=kind, batch_size=size))
        command = [sys.executable, str(repo/'JIT/cli/label_policy_family_first_landing.py'),
                   '--catalog', str(catalog), '--acquisition-frozen-policy', str(acquisition),
                   '--evaluator-frozen-policy', str(evaluator), '--output-dir', str(attempt/'result'),
                   '--shard-index', str(shard_index), '--shard-count', str(shard_count),
                   '--protocol-seed', str(seed), '--max-ticks', str(horizon),
                   '--execution-backend', kind, '--batch-size', str(size)]
        write(attempt/'command.json', command)
        env = dict(os.environ, PYTHONPATH=str(repo/'JIT/src'), JAX_PLATFORMS='cuda,cpu',
                   CUDA_VISIBLE_DEVICES=str(gpu), XLA_PYTHON_CLIENT_PREALLOCATE='false', PYTHONUNBUFFERED='1')
        begin = time.perf_counter(); samples = []; timed_out = False
        print(f'[benchmark] {kind} batch={size}; log={attempt / "process.log"}', flush=True)
        with (attempt/'process.log').open('w') as log:
            child = subprocess.Popen(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                while child.poll() is None:
                    samples.append(dict(seconds=time.perf_counter()-begin, **gpu_sample(gpu)))
                    try:
                        child.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        if time.perf_counter()-begin > timeout_seconds:
                            timed_out = True; child.terminate()
                            try: child.wait(timeout=10)
                            except subprocess.TimeoutExpired: child.kill(); child.wait()
            finally:
                if child.poll() is None:
                    child.terminate()
                    try: child.wait(timeout=10)
                    except subprocess.TimeoutExpired: child.kill(); child.wait()
        wall = time.perf_counter()-begin
        write(attempt/'gpu_samples.json', samples)
        record = dict(backend=kind, batch_size=size, status='timeout' if timed_out else 'engineering_error',
                      wall_seconds=wall, charged_interactions=ceiling, returncode=child.returncode,
                      peak_device_memory_mib=max((s['memory_mib'] for s in samples if s['memory_mib'] is not None), default=None))
        if child.returncode == 0 and not timed_out:
            record, reference = finalize_label_attempt(attempt, record, reference, ceiling, stop-start, wall)
        records.append(record)
        write(attempt/'completion.json', record)
        write(output/'summary.json', dict(status='running', attempts=records,
             charged_interactions=sum(r['charged_interactions'] for r in records), final_test_used=False))
        print(f'[benchmark] {record}', flush=True)
        if record['status'] != 'completed' or not record.get('endpoint_equal', False):
            break
    complete = len(records) == len(sizes)+1 and all(r['status']=='completed' for r in records)
    summary = dict(status='completed' if complete else 'stopped', attempts=records,
                   charged_interactions=sum(r['charged_interactions'] for r in records),
                   exact_execution_comparison_passed=complete and all(r['exact_equal'] for r in records),
                   production_auto_adoption=False, final_test_used=False, training_transitions=0)
    write(output/'summary.json', summary)
    render(output, records)
    write(output/'figure_manifest.json', {p.name: file_sha(p) for p in output.glob('timings.*')})
    return summary


def capacity_worker(args):
    """Replicate a small verified TRAIN panel into real simultaneous Warp worlds."""
    import jax
    import jax.numpy as jp
    import numpy as np
    from .policy_family_landing import _load_frozen
    from .unified_formal import build_unified_formal_environment
    from .checkpoint import load_checkpoint
    from .unified_training import checkpoint_identity
    from .ppo import make_checkpoint_policy
    from .unified_continuation_labels import validate_unified_boundary_catalog, validate_candidate_snapshot, fresh_unified_continuation_start
    from .unified_envelope_snapshot import load_unified_envelope_snapshot
    from .continuation.device_rollout import stack_worlds, repeat_worlds, make_device_rollout, prepare_parallel_worlds
    begin = time.perf_counter()
    acquisition, acquisition_sha = _load_frozen(args.acquisition)
    evaluator, _ = _load_frozen(args.evaluator)
    raw = read(args.catalog)
    if read(args.catalog.parent/'protocol.json').get('logical_role') != 'train':
        raise ValueError('capacity test requires existing TRAIN data')
    rows = validate_unified_boundary_catalog(raw, policy_record=acquisition, frozen_manifest_sha256=acquisition_sha)
    config, _, env = build_unified_formal_environment(Path(evaluator['formal_config']))
    payload = load_checkpoint(Path(evaluator['checkpoint']), expected=checkpoint_identity(config, env))
    policy = make_checkpoint_policy(env, payload, deterministic=True)
    # Evenly spaced indices cover the source trajectory/phase sequence. Replicas
    # remain capacity measurements, never extra independent arrivals or labels.
    indices = np.linspace(0, len(rows)-1, min(16, len(rows)), dtype=int).tolist()
    states = []
    for index in indices:
        row = rows[index]
        snap = load_unified_envelope_snapshot(args.catalog.parent/row['source_bank']/row['snapshot'])
        validate_candidate_snapshot(snap, row, policy_record=acquisition)
        states.append(fresh_unified_continuation_start(snap, env))
    batch = repeat_worlds(stack_worlds(states), args.worlds)
    # Only scratch resource capacities scale. XML, solver, physics, controller,
    # observations and endpoint remain unchanged. Contact buffers are GLOBAL.
    contacts = max(int(env.resolved_config.model['naconmax']), args.worlds*64)
    ccd = max(env._reset_data_naccdmax() or 0, args.worlds*64)
    batch = prepare_parallel_worlds(batch, env, args.worlds)
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.worlds)
    jax.block_until_ready((batch, keys))
    initialized = time.perf_counter()-begin
    run = make_device_rollout(policy, env.step, args.ticks, vectorized=True)
    begin_compile = time.perf_counter(); compiled = run.lower(batch, keys).compile()
    compile_seconds = time.perf_counter()-begin_compile
    begin_run = time.perf_counter(); result = compiled(batch, keys); jax.block_until_ready(result)
    seconds = time.perf_counter()-begin_run
    final, counts, bad, flags, cost = result
    counts, bad, flags, cost = jax.device_get((counts, bad, flags, cost))
    report = dict(status='completed' if not np.any(bad) else 'nonfinite_switching_or_capacity',
                  worlds=args.worlds, unique_source_candidates=len(indices), source_indices=indices,
                  ticks=args.ticks, environment_interactions=int(cost), useful_interactions=int(sum(counts)),
                  inactive_interactions=int(cost)-int(sum(counts)), initialization_seconds=initialized,
                  compile_seconds=compile_seconds, execution_seconds=seconds,
                  simulated_steps_per_second=int(cost)/seconds, useful_steps_per_second=int(sum(counts))/seconds,
                  naconmax=contacts, naccdmax=ccd, njmax=int(env.resolved_config.model['njmax']),
                  replicated_capacity_test=True, training_transitions=0, new_scientific_candidates=0,
                  horizon_truncated_lanes=int(sum((counts==args.ticks) & ~flags[:, 3])),
                  successful_lanes=int(sum(flags[:, 3])), final_test_used=False,
                  substep_capacity_exceeded=bool(np.any(jax.device_get(final.info['parallel_capacity_exceeded']))),
                  nacon=int(np.asarray(jax.device_get(final.data._impl.nacon)).max()))
    # Exceeding aggregate contact capacity cannot qualify as a successful test.
    if report['nacon'] >= contacts:
        report['status'] = 'contact_capacity_exhausted'
    write(args.output_dir/'worker_report.json', report)
    return 0 if report['status']=='completed' else 2


def run_capacity(repo, output, args):
    sizes = args.batch_sizes
    if not sizes or any(type(n) is not int or n < 1 for n in sizes) or sizes != sorted(set(sizes)):
        raise ValueError('positive increasing capacity sizes required')
    maximum = sum(sizes)*args.ticks
    if args.ticks <= 0 or args.timeout_seconds <= 0 or maximum > args.budget:
        raise ValueError('capacity budget/timeout invalid')
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    paths = [args.catalog.resolve(), args.acquisition.resolve(), args.evaluator.resolve()]
    request = dict(schema='jit_parallel_capacity_v1', worlds=sizes, ticks=args.ticks,
                   maximum_interactions=maximum, budget=args.budget, seed=args.seed,
                   timeout_seconds=args.timeout_seconds, memory_limit_mib=args.memory_limit_mib,
                   input_sha256={str(p):file_sha(p) for p in paths},
                   source_sha256={str(p.relative_to(repo)): file_sha(p) for p in (repo/'JIT/src/jit_dvgc').rglob('*.py')},
                   scratch_contacts_per_world=64, scratch_ccd_per_world=64,
                   role='engineering_capacity', replicated=True, new_scientific_candidates=0,
                   final_test_used=False, training_transitions=0, started_utc=datetime.now(timezone.utc).isoformat())
    write(output/'request.json', request)
    records = []
    for size in sizes:
        attempt = output/f'worlds_{size}'; attempt.mkdir()
        write(attempt/'reservation.json', {'maximum_interactions':size*args.ticks})
        command = [sys.executable, str(repo/'JIT/cli/benchmark_continuation.py'), '--capacity-worker',
                   '--catalog',str(paths[0]),'--acquisition',str(paths[1]),'--evaluator',str(paths[2]),
                   '--output-dir',str(attempt),'--worlds',str(size),'--ticks',str(args.ticks),
                   '--seed',str(args.seed),'--budget',str(size*args.ticks)]
        write(attempt/'command.json', command)
        env = dict(os.environ, PYTHONPATH=str(repo/'JIT/src'), JAX_PLATFORMS='cuda,cpu',
                   CUDA_VISIBLE_DEVICES=str(args.gpu), XLA_PYTHON_CLIENT_PREALLOCATE='false', PYTHONUNBUFFERED='1')
        start = time.perf_counter(); samples = []; stop_reason = None
        print(f'[capacity] worlds={size}; log={attempt / "process.log"}', flush=True)
        with (attempt/'process.log').open('w') as log:
            child = subprocess.Popen(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                while child.poll() is None:
                    sample = dict(seconds=time.perf_counter()-start, **gpu_sample(args.gpu)); samples.append(sample)
                    if sample['seconds'] > args.timeout_seconds: stop_reason='timeout'
                    if (sample['memory_mib'] or 0) > args.memory_limit_mib: stop_reason='memory_limit'
                    if stop_reason:
                        child.terminate()
                        try: child.wait(timeout=10)
                        except subprocess.TimeoutExpired: child.kill(); child.wait()
                        break
                    try: child.wait(timeout=1)
                    except subprocess.TimeoutExpired: pass
            finally:
                if child.poll() is None:
                    child.terminate()
                    try: child.wait(timeout=10)
                    except subprocess.TimeoutExpired: child.kill(); child.wait()
        record = dict(backend='vectorized_capacity', batch_size=size, status=stop_reason or 'engineering_error',
                      wall_seconds=time.perf_counter()-start, returncode=child.returncode,
                      charged_interactions=size*args.ticks,
                      peak_device_memory_mib=max((s['memory_mib'] for s in samples if s['memory_mib'] is not None),default=None))
        write(attempt/'gpu_samples.json', samples)
        record = finalize_capacity_attempt(attempt, record, child.returncode, stop_reason)
        records.append(record); write(attempt/'completion.json', record)
        write(output/'summary.json',dict(status='running',attempts=records,charged_interactions=sum(r['charged_interactions'] for r in records)))
        print(f'[capacity] {record}',flush=True)
        if record['status']!='completed': break
        if len(records)>1 and record['useful_steps_per_second'] < records[-2]['useful_steps_per_second']:
            break
    summary=dict(status='completed' if all(r['status']=='completed' for r in records) else 'stopped',
                 attempts=records,charged_interactions=sum(r['charged_interactions'] for r in records),
                 new_scientific_candidates=0,final_test_used=False,production_auto_adoption=False)
    write(output/'summary.json',summary); render(output,records)
    write(output/'figure_manifest.json',{p.name:file_sha(p) for p in output.glob('timings.*')})
    return summary
