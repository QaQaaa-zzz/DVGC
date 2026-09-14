"""Bounded qualification batches with immutable inputs and complete receipts."""
import argparse
import json
import math
import numbers
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from .protocol import close_budget, reserve_budget, sha256, validate_config, write_json


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value) if math.isfinite(value) else None
    return value


def execute_cases(config, run_dir, source_root, *, case_runner=None, on_progress=None):
    if case_runner is None:
        from .runtime import QualificationRuntime, run_case
        runtimes = {}
        def case_runner(config, policy, source_root, case):
            if policy['name'] not in runtimes:
                runtimes[policy['name']] = QualificationRuntime(config, policy, source_root)
            return run_case(config, policy, source_root, case, runtime=runtimes[policy['name']])
    run_dir = Path(run_dir)
    trace_dir = run_dir / 'trajectories'
    trace_dir.mkdir(parents=True, exist_ok=True)
    maximum_ticks = round(config['timing']['episode_seconds'] / config['timing']['control_dt'])
    summaries = []
    failed = False
    engineering_errors = 0
    conservative = 0
    physics_steps = 0
    for policy in config['policies']:
        for case in config['cases']:
            case_id = f"{policy['name']}__{case['name']}"
            if on_progress:
                on_progress(case_id, summaries)
            started = time.monotonic()
            rows = []
            summary = {'status': 'unknown', 'reason': 'not_executed_after_engineering_error',
                       'qualified': False, 'charged_control_steps': 0, 'physics_steps': 0}
            if not failed:
                try:
                    summary, rows = case_runner(config, policy, source_root, case)
                    if not 0 <= summary['charged_control_steps'] <= maximum_ticks:
                        raise ValueError('runtime returned cost beyond per-case reservation')
                    if not 0 <= summary['physics_steps'] <= 4 * summary['charged_control_steps']:
                        raise ValueError('runtime returned inconsistent physics accounting')
                    physics_steps += summary['physics_steps']
                    if summary['status'] == 'unknown':
                        failed = True
                        engineering_errors += 1
                except Exception as exc:
                    failed = True
                    engineering_errors += 1
                    conservative += maximum_ticks
                    summary = {'status': 'unknown', 'reason': 'engineering_error', 'qualified': False,
                               'error': str(exc), 'traceback': traceback.format_exc(),
                               'charged_control_steps': maximum_ticks, 'physics_steps': None,
                               'accounting': 'full_case_reserved_charge_actual_work_unknown'}
            summary.update(case_id=case_id, policy=policy['name'], case_name=case['name'],
                           trigger_delay_seconds=case['trigger_delay_seconds'], seed=case['seed'],
                           wall_seconds=time.monotonic() - started)
            summary.setdefault('trigger_time', None)
            summary.setdefault('liftoff_time', None)
            if rows:
                trace = trace_dir / f'{case_id}.jsonl'
                with trace.open('x', encoding='utf-8') as stream:
                    for row in rows:
                        stream.write(json.dumps(json_safe(row), ensure_ascii=False, allow_nan=False) + '\n')
                summary['trajectory'] = str(trace.relative_to(run_dir))
                summary['trajectory_sha256'] = sha256(trace)
            summaries.append(json_safe(summary))
            write_json(run_dir / 'partial_results.json', {'cases': summaries})
    accounting = {'charged_control_steps': sum(row['charged_control_steps'] for row in summaries),
                  'physics_steps': physics_steps, 'physics_steps_complete': engineering_errors == 0,
                  'conservative_error_control_steps': conservative, 'engineering_errors': engineering_errors,
                  'ppo_training_steps': 0, 'supervised_optimizer_updates': 0,
                  'case_count': len(summaries)}
    return summaries, accounting


def _manifest(config, source_root, workspace):
    dependencies = {}
    for directory in (workspace / 'JUMP/src', workspace / 'JUMP/cli', workspace / 'JIT/src/jit_dvgc'):
        for path in sorted(directory.rglob('*.py')):
            dependencies[str(path.relative_to(workspace))] = sha256(path)
    policies = []
    for spec in config['policies']:
        directory = source_root / spec['checkpoint']
        sidecar = json.loads((directory / 'identity.json').read_text())
        payload_hash = sha256(directory / 'payload.pkl')
        if payload_hash != sidecar['payload_sha256'] or payload_hash != spec['payload_sha256']:
            raise ValueError(f"policy payload identity changed: {spec['name']}")
        policies.append({'name': spec['name'], 'path': str(directory), 'identity': sidecar,
                         'identity_sha256': sha256(directory / 'identity.json'), 'payload_sha256': payload_hash})
    return {'config': config, 'policies': policies, 'source_hashes': dependencies,
            'source_root': str(source_root), 'workspace': str(workspace),
            'code_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=workspace, text=True).strip(),
            'working_tree_source_hashes_authoritative': True,
            'backend': 'mujoco_cpu', 'created_unix': time.time(),
            'claim_boundary': 'new CPU qualification protocol; no legacy MJX equivalence or complete skill assumed'}


def run_declared(config_path, campaign_path, source_root, output, workspace):
    """Create a new run and seal configuration/startup errors as terminal status."""
    workspace = Path(workspace).resolve()
    source_root = Path(source_root).resolve()
    output = Path(output).resolve()
    project = workspace / 'JUMP'
    if not output.is_relative_to(project / 'runs'):
        raise ValueError('output must be inside this JUMP/runs directory')
    # Refuse an existing run before touching any of its status or artifacts.
    output.mkdir(parents=True, exist_ok=False)
    run_id = str(output.relative_to(project / 'runs'))
    ledger = project / 'runs/campaign_ledger.json'
    write_json(output / 'ACTIVE_RUN.json', {'execution': 'status.json'})
    started = time.time()
    status = {'schema': 'jump_execution_v1', 'phase': 'running', 'pid': os.getpid(),
              'stage': 'preflight', 'started_unix': started}
    write_json(output / 'status.json', status)
    metadata = {}
    charged = 0
    reserved = False
    terminal_phase = 'error'
    def timeout_handler(signum, frame):
        raise TimeoutError('declared qualification wall-clock budget exhausted')
    signal.signal(signal.SIGALRM, timeout_handler)
    try:
        config = json.loads(Path(config_path).read_text())
        maximum = validate_config(config)
        campaign = json.loads(Path(campaign_path).read_text())
        write_json(output / 'config.json', config)
        write_json(output / 'campaign.json', campaign)
        reserve_budget(ledger, campaign, run_id, config['budget']['stage'], maximum)
        reserved = True
        write_json(output / 'reservation.json', {'run_id': run_id, 'maximum_control_steps': maximum,
                    'campaign_sha256': sha256(campaign_path), 'data_role': config['data_role']})
        status.update(maximum_control_steps=maximum, data_role=config['data_role'])
        signal.setitimer(signal.ITIMER_REAL, config['budget']['wall_seconds'])
        metadata = _manifest(config, source_root, workspace)
        write_json(output / 'manifest.json', metadata)
        def progress(name, rows):
            status.update(stage=name, completed_cases=len(rows), updated_unix=time.time())
            write_json(output / 'status.json', status)
        charged = maximum  # Once stepping may start, unknown work retains its reservation.
        cases, accounting = execute_cases(config, output, source_root, on_progress=progress)
        charged = accounting['charged_control_steps']
        terminal_phase = 'error' if accounting['engineering_errors'] else 'completed'
        results = {'schema': 'jump_qualification_results_v1', 'cases': cases,
                   'accounting': accounting, 'metadata': metadata,
                   'outcome_counts': {label: sum(row['status'] == label for row in cases)
                                      for label in ('success', 'failure', 'unknown')},
                   'qualified_cases': sum(bool(row.get('qualified')) for row in cases)}
        write_json(output / 'results.json', results)
        from .reporting import build_report
        report = build_report(output)
        status.update(accounting=accounting, outcome_counts=results['outcome_counts'],
                      qualified_cases=results['qualified_cases'], completed_cases=len(cases),
                      stage='report_complete', report=json_safe(report))
        if terminal_phase == 'error':
            status['error'] = next((row.get('error') or row['reason'] for row in cases
                                    if row['status'] == 'unknown'), 'unknown qualification result')
    except Exception as exc:
        terminal_phase = 'timeout' if isinstance(exc, TimeoutError) else 'error'
        status.update(error=str(exc), traceback=traceback.format_exc())
        write_json(output / 'error.json', status)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if reserved:
            close_budget(ledger, run_id, charged, terminal_phase)
        status.update(phase=terminal_phase, finished_unix=time.time(), wall_seconds=time.time() - started,
                      charged_control_steps=charged)
        write_json(output / 'status.json', status)
    return {'output': str(output), 'phase': terminal_phase,
            'charged_control_steps': charged, 'outcomes': status.get('outcome_counts')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    args = parser.parse_args()
    result = run_declared(args.config, args.campaign, args.source_root, args.output,
                          Path(__file__).resolve().parents[3])
    print(json.dumps(result, ensure_ascii=False))
    if result['phase'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
