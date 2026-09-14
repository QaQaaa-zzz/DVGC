"""Validated declarations and an atomic, concurrency-safe control-step ledger."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False, encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        temporary = Path(stream.name)
    os.replace(temporary, path)


def validate_config(config):
    if config['schema'] != 'jump_qualification_v1' or config['backend'] != 'mujoco_cpu':
        raise ValueError('unsupported schema/backend')
    if config['data_role'] != 'ENGINEERING_DEVELOPMENT':
        raise ValueError('qualification is development, not final TEST')
    for section in ('scene', 'timing', 'limits', 'initial', 'action', 'budget'):
        for key, value in config[section].items():
            if isinstance(value, (float, int)) and not math.isfinite(value):
                raise ValueError(f'nonfinite {section}.{key}')
    timing = config['timing']
    if timing['sim_dt'] != .005 or timing['control_dt'] != .02:
        raise ValueError('timing contract must be 5ms physics / 20ms control')
    if not 0 < timing['episode_seconds'] <= 8:
        raise ValueError('episode horizon must be positive and at most 8s')
    for key in ('settle_seconds', 'airborne_seconds', 'recovery_seconds', 'hold_seconds', 'approach_timeout_seconds'):
        if timing[key] <= 0:
            raise ValueError(f'invalid {key}')
    if timing['recovery_seconds'] != 2.0 or timing['hold_seconds'] != .5:
        raise ValueError('recovery contract must retain 2s / .5s')
    if timing['approach_timeout_seconds'] >= timing['episode_seconds']:
        raise ValueError('approach timeout must precede episode horizon')
    for key in ('length', 'height', 'half_width'):
        if config['scene'][key] <= 0:
            raise ValueError(f'invalid scene {key}')
    if not 0 <= config['scene']['landing_near'] < config['scene']['landing_far']:
        raise ValueError('invalid landing interval')
    if any(value <= 0 for value in config['limits'].values()):
        raise ValueError('all physical/measurement limits must be positive')
    if config['limits']['min_forward_speed'] > config['limits']['nominal_speed']:
        raise ValueError('minimum forward speed exceeds nominal')
    if not config['policies'] or not config['cases']:
        raise ValueError('empty qualification panel')
    for group in ('policies', 'cases'):
        names = [row['name'] for row in config[group]]
        if len(names) != len(set(names)) or any(not re.fullmatch(r'[A-Za-z0-9_-]+', name) for name in names):
            raise ValueError(f'invalid or duplicate {group} names')
    for case in config['cases']:
        delay = case['trigger_delay_seconds']
        if delay is not None and (not math.isfinite(delay) or delay < 0 or delay >= timing['episode_seconds']):
            raise ValueError('invalid trigger delay')
        if not isinstance(case['seed'], int) or case['seed'] < 0:
            raise ValueError('invalid seed')
    ticks = timing['episode_seconds'] / timing['control_dt']
    if not math.isclose(ticks, round(ticks), abs_tol=1e-9):
        raise ValueError('horizon must contain whole controls')
    maximum = len(config['policies']) * len(config['cases']) * round(ticks)
    budget = config['budget']
    if not isinstance(budget['max_control_steps'], int) or maximum > budget['max_control_steps']:
        raise ValueError('qualification exceeds declared budget')
    if budget['wall_seconds'] <= 0:
        raise ValueError('wall budget must be positive')
    return maximum


@contextmanager
def _locked_ledger(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(path.read_text()) if path.exists() else {'schema': 'jump_budget_ledger_v1', 'runs': {}}
        yield data
        write_json(path, data)


def reserve_budget(path, campaign, run_id, stage, amount):
    total = campaign['total_control_steps']
    stages = campaign['stages']
    if not isinstance(total, int) or total <= 0 or any(not isinstance(v, int) or v < 0 for v in stages.values()) or sum(stages.values()) != total:
        raise ValueError('invalid campaign budget allocation')
    if stage not in stages or not isinstance(amount, int) or amount <= 0:
        raise ValueError('invalid stage budget reservation')
    with _locked_ledger(path) as data:
        declaration_sha = hashlib.sha256(json.dumps(campaign, sort_keys=True).encode()).hexdigest()
        if data.get('campaign_sha256', declaration_sha) != declaration_sha:
            raise ValueError('campaign budget changed; existing ledger must remain immutable')
        if run_id in data['runs']:
            raise ValueError('run already has a budget receipt')
        def cost(row):
            return row['reserved_control_steps'] if row['status'] == 'reserved' else row['charged_control_steps']
        used = sum(cost(row) for row in data['runs'].values())
        stage_used = sum(cost(row) for row in data['runs'].values() if row['stage'] == stage)
        if used + amount > total or stage_used + amount > stages[stage]:
            raise ValueError('campaign/stage budget exceeded')
        data['campaign_sha256'] = declaration_sha
        data['campaign'] = campaign
        data['runs'][run_id] = {'stage': stage, 'reserved_control_steps': amount, 'status': 'reserved', 'reserved_unix': time.time()}


def close_budget(path, run_id, charged, status):
    with _locked_ledger(path) as data:
        row = data['runs'][run_id]
        if row['status'] != 'reserved':
            raise ValueError('reservation already closed')
        if not isinstance(charged, int) or charged < 0 or charged > row['reserved_control_steps']:
            raise ValueError('actual cost exceeds reservation')
        if status not in ('completed', 'error', 'timeout'):
            raise ValueError('invalid closing status')
        row.update(charged_control_steps=charged, status=status, closed_unix=time.time())
