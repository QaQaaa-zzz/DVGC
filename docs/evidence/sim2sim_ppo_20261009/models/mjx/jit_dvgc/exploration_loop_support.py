"""Append newly witnessed complete contexts to inherited TRAIN reset support."""
from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path

from .evidence_integrity import canonical_sha256
from .exploration_pool import validate_pool, _validate_result, _snapshot, _sha
from .iterative_probe_training import SUPPORT_SCHEMA
from .jump_evidence_validation import verify_hash


def append_witnessed_support(base_support, pool, pool_path):
    """Preserve inherited rows and append only verified same-context positives.

    New trajectory groups receive the mean inherited trajectory-group weight in
    their phase; rows within each new group share that mass. Inherited relative
    weights are untouched. The downstream candidate-support adapter normalizes
    the combined witnessed stratum to its declared quota. Call with the original
    inherited support and the cumulative learned-arm pool for stable weighting.
    """
    verify_hash(base_support, 'support_sha256')
    validate_pool(pool)
    if base_support.get('schema') != SUPPORT_SCHEMA or base_support.get('role') != 'train' or base_support.get('final_test_used') is not False:
        raise ValueError('inherited witnessed TRAIN support required')
    if any(row.get('witnessed') is not True for row in base_support['entries']):
        raise ValueError('inherited support contains unwitnessed rows')
    pool_path = Path(pool_path).resolve()
    if json.loads(pool_path.read_text()) != pool:
        raise ValueError('pool input differs from declared on-disk pool')
    result = deepcopy(base_support)
    result.pop('support_sha256')
    inputs = result['inputs']
    for path, expected in inputs.items():
        if _sha(path) != expected:
            raise ValueError('inherited support input changed')
    def lock(path):
        path = str(Path(path).resolve())
        value = _sha(path)
        if path in inputs and inputs[path] != value:
            raise ValueError('support provenance input changed')
        inputs[path] = value
    lock(pool_path)
    keys = set()
    contexts = set()
    group_weights = defaultdict(lambda: defaultdict(float))
    for row in result['entries']:
        if row['key'] in keys:
            raise ValueError('duplicate inherited support key')
        keys.add(row['key'])
        if row['phase'] not in ('upstream', 'downstream') or not math.isfinite(row['sampling_weight']) or row['sampling_weight'] <= 0:
            raise ValueError('invalid inherited support phase/weight')
        snap = _snapshot(dict(snapshot_dir=row['snapshot'], state_sha256=row['state_sha256'], context_sha256=row['snapshot_context_sha256']))
        if row['phase'] != ('upstream' if snap.active_phase == 0 else 'downstream'):
            raise ValueError('inherited support snapshot phase mismatch')
        contexts.add((row['state_sha256'], row['snapshot_context_sha256']))
        group_weights[row['phase']][row['trajectory_id']] += row['sampling_weight']
        for filename in ('identity.json', 'snapshot.pkl'):
            lock(Path(row['snapshot']) / filename)
    if set(group_weights) != {'upstream', 'downstream'}:
        raise ValueError('inherited witnessed support must cover both phases')
    additions = []
    for key, entry in sorted(pool['entries'].items()):
        if entry['status'] != 'witnessed':
            continue
        positive = next((o for o in entry['observations'] if o['label'] == 1), None)
        if positive is None:
            raise ValueError('witnessed pool entry lacks positive receipt')
        snap = _validate_result(entry, positive['result'])
        if entry['phase'] != ('upstream' if snap.active_phase == 0 else 'downstream'):
            raise ValueError('pool snapshot phase mismatch')
        context = (entry['state_sha256'], entry['context_sha256'])
        if key in keys:
            existing = next(row for row in result['entries'] if row['key'] == key)
            if context != (existing['state_sha256'], existing['snapshot_context_sha256']):
                raise ValueError('support key context conflict')
            continue
        if context in contexts:
            continue  # distinct acquisition provenance remains in the locked pool
        source = entry['sources'][0]['candidate']
        row = dict(key=key, phase=entry['phase'], snapshot=entry['snapshot_dir'],
            state_sha256=entry['state_sha256'], snapshot_context_sha256=entry['context_sha256'],
            trajectory_id=source['prefix_file'] + '::' + str(source['episode_index']),
            root_cell=entry['root_cell'], witnessed=True, evidence_status='witnessed', role='train',
            acquisition_protocol_sha256=entry['acquisition_protocol_sha256'],
            witness=positive['result']['witness'], witness_bank_sha256=positive['bank_sha256'],
            witness_receipt_sha256=positive['receipt_sha256'],
            first_discovery_round=entry['first_discovery_round'], first_witness_round=positive['round_index'])
        for filename in ('identity.json', 'snapshot.pkl'):
            lock(Path(row['snapshot']) / filename)
        additions.append(row)
        contexts.add(context); keys.add(key)
    counts = Counter((r['phase'], r['trajectory_id']) for r in additions)
    for row in additions:
        phase = row['phase']
        mean_group_mass = sum(group_weights[phase].values()) / len(group_weights[phase])
        row['sampling_weight'] = mean_group_mass / counts[(phase, row['trajectory_id'])]
    result['entries'].extend(additions)
    result.update(inherited_support_sha256=base_support['support_sha256'],
        appended_pool_sha256=pool['pool_sha256'], appended_witnessed_count=len(additions),
        appended_sampling_contract='preserve inherited row weights; new trajectory groups receive mean inherited group mass per phase',
        context_deduplication='state_sha256 plus snapshot_context_sha256; distinct acquisition histories remain in pool')
    result['support_sha256'] = canonical_sha256(result)
    return result
