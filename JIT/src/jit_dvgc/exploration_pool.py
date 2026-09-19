"""Append-only TRAIN arrivals, separate from the witnessed empirical envelope.

All operations are offline. Mutations return a new self-hashed document; callers
persist it as a new round artifact, leaving previous versions immutable.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from .evidence_integrity import canonical_sha256

SCHEMA = 'jit_pending_exploration_pool_v1'
PROVENANCE_SCHEMA = 'jit_residual_prefix_acquisition_identity_v1'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verify(doc, field='receipt_sha256'):
    if canonical_sha256({k: v for k, v in doc.items() if k != field}) != doc.get(field):
        raise ValueError(f'{field} self-hash mismatch')


def _seal(pool):
    pool.pop('pool_sha256', None)
    pool['pool_sha256'] = canonical_sha256(pool)
    return pool


def candidate_key(state_sha256, context_sha256, acquisition_protocol_sha256):
    values = [state_sha256, context_sha256, acquisition_protocol_sha256]
    if any(not isinstance(v, str) or len(v) != 64 or any(c not in '0123456789abcdef' for c in v) for v in values):
        raise ValueError('candidate identity must contain SHA-256 values')
    return canonical_sha256(dict(zip(('state_sha256', 'context_sha256', 'acquisition_protocol_sha256'), values)))


def validate_pool(pool):
    _verify(pool, 'pool_sha256')
    if pool.get('schema') != SCHEMA or pool.get('role') != 'TRAIN':
        raise ValueError('pool must use TRAIN schema')
    for key, entry in pool['entries'].items():
        if key != candidate_key(*(entry[k] for k in ('state_sha256', 'context_sha256', 'acquisition_protocol_sha256'))):
            raise ValueError('pool candidate key mismatch')
        if entry['status'] != ('witnessed' if any(o['label'] == 1 for o in entry['observations']) else 'pending'):
            raise ValueError('pool status disagrees with observation history')
    return pool


def _snapshot(entry):
    from .unified_envelope_snapshot import load_unified_envelope_snapshot, physical_state_sha256, snapshot_context_sha256
    snap = load_unified_envelope_snapshot(entry['snapshot_dir'])
    if physical_state_sha256(snap) != entry['state_sha256'] or snapshot_context_sha256(snap) != entry['context_sha256']:
        raise ValueError('saved snapshot state/context mismatch')
    return snap


def _validate_result(entry, result):
    _verify(result)
    snap = _snapshot(entry)
    if result.get('state_sha256') != entry['state_sha256'] or result.get('snapshot_context_sha256') != entry['context_sha256']:
        raise ValueError('evaluation state/context mismatch')
    # Identity sidecars may differ in proposer metadata across identical contexts.
    # The result must bind a saved sidecar with this same complete state/context.
    result_snapshot = result.get('snapshot_dir', entry['snapshot_dir'])
    if result['snapshot_identity_sha256'] != _sha(Path(result_snapshot) / 'identity.json'):
        raise ValueError('evaluation snapshot identity mismatch')
    if str(result_snapshot) != entry['snapshot_dir']:
        _snapshot({**entry, 'snapshot_dir': str(result_snapshot)})
    bank = result.get('bank_sha256')
    candidate_key(entry['state_sha256'], entry['context_sha256'], bank)
    labels = result.get('labels', {})
    if not labels or any(type(v) is not int and v is not None or v not in (None, 0, 1) for v in labels.values()):
        raise ValueError('invalid evaluator label map')
    attempts = {}
    for attempt in result.get('attempts', []):
        _verify(attempt)
        name = attempt['evaluator']
        if name in attempts or name not in labels or attempt['snapshot_context_sha256'] != entry['context_sha256']:
            raise ValueError('evaluation attempt identity mismatch')
        if attempt.get('label') != labels[name]:
            raise ValueError('attempt label mismatch')
        if attempt.get('label') is not None and attempt.get('status') != 'completed':
            raise ValueError('incomplete attempt cannot supply a label')
        if attempt.get('label') == 1:
            flags = attempt.get('end_flags', {})
            if attempt.get('outcome') != 'first_valid_landing' or flags.get('physical_failure') is not False or flags.get('down/valid_contact_seen') is not True:
                raise ValueError('positive attempt lacks unconflicted landing evidence')
        attempts[name] = attempt
    if any(value is not None and name not in attempts for name, value in labels.items()):
        raise ValueError('label missing evaluator receipt')
    expected = 1 if 1 in labels.values() else (0 if all(v == 0 for v in labels.values()) else None)
    if result.get('label') != expected or (expected == 1 and labels.get(result.get('witness')) != 1) or (expected != 1 and result.get('witness') is not None):
        raise ValueError('existence label/witness mismatch')
    return snap


def add_evaluation(pool, key, result, *, round_index):
    """Append a verified bank observation; failure/unknown never imply impossible."""
    validate_pool(pool)
    if type(round_index) is not int or round_index < 0:
        raise ValueError('round_index must be nonnegative')
    new = deepcopy(pool)
    entry = new['entries'][key]
    if round_index < entry['first_discovery_round']:
        raise ValueError('evaluation predates discovery')
    _validate_result(entry, result)
    observation = dict(round_index=round_index, bank_sha256=result['bank_sha256'],
        evaluator=result.get('witness'), receipt_sha256=result['receipt_sha256'],
        label=result['label'], result=deepcopy(result))
    if not any(o['receipt_sha256'] == observation['receipt_sha256'] for o in entry['observations']):
        entry['observations'].append(observation)
    entry['status'] = 'witnessed' if any(o['label'] == 1 for o in entry['observations']) else 'pending'
    return _seal(new)


def _verify_prefix(candidate, snap, root):
    from flax import serialization
    from .handoff_bank import pytree_sha256
    controller = candidate['controller']
    _verify(controller, 'controller_sha256')
    checkpoint = Path(controller['residual_checkpoint'])
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    if _sha(checkpoint) != controller['residual_checkpoint_sha256']:
        raise ValueError('residual checkpoint hash mismatch')
    state = serialization.msgpack_restore(checkpoint.read_bytes())
    if pytree_sha256(state['params']['policy']) != controller['residual_actor_sha256']:
        raise ValueError('residual actor identity mismatch')
    from brax.training.acme.running_statistics import RunningStatisticsState
    from brax.training.types import UInt64
    normalizer = dict(state['normalizer'])
    if isinstance(normalizer.get('count'), dict):
        normalizer['count'] = UInt64(**normalizer['count'])
    if pytree_sha256(RunningStatisticsState(**normalizer)) != controller['normalizer_sha256']:
        raise ValueError('normalizer identity mismatch')
    identity = json.loads(checkpoint.with_name('identity.json').read_text())
    if identity['base_actor_sha256'] != controller['base_actor_sha256'] or identity['state_sha256'] != _sha(checkpoint) or identity['delta_limit'] != controller['delta_limit']:
        raise ValueError('checkpoint controller identity mismatch')
    if snap.policy_actor_sha256 != controller['controller_sha256'] or snap.policy_payload_sha256 != controller['residual_checkpoint_sha256']:
        raise ValueError('snapshot composite controller mismatch')
    prefix = Path(candidate['prefix_file'])
    if not prefix.is_absolute():
        prefix = root / prefix
    if _sha(prefix) != candidate['prefix_file_sha256']:
        raise ValueError('prefix file hash mismatch')
    with np.load(prefix, allow_pickle=False) as data:
        t, e = candidate['tick'], candidate['episode_index']
        fields = {'data/qpos': snap.qpos, 'data/qvel': snap.qvel, 'data/ctrl': snap.ctrl,
            'obs/state': snap.observation, 'history/frames': snap.observation_fifo,
            'history/valid_count': snap.history_valid_count}
        from .exploration_continuation import INFO_FIELDS
        fields.update({'info/' + k: getattr(snap, k) for k in INFO_FIELDS if k != 'expert_switching_used'})
        fields.update({'info/done': False, 'info/expert_switching_used': False})
        for prefix_name, events in [('up', snap.up_events), ('down', snap.down_events)]:
            fields.update({prefix_name + '/' + k: v for k, v in events.items()})
        if t < 0 or e < 0:
            raise ValueError('negative prefix index')
        for name, expected in fields.items():
            if not np.array_equal(data['snap/' + name][t, e], expected):
                raise ValueError('prefix snapshot mismatch: ' + name)
    if candidate.get('generated_by_env_step_only') is not True or snap.parent_state_sha256 != candidate['jump_start_state_sha256']:
        raise ValueError('candidate lacks real forward prefix provenance')


def import_candidates(run_root, prior_pool=None, *, round_index=0):
    """Import every saved residual candidate, including unsuccessful/deferred ones."""
    root = Path(run_root).resolve()
    if type(round_index) is not int or round_index < 0:
        raise ValueError('round_index must be nonnegative')
    pool = deepcopy(validate_pool(prior_pool)) if prior_pool is not None else _seal(dict(schema=SCHEMA, role='TRAIN', entries={}))
    declaration = json.loads((root / 'declaration.json').read_text())
    if str(declaration.get('role', '')).upper() != 'TRAIN' or declaration.get('final_test_used') is not False:
        raise ValueError('only TRAIN explorer outputs may enter pool')
    for source in sorted(root.glob('*_candidates.json')):
        for candidate in json.loads(source.read_text()):
            _verify(candidate)
            context = candidate['context_sha256']
            snapshot_dir = Path(candidate.get('snapshot_dir', root / 'suffixes' / context / 'snapshot'))
            if not snapshot_dir.is_absolute():
                snapshot_dir = root / snapshot_dir
            entry = dict(state_sha256=candidate['state_sha256'], context_sha256=context, snapshot_dir=str(snapshot_dir.resolve()))
            snap = _snapshot(entry)
            _verify_prefix(candidate, snap, root)
            protocol = candidate.get('acquisition_protocol_sha256') or canonical_sha256(dict(schema=PROVENANCE_SCHEMA,
                controller=candidate['controller'], prefix_file_sha256=candidate['prefix_file_sha256']))
            key = candidate_key(entry['state_sha256'], context, protocol)
            provenance = dict(candidate_file=str(source), candidate_receipt_sha256=candidate['receipt_sha256'], candidate=deepcopy(candidate))
            if key not in pool['entries']:
                pool['entries'][key] = dict(entry, key=key, acquisition_protocol_sha256=protocol,
                    acquisition_identity_schema=PROVENANCE_SCHEMA, root_cell=candidate['cell'],
                    origin_source_pi=declaration['spec']['proposer'], origin_actor_sha256=candidate['controller']['base_actor_sha256'],
                    phase='upstream' if snap.active_phase == 0 else 'downstream', first_discovery_round=round_index,
                    status='pending', observations=[], sources=[])
            existing = pool['entries'][key]
            if existing['root_cell'] != candidate['cell']:
                raise ValueError('same candidate has inconsistent physical cell')
            if not any(p['candidate_receipt_sha256'] == candidate['receipt_sha256'] for p in existing['sources']):
                existing['sources'].append(provenance)
            _seal(pool)
            result_path = root / 'suffixes' / context / 'result.json'
            if candidate.get('suffix_receipt_sha256') is not None:
                result = json.loads(result_path.read_text())
                if result.get('receipt_sha256') != candidate['suffix_receipt_sha256'] or result.get('label') != candidate.get('label') or result.get('witness') != candidate.get('witness'):
                    raise ValueError('candidate suffix receipt mismatch')
                for attempt in result.get('attempts', []):
                    attempt_dir = result_path.parent / attempt['evaluator']
                    if json.loads((attempt_dir / 'receipt.json').read_text()) != attempt:
                        raise ValueError('saved evaluator receipt mismatch')
                    for filename, expected_sha in attempt.get('files', {}).items():
                        artifact = (attempt_dir / filename).resolve()
                        if not artifact.is_relative_to(attempt_dir.resolve()) or _sha(artifact) != expected_sha:
                            raise ValueError('saved evaluator artifact hash mismatch')
                pool = add_evaluation(pool, key, result, round_index=round_index)
            elif candidate.get('label') is not None:
                raise ValueError('candidate label lacks suffix receipt')
    return _seal(pool)


def select_pending(pool, limit):
    """Deterministic phase-balanced selection of complete pending reset contexts."""
    validate_pool(pool)
    if type(limit) is not int or limit < 0:
        raise ValueError('limit must be nonnegative')
    groups = {}
    for key, entry in sorted(pool['entries'].items()):
        if entry['status'] == 'pending':
            groups.setdefault(entry['phase'], []).append(entry)
    selected = []
    while groups and len(selected) < limit:
        for phase in sorted(list(groups)):
            selected.append(deepcopy(groups[phase].pop(0)))
            if not groups[phase]:
                del groups[phase]
            if len(selected) == limit:
                break
    return selected
