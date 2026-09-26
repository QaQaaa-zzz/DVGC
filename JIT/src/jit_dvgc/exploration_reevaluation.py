"""Budgeted delayed TRAIN feedback for a durable reached-candidate pool."""
from copy import deepcopy
import json
from pathlib import Path
import time

from .evidence_integrity import canonical_sha256
from .exploration_pool import validate_pool, select_pending, add_evaluation, _snapshot, _sha
from .probe_bank import load_probe_bank, _write_new

SCHEMA = 'jit_pending_pool_reevaluation_v1'


def _seal(doc, field='receipt_sha256'):
    doc[field] = canonical_sha256(doc)
    return doc


def run(pool_path, bank_path, *, order, horizon, budget, output, round_index, limit=None, keys=None):
    """Evaluate a declared finite selection, retaining unknowns and old bank labels.

    ``keys`` explicitly includes any requested canonical entries (even witnessed);
    otherwise select phase-balanced pending entries. A subset ``order`` defines
    a new bank version; its failures never imply failure under the parent bank.
    """
    from .exploration_continuation import FrozenSuffixEvaluator
    if type(round_index) is not int or round_index < 0:
        raise ValueError('round_index must be nonnegative')
    if type(budget) is not int or budget < 0 or type(horizon) is not int or horizon <= 0:
        raise ValueError('invalid finite suffix budget/horizon')
    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError('limit must be nonnegative')
    pool_path, bank_path, output = Path(pool_path).resolve(), Path(bank_path).resolve(), Path(output).resolve()
    pool = validate_pool(json.loads(pool_path.read_text()))
    bank = load_probe_bank(bank_path)
    evaluators = {m['name'] for m in bank['members'] if 'evaluator' in m['roles']}
    order = list(order)
    if not order or len(set(order)) != len(order) or not set(order) <= evaluators:
        raise ValueError('order must be a nonempty unique subset of declared evaluators')
    if horizon != bank['max_ticks']:
        raise ValueError('horizon differs from locked bank')
    if keys is None:
        selected = select_pending(pool, len(pool['entries']) if limit is None else limit)
    else:
        keys = list(keys)
        if len(set(keys)) != len(keys) or any(k not in pool['entries'] for k in keys):
            raise ValueError('requested keys must be unique canonical pool entries')
        selected = [deepcopy(pool['entries'][k]) for k in sorted(keys)]
        if limit is not None:
            selected = selected[:limit]
    if any(round_index < e['first_discovery_round'] for e in selected):
        raise ValueError('reevaluation predates discovery')
    # Verify all selected complete snapshot identities before any interaction.
    for entry in selected:
        _snapshot(entry)
    output.mkdir(parents=True, exist_ok=False)
    effective_bank_path = bank_path
    if set(order) != evaluators:
        subset = deepcopy(bank)
        subset.pop('bank_sha256')
        subset['parent_bank_sha256'] = bank['bank_sha256']
        subset['selection_schema'] = 'jit_explicit_evaluator_subset_v1'
        subset['version'] = bank['version'] + '_subset_' + canonical_sha256(order)[:12]
        subset['members'] = []
        for member in bank['members']:
            member = deepcopy(member)
            if member['name'] not in order:
                member['roles'] = [role for role in member['roles'] if role != 'evaluator']
            if member['roles']:
                subset['members'].append(member)
        subset = _seal(subset, 'bank_sha256')
        effective_bank_path = output / 'evaluation_bank.json'
        _write_new(effective_bank_path, subset)
    else:
        subset = bank
    declaration = _seal(dict(schema=SCHEMA, role='TRAIN', final_test_used=False,
        source_pool=str(pool_path), source_pool_sha256=pool['pool_sha256'], source_pool_file_sha256=_sha(pool_path),
        source_bank=str(bank_path), source_bank_sha256=bank['bank_sha256'], source_bank_file_sha256=_sha(bank_path),
        evaluation_bank_sha256=subset['bank_sha256'], order=order, horizon=horizon, budget=budget,
        selected_keys=[e['key'] for e in selected], round_index=round_index,
        endpoint='unconflicted_first_valid_landing', stop_condition='selection_complete_or_next_full_horizon_unaffordable',
        same_context_reuse='same locked evaluator bank within this run only'))
    _write_new(output / 'declaration.json', declaration)
    evaluator = FrozenSuffixEvaluator(effective_bank_path, order, horizon, output / 'suffixes', budget)
    cache, completed, started = {}, [], time.monotonic()
    status = 'selection_complete'
    try:
        for index, entry in enumerate(selected):
            context = entry['context_sha256']
            if context not in cache and budget - evaluator.charged_interactions < horizon:
                status = 'budget_exhausted'
                break
            if _sha(pool_path) != declaration['source_pool_file_sha256'] or _sha(bank_path) != declaration['source_bank_file_sha256']:
                raise ValueError('declared pool/bank changed during reevaluation')
            before = evaluator.charged_interactions
            reused = context in cache
            if not reused:
                raw = evaluator.evaluate(_snapshot(entry))
                binding = deepcopy(raw)
                binding.pop('receipt_sha256')
                binding.update(source_receipt_sha256=raw['receipt_sha256'],
                    snapshot_dir=str(output / 'suffixes' / context / 'snapshot'),
                    binding_schema='jit_suffix_snapshot_location_binding_v1')
                cache[context] = _seal(binding)
                _write_new(output / 'suffixes' / context / 'pool_binding.json', cache[context])
            result = cache[context]
            if result['bank_sha256'] != subset['bank_sha256'] or set(result['labels']) != set(order):
                raise ValueError('evaluation bank identity mismatch')
            members = {member['name']: member for member in subset['members']}
            for attempt in result['attempts']:
                policy = members[attempt['evaluator']]['policy']
                if attempt.get('actor_sha256') != policy['actor_sha256'] or attempt.get('payload_sha256') != policy['payload_sha256']:
                    raise ValueError('evaluation actor/payload identity mismatch')
            if evaluator.charged_interactions > budget:
                raise ValueError('suffix evaluator exceeded declared budget')
            pool = add_evaluation(pool, entry['key'], result, round_index=round_index)
            completed.append(entry['key'])
            step = output / f'step_{index:06d}'
            _write_new(step / 'pool.json', pool)
            _write_new(step / 'receipt.json', _seal(dict(key=entry['key'], label=result['label'],
                result_receipt_sha256=result['receipt_sha256'], pool_sha256=pool['pool_sha256'],
                charged_interactions=evaluator.charged_interactions-before,
                cumulative_charged_interactions=evaluator.charged_interactions, reused_context=reused,
                wall_seconds=time.monotonic()-started)))
    except Exception as exc:
        _write_new(output / 'failure.json', _seal(dict(status='failed', error=f'{type(exc).__name__}: {exc}',
            charged_interactions=evaluator.charged_interactions, completed_keys=completed,
            pool_sha256=pool['pool_sha256'], wall_seconds=time.monotonic()-started)))
        _write_new(output / 'pool.json', pool)
        raise
    _write_new(output / 'pool.json', pool)
    report = _seal(dict(schema=SCHEMA, status=status, role='TRAIN', final_test_used=False,
        declaration_sha256=declaration['receipt_sha256'], selected_count=len(selected), completed_keys=completed,
        unevaluated_keys=[e['key'] for e in selected if e['key'] not in completed],
        charged_interactions=evaluator.charged_interactions, budget=budget, unique_contexts=len(cache),
        pool_sha256=pool['pool_sha256'], wall_seconds=time.monotonic()-started))
    _write_new(output / 'status.json', report)
    return report
