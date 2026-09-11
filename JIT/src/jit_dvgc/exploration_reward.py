"""Host-side, source-policy-scoped novelty credit.

``continuation_reward_batch`` pays at candidate ticks using witnessed frozen-bank
suffixes. ``reward_batch`` preserves historical clean-episode terminal rewards.
Cells are exact canonical physical-cell strings; each new cell pays one unit.
"""

from collections import Counter
import hashlib
import json
import re

import numpy as np


def _cells(value):
    if not isinstance(value, list) or any(
        not isinstance(cell, str) or not cell or cell != cell.strip()
        for cell in value
    ):
        raise ValueError('cells must be a list of nonempty exact cell strings')
    return set(value)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def reward_batch(ledger, episodes, *, expected_policy_sha256):
    """Return terminal rewards, a new ledger, and complete credit evidence.

    The caller must supply the frozen current-policy identity independently of
    the ledger. Persist the returned ledger with the optimizer state; never
    initialize it from a multi-policy union. Invalid metadata raises; valid
    failure, incomplete, conflicting or nonfinite outcomes earn zero.
    """
    if not isinstance(ledger, dict) or set(ledger) != {'policy_sha256', 'cells'}:
        raise ValueError('ledger requires exactly policy_sha256 and cells')
    policy = ledger['policy_sha256']
    if (not isinstance(policy, str) or re.fullmatch('[0-9a-f]{64}', policy) is None
            or policy != expected_policy_sha256):
        raise ValueError('current policy identity mismatch or invalid SHA256')
    seen = _cells(ledger['cells'])
    if not isinstance(episodes, list):
        raise ValueError('episodes must be a list')
    eligible = []
    episode_cells = []
    flags_list = []
    for episode in episodes:
        if not isinstance(episode, dict):
            raise ValueError('episode must be a mapping')
        flags = {key: episode.get(key) for key in ('success', 'physical_failure', 'completed')}
        flags['finite'] = episode.get('finite', True)
        if any(type(value) is not bool for value in flags.values()):
            raise ValueError('episode outcome flags must be explicit booleans')
        cells = _cells(episode.get('cells'))
        clean = flags['success'] and flags['completed'] and flags['finite'] and not flags['physical_failure']
        eligible.append(clean)
        episode_cells.append(cells - seen if clean else set())
        flags_list.append(flags)
    counts = Counter(cell for cells in episode_cells for cell in cells)
    rewards = np.zeros(len(episodes), dtype=np.float64)
    records = []
    for index, cells in enumerate(episode_cells):
        credits = {cell: 1. / counts[cell] for cell in sorted(cells)}
        rewards[index] = sum(credits.values())
        records.append(dict(episode_index=index, **flags_list[index],
                            eligible=eligible[index], cells=sorted(set(episodes[index]['cells'])),
                            cell_credits=credits, terminal_reward=float(rewards[index])))
    before = dict(policy_sha256=policy, cells=sorted(seen))
    updated = dict(policy_sha256=policy, cells=sorted(seen | set(counts)))
    evidence = dict(schema='jit_per_policy_exploration_reward_v1', policy_sha256=policy,
                    ledger_before_sha256=_hash(before), ledger_after_sha256=_hash(updated),
                    novelty_credit='one_per_new_cell_shared_equally_across_clean_successful_episodes',
                    reward_assignment='terminal_transition', episodes=records,
                    novel_cells=sorted(counts), novel_cell_count=len(counts),
                    total_reward=float(rewards.sum()))
    return rewards, updated, evidence


def _candidate_reward_batch(ledger, candidates, *, shape, expected_policy_sha256, arrival=False):
    """Pay candidate-time novelty after SAME-context frozen-bank validation.

    ``shape`` is (num_steps, num_episodes); ``tick`` is the zero-based transition
    receiving credit. Source-policy identity scopes the novelty ledger. Its
    initial cells are the declared frozen source baseline, not a bank union.
    A bank policy supplies a continuation witness, not a source-baseline cell.
    Parent episode outcome is irrelevant: the perturbed trajectory may fail
    after proposing a state from which a frozen evaluator succeeds.

    The caller must validate the actual continuation receipt and its complete
    state/context identities before calling. This function validates metadata,
    not simulator evidence. Label None remains unknown; neither None nor zero
    consumes a cell. Persist the returned ledger with optimizer state. Exact
    duplicate candidate identities are rejected; distinct sampled occurrences
    share one credit per new physical cell, independent of batch order.
    """
    if not isinstance(ledger, dict) or set(ledger) != {'policy_sha256', 'cells'}:
        raise ValueError('ledger requires exactly policy_sha256 and cells')
    policy = ledger['policy_sha256']
    if (not isinstance(policy, str) or re.fullmatch('[0-9a-f]{64}', policy) is None
            or policy != expected_policy_sha256):
        raise ValueError('source policy identity mismatch or invalid SHA256')
    seen = _cells(ledger['cells'])
    if (not isinstance(shape, (tuple, list)) or len(shape) != 2
            or any(type(size) is not int or size <= 0 for size in shape)):
        raise ValueError('shape must contain positive num_steps and num_episodes')
    if not isinstance(candidates, list):
        raise ValueError('candidates must be a list')
    identities = set()
    records = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError('candidate must be a mapping')
        index, tick = candidate.get('episode_index'), candidate.get('tick')
        if (type(index) is not int or not 0 <= index < shape[1]
                or type(tick) is not int or not 0 <= tick < shape[0]):
            raise ValueError('candidate episode_index or tick outside shape')
        cell = candidate.get('cell')
        _cells([cell])
        label = candidate.get('label')
        if 'label' not in candidate or (label is not None and
                                      (type(label) is not int or label not in (0, 1))):
            raise ValueError('candidate label must be explicit 1, 0 or None')
        for field in ('state_sha256', 'context_sha256', 'receipt_sha256'):
            value = candidate.get(field)
            if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
                raise ValueError(f'candidate {field} must be a SHA256')
        witness = candidate.get('witness')
        if label == 1 and (not isinstance(witness, str) or not witness.strip()
                           or witness != witness.strip()):
            raise ValueError('successful continuation requires an evaluator witness')
        identity = (index, tick, candidate['state_sha256'], candidate['context_sha256'])
        if identity in identities:
            raise ValueError('duplicate candidate identity')
        identities.add(identity)
        records.append(dict(episode_index=index, tick=tick, cell=cell, label=label,
                            state_sha256=candidate['state_sha256'],
                            context_sha256=candidate['context_sha256'],
                            receipt_sha256=candidate['receipt_sha256'], witness=witness,
                            eligible=arrival or label == 1, credit=0.))
    counts = Counter(row['cell'] for row in records
                     if row['eligible'] and row['cell'] not in seen)
    rewards = np.zeros(shape, dtype=np.float64)
    for row in records:
        if row['eligible'] and row['cell'] in counts:
            row['credit'] = 1. / counts[row['cell']]
            rewards[row['tick'], row['episode_index']] += row['credit']
    before = dict(policy_sha256=policy, cells=sorted(seen))
    updated = dict(policy_sha256=policy, cells=sorted(seen | set(counts)))
    evidence = dict(schema=('jit_arrival_exploration_reward_v1' if arrival else 'jit_continuation_exploration_reward_v1'),
                    policy_sha256=policy, ledger_before_sha256=_hash(before),
                    ledger_after_sha256=_hash(updated),
                    novelty_credit=('one_per_new_arrival_cell_shared_across_candidates' if arrival else 'one_per_new_cell_shared_across_witnessed_candidates'),
                    envelope_admission=False if arrival else 'requires_validated_suffix',
                    reward_assignment='candidate_tick', parent_episode_success_required=False,
                    candidates=records, novel_cells=sorted(counts),
                    novel_cell_count=len(counts), total_reward=float(rewards.sum()))
    return rewards, updated, evidence


def continuation_reward_batch(ledger, candidates, *, shape, expected_policy_sha256):
    """Candidate-time reward requiring an independently validated suffix witness."""
    return _candidate_reward_batch(ledger, candidates, shape=shape,
        expected_policy_sha256=expected_policy_sha256)


def arrival_reward_batch(ledger, candidates, *, shape, expected_policy_sha256):
    """Provisional novelty, not envelope membership or a successful suffix label.

    Caller validates real complete prefix/snapshot provenance. Bank failure and
    deferred labels remain unchanged; this ledger tracks arrivals for this pi.
    Later evaluation only changes the candidate pool, never old PPO samples.
    """
    for candidate in candidates:
        if candidate.get('generated_by_env_step_only') is not True:
            raise ValueError('arrival reward requires real forward provenance')
    return _candidate_reward_batch(ledger, candidates, shape=shape,
        expected_policy_sha256=expected_policy_sha256, arrival=True)
