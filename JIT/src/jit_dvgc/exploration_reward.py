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


def trajectory_reward_batch(ledger, data, episodes, *, expected_policy_sha256, weights):
    """Full real-arrival novelty plus explicit physical-failure and delta costs.

    Candidate export caps do not affect learning. Inactive simulator padding never
    earns reward or penalty; each episode pays failure once. This is provisional
    exploration reward, not an envelope admission rule or a future-learnability label.
    """
    if ledger.get('policy_sha256') != expected_policy_sha256:
        raise ValueError('source policy identity mismatch')
    if set(weights) != {'novelty','physical_failure','residual_energy'} or any(
        type(v) not in (int,float) or not np.isfinite(v) or v<0 for v in weights.values()):
        raise ValueError('finite nonnegative explicit reward weights required')
    mask=np.asarray(data['mask'],bool)
    if mask.ndim!=2 or len(episodes)!=mask.shape[1]:raise ValueError('episode shape mismatch')
    finite=np.asarray(data['finite'],bool);failure=np.asarray(data['physical_failure'],bool)
    terminal=np.asarray(data['terminal'],bool);delta=np.asarray(data['normalized_delta'])
    if finite.shape!=mask.shape or failure.shape!=mask.shape or terminal.shape!=mask.shape or delta.shape!=mask.shape+(4,):
        raise ValueError('rollout reward shapes differ')
    if np.any(mask&~finite) or not np.isfinite(delta[mask]).all():raise ValueError('nonfinite active rollout')
    seen=_cells(ledger['cells']);visits={}
    novelty=np.zeros(mask.shape);penalty=np.zeros(mask.shape)
    for e,episode in enumerate(episodes):
        local=set()
        for tick in np.flatnonzero(mask[:,e]):
            if terminal[tick,e] or failure[tick,e]:continue
            cell=episode['cells'][tick]
            if cell not in seen and cell not in local:
                visits.setdefault(cell,[]).append((int(tick),e));local.add(cell)
        failed=np.flatnonzero(mask[:,e]&failure[:,e])
        if len(failed):penalty[failed[0],e]=-weights['physical_failure']
    for occurrences in visits.values():
        for tick,e in occurrences:novelty[tick,e]+=weights['novelty']/len(occurrences)
    energy=np.zeros(mask.shape)
    energy[mask]=-weights['residual_energy']*np.mean(delta[mask]**2,axis=-1)
    parts=dict(novelty=novelty,physical_failure=penalty,residual_energy=energy)
    updated=dict(policy_sha256=expected_policy_sha256,cells=sorted(seen|set(visits)))
    evidence=dict(schema='jit_full_trajectory_quality_reward_v1',envelope_admission=False,
        policy_sha256=expected_policy_sha256,weights=weights,ledger_before_sha256=_hash(ledger),
        ledger_after_sha256=_hash(updated),novel_cells=sorted(visits),novel_cell_count=len(visits),
        reward_assignment='first_occurrence_per_episode_shared_per_new_cell',
        component_sums={k:float(v.sum()) for k,v in parts.items()},total_reward=float(sum(parts.values()).sum()),
        candidate_cap_affects_reward=False,physical_failure_count=int(np.sum(np.any(mask&failure,axis=0))))
    return sum(parts.values()),updated,evidence,parts
