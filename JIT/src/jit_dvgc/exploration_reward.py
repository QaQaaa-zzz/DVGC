"""Host-side, policy-scoped novelty credit for completed explorer episodes.

Cells are exact canonical physical-cell strings. Each new cell contributes one
unit across all clean successful episodes in this batch, shared equally. The
returned rewards belong at terminal transitions, before the PPO update.
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
