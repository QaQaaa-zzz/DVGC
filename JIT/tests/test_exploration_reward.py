import copy

import numpy as np
import pytest

from jit_dvgc.exploration_reward import reward_batch


POLICY = 'a' * 64


def episode(cells, **flags):
    return {**dict(cells=cells, success=True, physical_failure=False,
                   completed=True, finite=True), **flags}


def test_global_cell_credit_and_immutable_inputs():
    ledger = dict(policy_sha256=POLICY, cells=['old'])
    episodes = [episode(['old', 'a', 'a', 'b']), episode(['a', 'c'])]
    before = copy.deepcopy((ledger, episodes))
    rewards, updated, evidence = reward_batch(ledger, episodes, expected_policy_sha256=POLICY)
    np.testing.assert_array_equal(rewards, [1.5, 1.5])
    assert updated['cells'] == ['a', 'b', 'c', 'old']
    assert (ledger, episodes) == before
    assert evidence['novel_cell_count'] == 3
    assert evidence['episodes'][0]['cell_credits'] == {'a': .5, 'b': 1.}
    assert evidence['ledger_before_sha256'] != evidence['ledger_after_sha256']
    again, same, _ = reward_batch(updated, episodes, expected_policy_sha256=POLICY)
    assert not again.any()
    assert same == updated


def test_batch_permutation_changes_only_episode_order():
    ledger = dict(policy_sha256=POLICY, cells=[])
    episodes = [episode(['a']), episode(['a', 'b']), episode(['b', 'c'])]
    r, new, _ = reward_batch(ledger, episodes, expected_policy_sha256=POLICY)
    reverse, other, _ = reward_batch(ledger, episodes[::-1], expected_policy_sha256=POLICY)
    np.testing.assert_array_equal(r, reverse[::-1])
    assert r.sum() == 3
    assert new == other


@pytest.mark.parametrize('flags', [dict(success=False), dict(physical_failure=True),
                                   dict(completed=False), dict(finite=False)])
def test_ineligible_episodes_do_not_consume_cells(flags):
    r, new, ev = reward_batch(dict(policy_sha256=POLICY, cells=[]),
                             [episode(['x'], **flags), episode(['x'])],
                             expected_policy_sha256=POLICY)
    np.testing.assert_array_equal(r, [0, 1])
    assert new['cells'] == ['x']
    assert ev['episodes'][0]['cell_credits'] == {}
    assert not ev['episodes'][0]['eligible']


def test_identity_mismatch_and_global_count_rejected():
    with pytest.raises(ValueError, match='policy'):
        reward_batch(dict(policy_sha256=POLICY, cells=[]), [], expected_policy_sha256='b'*64)
    with pytest.raises(ValueError):
        reward_batch(dict(policy_sha256=POLICY, cells=[], globalcount=10), [], expected_policy_sha256=POLICY)


@pytest.mark.parametrize('cells', [[float('nan')], [''], [1], 'abc'])
def test_invalid_cell_identity_rejected(cells):
    with pytest.raises(ValueError):
        reward_batch(dict(policy_sha256=POLICY, cells=[]), [episode(cells)], expected_policy_sha256=POLICY)


def test_boolean_flags_are_strict_and_finite_defaults_true():
    ep = episode(['a']); del ep['finite']
    r, _, _ = reward_batch(dict(policy_sha256=POLICY, cells=[]), [ep], expected_policy_sha256=POLICY)
    assert r.tolist() == [1.]
    ep['completed'] = 1
    with pytest.raises(ValueError):
        reward_batch(dict(policy_sha256=POLICY, cells=[]), [ep], expected_policy_sha256=POLICY)


def candidate(cell='new', **updates):
    return dict(dict(episode_index=0, tick=1, cell=cell, state_sha256='b' * 64,
                     context_sha256='c' * 64, label=1, witness='frozen-bank/pi_0',
                     receipt_sha256='d' * 64), **updates)


def continuation(ledger, candidates, shape=(4, 2)):
    from jit_dvgc import exploration_reward
    assert hasattr(exploration_reward, 'continuation_reward_batch')
    return exploration_reward.continuation_reward_batch(
        ledger, candidates, shape=shape, expected_policy_sha256=POLICY)


def test_continuation_rewards_exact_tick_even_if_parent_episode_failed():
    ledger = dict(policy_sha256=POLICY, cells=['baseline'])
    rows = [candidate(success=False, physical_failure=True, completed=False),
            candidate('baseline', episode_index=1, tick=3)]
    before = copy.deepcopy((ledger, rows))
    rewards, updated, evidence = continuation(ledger, rows)
    expected = np.zeros((4, 2)); expected[1, 0] = 1
    np.testing.assert_array_equal(rewards, expected)
    assert updated['cells'] == ['baseline', 'new']
    assert evidence['total_reward'] == 1
    assert (ledger, rows) == before
    again, unchanged, _ = continuation(updated, rows)
    assert not again.any()
    assert unchanged == updated


def test_continuation_unknown_and_failure_do_not_consume_cells():
    ledger = dict(policy_sha256=POLICY, cells=[])
    rows = [candidate(label=None, witness=None), candidate('other', label=0, tick=2, witness=None)]
    rewards, updated, evidence = continuation(ledger, rows)
    assert not rewards.any()
    assert updated == ledger
    assert [row['label'] for row in evidence['candidates']] == [None, 0]
    later, updated, _ = continuation(updated, [candidate()])
    assert later.sum() == 1
    assert updated['cells'] == ['new']


def test_continuation_duplicate_cell_credit_conservation_and_permutation():
    rows = [candidate(), candidate(tick=2), candidate(episode_index=1),
            candidate('other', tick=3, episode_index=1)]
    ledger = dict(policy_sha256=POLICY, cells=[])
    rewards, updated, evidence = continuation(ledger, rows)
    assert rewards[1, 0] == pytest.approx(1 / 3)
    assert rewards[2, 0] == pytest.approx(1 / 3)
    assert rewards[1, 1] == pytest.approx(1 / 3)
    assert rewards[3, 1] == 1
    assert rewards.sum() == 2
    reverse, other, _ = continuation(ledger, rows[::-1])
    np.testing.assert_array_equal(rewards, reverse)
    assert updated == other
    assert evidence['novel_cell_count'] == 2
    with pytest.raises(ValueError, match='duplicate'):
        continuation(ledger, [rows[0], rows[0]])


@pytest.mark.parametrize('updates', [
    dict(state_sha256='invalid'), dict(context_sha256=None), dict(receipt_sha256=''),
    dict(witness=None), dict(witness=' '), dict(cell=''), dict(label=True), dict(label=2),
    dict(tick=-1), dict(tick=4), dict(tick=True), dict(episode_index=2),
])
def test_continuation_invalid_candidate_identity_rejected(updates):
    with pytest.raises(ValueError):
        continuation(dict(policy_sha256=POLICY, cells=[]), [candidate(**updates)])


@pytest.mark.parametrize('shape', [(0, 2), (4,), (4, -1), (4, True)])
def test_continuation_invalid_shape_rejected(shape):
    with pytest.raises(ValueError):
        continuation(dict(policy_sha256=POLICY, cells=[]), [], shape=shape)


def test_continuation_source_policy_identity_required():
    with pytest.raises(ValueError, match='policy'):
        continuation(dict(policy_sha256='f' * 64, cells=[]), [])
