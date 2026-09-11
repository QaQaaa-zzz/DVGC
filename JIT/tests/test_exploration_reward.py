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
