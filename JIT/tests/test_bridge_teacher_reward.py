"""Teacher reward must credit verified G recovery, never student outcomes."""
from copy import deepcopy

import numpy as np
import pytest

from jit_dvgc.generative_bridge import rewards


def arrival(root, cells=None, **fields):
    return dict(root_id=root, pulse_cells=[root] if cells is None else cells,
                stage_reached=True, pulse_applied_steps=3, label=0,
                prefix_file='prefixes.npz', prefix_sha256='prefix-sha', **fields)


def queried(root, solved=True, **fields):
    result = dict(
        root_id=root,
        teacher_status='verified_solution' if solved else 'searched_no_solution',
        source_actor_sha256='source-sha', source_recheck_label=0,
        generator={'checkpoint_manifest_sha256': 'generator-sha'},
        proposal_sha256='proposal-sha',
        candidates=[dict(candidate_id=0, kind='source_only', full_success=False),
                    dict(candidate_id=1, kind='diffusion', full_success=solved,
                         action_delta_cost=.25, task_return=8.)])
    if solved:
        result.update(selected_candidate_id=1,
                      verification=dict(candidate_id=1, full_success=True,
                                        repeat_labels=[1, 1], repeat_complete=[True, True]),
                      demo={'metadata': {'verification_receipt_sha256': 'verification-sha'}})
    result.update(fields)
    return result


def test_verified_teacher_bonus_ignores_student_failure_and_adoption():
    rows = [arrival('new', learning_attempted=False, successor_adopted=False)]
    teacher = {'new': queried('new')}
    original = deepcopy((rows, teacher))
    value, eligible, ledger, parts = rewards.teacher_feedback(rows, [], teacher)
    np.testing.assert_allclose(value, [2.02])
    assert eligible.tolist() == [True] and ledger == ['new']
    assert parts == {'novelty': [.02], 'quality': [2.]}
    assert (rows, teacher) == original


def test_window_novelty_preserves_duplicate_sharing_and_requires_positive_bonus():
    rows = [arrival('a', ['new', 'old', 'old']), arrival('b', ['new']),
            arrival('c', ['old'])]
    teachers = {r['root_id']: queried(r['root_id']) for r in rows}
    value, eligible, ledger, parts = rewards.teacher_feedback(rows, ['old'], teachers)
    np.testing.assert_allclose(value, [2.005, 2.01, -.02])
    np.testing.assert_allclose(parts['quality'], [2., 2., 0.])
    assert eligible.all() and ledger == ['new', 'old']


def test_completed_generated_search_failure_is_negative_even_if_student_succeeded():
    row = arrival('new', learning_attempted=True, successor_adopted=True)
    row['label'] = 1
    value, eligible, _, parts = rewards.teacher_feedback([row], [], {'new': queried('new', False)})
    np.testing.assert_allclose(value, [-.08])
    assert eligible[0] and parts['quality'] == [-.1]


@pytest.mark.parametrize('status', ['not_scheduled', 'incomplete', 'invalid',
                                  'replay_rejected', 'error', 'ambiguous'])
def test_unknown_outcomes_are_excluded_from_ppo_instead_of_pseudo_negatives(status):
    teacher = queried('new', teacher_status=status)
    value, eligible, ledger, parts = rewards.teacher_feedback([arrival('new')], [], {'new': teacher})
    assert value.tolist() == [0.] and eligible.tolist() == [False]
    assert parts['quality'] == [0.] and ledger == ['new']


def test_unqueried_row_has_no_quality_even_with_successful_student():
    row = arrival('new', learning_attempted=True, successor_adopted=True)
    row['label'] = 1
    value, eligible, _, _ = rewards.teacher_feedback([row], [], {})
    assert value.tolist() == [0.] and not eligible[0]


@pytest.mark.parametrize('mutation', ['source_only', 'colored_noise', 'missing_verification',
                                     'wrong_id', 'search_failed', 'review_failed',
                                     'review_incomplete', 'conflict', 'incomplete_candidate'])
def test_claimed_solution_needs_a_generated_full_task_winner_and_verified_reviews(mutation):
    teacher = queried('new')
    if mutation in ('source_only', 'colored_noise'):
        teacher['candidates'][1]['kind'] = mutation
    elif mutation == 'missing_verification':
        del teacher['verification']
    elif mutation == 'wrong_id':
        teacher['verification']['candidate_id'] = 7
    elif mutation == 'search_failed':
        teacher['candidates'][1]['full_success'] = False
    elif mutation == 'review_failed':
        teacher['verification']['repeat_labels'] = [1, 0]
    elif mutation == 'review_incomplete':
        teacher['verification']['repeat_complete'] = [True, False]
    elif mutation == 'conflict':
        teacher['source_recheck_label'] = 1
    else:
        teacher['candidates'][1]['incomplete'] = True
    value, eligible, _, parts = rewards.teacher_feedback([arrival('new')], [], {'new': teacher})
    assert value.tolist() == [0.] and not eligible[0] and parts['quality'] == [0.]


def test_source_only_success_cannot_be_pseudo_negative_generated_failure():
    teacher = queried('new', False)
    teacher['candidates'][0]['full_success'] = True
    value, eligible, _, _ = rewards.teacher_feedback([arrival('new')], [], {'new': teacher})
    assert value.tolist() == [0.] and not eligible[0]


def test_unsolved_incomplete_generated_candidates_are_unknown():
    teacher = queried('new', False)
    teacher['candidates'][1]['invalid'] = True
    value, eligible, _, _ = rewards.teacher_feedback([arrival('new')], [], {'new': teacher})
    assert value.tolist() == [0.] and not eligible[0]


def test_direct_failure_paid_once_with_no_cells_and_unknown_teacher():
    row = arrival('failed', [], prefix_terminal=True, prefix_physical_failure=True)
    value, eligible, ledger, parts = rewards.teacher_feedback([row], [], {})
    assert value.tolist() == [-2.] and eligible.tolist() == [True] and ledger == []
    assert parts == {'novelty': [0.], 'quality': [-2.]}


def test_stage_not_reached_and_zero_pulse_failure_are_not_training_samples():
    rows = [arrival('a', stage_reached_override=False),
            arrival('b', prefix_terminal=True, prefix_physical_failure=True)]
    rows[0]['stage_reached'] = False
    rows[1]['pulse_applied_steps'] = 0
    value, eligible, ledger, _ = rewards.teacher_feedback(rows, [], {})
    assert value.tolist() == [0., 0.] and not eligible.any() and ledger == ['b']


def test_teacher_reward_audit_preserves_identity_and_selected_candidate():
    rows = rewards.teacher_reward_rows([arrival('new')], {'new': queried('new')})
    evidence = rows[0]['teacher_reward_evidence']
    assert evidence['teacher_quality_outcome'] == 1
    assert evidence['selected_candidate_kind'] == 'diffusion'
    assert evidence['selected_candidate_id'] == 1
    assert evidence['source_actor_sha256'] == 'source-sha'
    assert evidence['generator_checkpoint_manifest_sha256'] == 'generator-sha'
    assert evidence['verification_receipt_sha256'] == 'verification-sha'
    assert rows[0]['prefix_sha256'] == 'prefix-sha'
    assert evidence['selected_action_delta_cost'] == .25


def test_missing_window_provenance_is_rejected():
    row = arrival('new')
    del row['pulse_cells']
    with pytest.raises(ValueError, match='pulse_cells'):
        rewards.teacher_feedback([row], [], {'new': queried('new')})


def test_legacy_default_remains_student_credit_and_independent_from_teacher():
    row = arrival('new', initial_label=0, learning_attempted=True, successor_adopted=True)
    row.update(label=1, teacher_status='searched_no_solution')
    value, eligible, _, _ = rewards.feedback([row], [])
    np.testing.assert_allclose(value, [2.02])
    assert eligible[0]
