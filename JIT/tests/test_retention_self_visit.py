import pytest

from jit_dvgc import retention_self_visit as sv


def test_budget_includes_all_capture_and_fixed_repeat_worlds():
    b = sv.budget(877514)
    assert b['capture'] == 8 * 17 * 400
    assert b['queries'] == 24 * 3 * 17 * 400
    assert b['total'] == 544000
    assert b['cumulative_maximum'] == 1421514
    with pytest.raises(ValueError):
        sv.budget(1600000)


def test_only_complete_train_roots_and_original_remaining_window():
    rows = [dict(root=dict(role='TRAIN', data_role='train', root_id=str(i)), lane=1)
            for i in range(8)]
    sv.validate_roots(rows)
    assert [sv.remaining_window(j) for j in (4, 8, 12)] == [12, 8, 4]
    with pytest.raises(ValueError):
        sv.remaining_window(16)
    rows[2]['root']['role'] = 'SOLVER_DEV'
    with pytest.raises(ValueError):
        sv.validate_roots(rows)


def test_source_recovery_and_generator_increment_are_separate():
    assert sv.classify([1, 1, 1], [1, 1, 1]) == 'source_recoverable'
    assert sv.classify([0, 0, 0], [1, 1, 1]) == 'G_incremental_verified'
    assert sv.classify([0, 1, 0], [1, 1, 1]) == 'source_ambiguous'
    assert sv.classify([0, 0, 0], [1, 0, 1]) == 'winner_ambiguous'
    assert sv.classify([0, 0, 0], None) == 'searched_no_solution'
    with pytest.raises(ValueError):
        sv.classify([0, None, 0], None)


def test_execution_authorization_is_bound_to_plan_and_new_stage(tmp_path):
    from jit_dvgc.retention_repair import write, sha
    plan = tmp_path / 'plan.json'
    grant = tmp_path / 'grant.json'
    write(plan, {'schema': sv.SCHEMA, 'training_updates': 0})
    write(grant, {'schema': sv.GRANT_SCHEMA, 'plan_sha256': sha(plan),
                  'execute': True, 'source': 'user explicitly requested implementation and start'})
    sv.require_grant(plan, grant)
    write(plan, {'schema': sv.SCHEMA, 'training_updates': 1})
    with pytest.raises(ValueError):
        sv.require_grant(plan, grant)


def test_no_search_winner_but_replay_positive_is_ambiguous_not_no_solution():
    labels = [[0] * 17 for _ in range(3)]
    labels[1][7] = 1
    assert sv.classify_query([0, 0, 0], None, labels) == 'search_no_winner_replay_positive'
    assert sv.classify_query([0, 0, 0], None, [[0] * 17] * 3) == 'searched_no_solution'


def test_partial_or_duplicate_query_matrix_never_claims_complete():
    roots = [str(i) for i in range(8)]
    rows = [dict(root_id=r, step=j) for r in roots for j in (4, 8, 12)]
    assert sv.coverage(rows, roots)['complete']
    assert not sv.coverage(rows[:-1], roots)['complete']
    with pytest.raises(ValueError):
        sv.coverage(rows + rows[:1], roots)
