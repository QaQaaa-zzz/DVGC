import pytest
from jit_dvgc.continuation_benchmark import compare_labels, validate_request


def test_compare_separates_endpoint_and_step_changes():
    row = dict(candidate_index=1, state_sha256='s', snapshot_context_sha256='c',
               label_protocol_sha256='p', label=1, outcome_class='first_valid_landing',
               physical_failure=False, timeout=False, final_active_phase=1,
               environment_interactions=3)
    result = compare_labels([row], [{**row, 'environment_interactions': 4}])
    assert result['endpoint_equal'] and not result['exact_equal']
    assert result['step_difference_count'] == 1
    with pytest.raises(ValueError, match='identity'):
        compare_labels([row], [{**row, 'label_protocol_sha256': 'other'}])
    with pytest.raises(ValueError, match='count'):
        compare_labels([row], [])


def test_budget_covers_all_declared_cases_and_rejects_unbounded_sizes():
    with pytest.raises(ValueError, match='budget'):
        validate_request(16, 400, [8, 32], 100)
    assert validate_request(16, 400, [8, 32], 19200) == 19200
    for sizes in ([], [0], [True], [8, 8]):
        with pytest.raises(ValueError):
            validate_request(16, 400, sizes, 1000000)


def test_missing_attempt_artifacts_keep_failure_charge(tmp_path):
    from jit_dvgc.continuation_benchmark import finalize_label_attempt
    record, reference = finalize_label_attempt(tmp_path,
        dict(status='engineering_error', charged_interactions=6400), None, 6400, 16, 1.)
    assert record['status'] == 'engineering_error'
    assert record['charged_interactions'] == 6400
    assert 'error' in record and reference is None


def test_valid_worker_file_does_not_hide_nonzero_exit(tmp_path):
    from jit_dvgc.continuation_benchmark import finalize_capacity_attempt
    from jit_dvgc.jump_evidence_validation import write
    write(tmp_path/'worker_report.json', {'status':'completed','environment_interactions':2})
    result = finalize_capacity_attempt(tmp_path, dict(status='engineering_error', charged_interactions=6400), 3, None)
    assert result['status']=='engineering_error' and result['charged_interactions']==6400


def test_endpoint_disagreement_is_not_a_completed_comparison(tmp_path):
    from jit_dvgc.continuation_benchmark import finalize_label_attempt
    from jit_dvgc.jump_evidence_validation import write
    row = dict(candidate_index=1, state_sha256='s', snapshot_context_sha256='c',
               label_protocol_sha256='p', label=1, outcome_class='first_valid_landing',
               physical_failure=False, timeout=False, final_active_phase=1, environment_interactions=3)
    write(tmp_path/'result/labels.json', [{**row, 'label':0}])
    write(tmp_path/'result/summary.json',dict(environment_interactions=3,useful_label_interactions=3,
        elapsed_seconds=1.,device_compile_seconds=[.1],snapshot_restore_seconds=.1,device_batch_seconds=[.1]))
    record, _ = finalize_label_attempt(tmp_path,
        dict(status='engineering_error',backend='vectorized',charged_interactions=400),[row],400,1,1.)
    assert record['status']=='endpoint_mismatch' and record['charged_interactions']==3
