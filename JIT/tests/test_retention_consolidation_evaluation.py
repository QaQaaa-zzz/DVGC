import pytest
from jit_dvgc import retention_consolidation_evaluation as evaluation


def config():
    return dict(required_groups={'old_nominal': dict(max_lost=0, max_rate_drop=0.),
                    'accepted_parent_recovery': dict(max_lost=0, max_rate_drop=0.)},
                mixed_policy='block', unknown_policy='block', min_expansion_gain=1)


def row(case, group='old_nominal', parent=1, student=1, pi0=1, role='DEV_RETENTION'):
    return dict(case_id=case, condition_identity=case, group=group, role=role,
                parent_labels=[parent]*3, student_labels=[student]*3,
                pi0_labels=[pi0]*3, execution='independent_complete')


def candidate(attempt=100, **changes):
    rows = [row('old'), row('parent_recovery', 'accepted_parent_recovery', pi0=0),
            row('new', 'expansion', parent=0, pi0=0, role='DEV_EXPANSION')]
    rows[-1]['teacher_repeat_labels'] = [1]*3
    return dict(checkpoint=f'checkpoint{attempt}', attempt=attempt,
                identity_valid=True, data_valid=True, numerical_valid=True,
                rows=rows, **changes)


def test_three_repeats_check_stability_not_training_seed_count():
    assert evaluation.stable_repeat_label([1,1,1]) == 1
    assert evaluation.stable_repeat_label([1,0,1]) == 'mixed'
    assert evaluation.stable_repeat_label([1,None,1]) == 'unknown'
    assert evaluation.stable_repeat_label([1,'timeout',1]) == 'timeout'
    assert evaluation.stable_repeat_label([1,'execution_error',None]) == 'execution_error'
    with pytest.raises(ValueError): evaluation.stable_repeat_label([1,1])
    with pytest.raises(ValueError): evaluation.stable_repeat_label([True]*3)


def test_lost_gained_do_not_offset_and_pi0_stays_separate():
    rows = candidate()['rows'] + [row('loss', parent=1, student=0)]
    result = evaluation.evaluate_capability_groups(rows, config())
    assert result['parent']['old_nominal']['lost'] == 1
    assert result['pi0']['accepted_parent_recovery']['gained'] == 1
    assert not result['eligible']
    assert result['expansion']['gained'] == 1


def test_unresolved_categories_not_lost_and_block_adoption():
    for label in ('mixed', 'unknown', 'timeout', 'execution_error'):
        rows = candidate()['rows']
        rows[0]['student_labels'] = ([1,0,1] if label == 'mixed' else [label]*3)
        result = evaluation.evaluate_capability_groups(rows, config())
        assert result['parent']['old_nominal'][label] == 1
        assert result['parent']['old_nominal']['lost'] == 0
        assert not result['eligible']


def test_composite_teacher_success_is_not_student_absorption():
    rows = candidate()['rows']
    rows[-1]['execution'] = 'teacher_prefix_source_tail'
    result = evaluation.evaluate_capability_groups(rows, config())
    assert result['expansion']['gained'] == 0
    assert result['expansion']['unknown'] == 1


def test_required_parent_recovery_missing_and_rate_drop_gate():
    rows = candidate()['rows'][:1]
    assert not evaluation.evaluate_capability_groups(rows, config())['eligible']
    cfg = config(); cfg['required_groups']['old_nominal']['max_lost'] = 2
    rows = candidate()['rows'] + [row('loss', student=0)]
    assert not evaluation.evaluate_capability_groups(rows, cfg)['eligible']


def test_select_valid_gates_then_expansion_loss_and_earliest():
    bad = candidate(10); bad['rows'][0]['student_labels'] = [0]*3
    invalid = candidate(5); invalid['identity_valid'] = False
    result = evaluation.select_checkpoint([candidate(250), bad, invalid, candidate(100)], 'parent', config())
    assert result['stage_candidate'] == 'checkpoint100'
    assert result['published_policy'] is None
    assert result['learner_last'] == 'checkpoint250'
    assert result['phase'] == 'accepted_dev_improvement'


def test_no_improvement_and_update_zero_keep_parent_not_training_success():
    c = candidate(0)
    result = evaluation.select_checkpoint([c], 'parent', config())
    assert result['phase'] == 'no_accepted_improvement'
    assert result['stage_candidate'] == 'parent'
    c = candidate(); c['rows'][-1]['student_labels'] = [0]*3
    assert evaluation.select_checkpoint([c], 'parent', config())['stage_candidate'] == 'parent'


def test_duplicate_condition_and_test_data_fail_closed():
    rows = candidate()['rows']; rows[1]['condition_identity'] = 'old'
    with pytest.raises(ValueError): evaluation.evaluate_capability_groups(rows, config())
    rows = candidate()['rows']; rows[0]['role'] = 'TEST'
    with pytest.raises(ValueError): evaluation.evaluate_capability_groups(rows, config())


def test_replay_adapter_preserves_witness_rules_and_refuses_prefixes():
    base = dict(bank='bank', proposer='pi0', horizon=400, budget=1200)
    spec = evaluation.build_independent_replay_spec(base, 'rows.json', {'path':'student.pkl'})
    assert spec['preserve_snapshot_episode_context'] is True
    assert spec['warmup_initializer']['path'] == 'student.pkl'
    assert 'actor_witness_eligible' not in spec
    with pytest.raises(ValueError):
        evaluation.build_independent_replay_spec(dict(base, bridge_action_plan={}), 'rows', {})


def test_replay_result_reads_attempt_not_actor_witness_and_preserves_timeout():
    result = dict(label=1, witness=None, attempts=[dict(label=1,outcome='stable_forward_recovery',
                  controller_kind='warmup_candidate', actor_witness_eligible=False)])
    assert evaluation.independent_replay_label(result) == 1
    result['attempts'][0].update(label=0, outcome='timeout')
    assert evaluation.independent_replay_label(result) == 'timeout'
    result['attempts'][0]['controller_kind'] = 'composite_teacher'
    assert evaluation.independent_replay_label(result) == 'unknown'


def test_real_runtime_outcome_names_and_source_actor_provenance():
    result = dict(attempts=[dict(label=0,outcome='timeout_before_landing',actor_witness_eligible=True)])
    assert evaluation.independent_replay_label(result) == 'timeout'
    result['attempts'][0]['outcome'] = 'horizon_exhausted_before_landing'
    assert evaluation.independent_replay_label(result) == 'timeout'
    result['attempts'][0].update(label=1,outcome='first_valid_landing')
    assert evaluation.independent_replay_label(result) == 'unknown'


def test_collect_adapter_keeps_nonfinite_and_incomplete_horizon_separate():
    import numpy as np
    tape = dict(prefix_mask=np.ones((2,1),bool), finite=np.ones((2,1),bool),
                success=np.zeros((2,1),bool), terminal=np.zeros((2,1),bool),
                physical_failure=np.zeros((2,1),bool))
    assert evaluation.independent_collection_label(tape,0,2) == 'timeout'
    tape['success'][0,0] = True
    assert evaluation.independent_collection_label(tape,0,2) == 'unknown'
    tape['finite'][1,0] = False
    assert evaluation.independent_collection_label(tape,0,2) == 'execution_error'


def test_collect_runtime_end_code_timeout_is_not_physical_failure():
    import numpy as np
    tape = dict(prefix_mask=np.ones((1,1),bool), finite=np.ones((1,1),bool),
                success=np.zeros((1,1),bool), terminal=np.ones((1,1),bool),
                physical_failure=np.zeros((1,1),bool), end_code=np.array([[9]]))
    assert evaluation.independent_collection_label(tape,0,400) == 'timeout'
    tape['end_code'][0,0] = 2
    assert evaluation.independent_collection_label(tape,0,400) == 'execution_error'


def test_collection_midframe_success_or_terminal_reset_is_unknown():
    import numpy as np
    tape=dict(prefix_mask=np.ones((2,1),bool),finite=np.ones((2,1),bool),
        success=np.array([[True],[False]]),terminal=np.zeros((2,1),bool),physical_failure=np.zeros((2,1),bool))
    assert evaluation.independent_collection_label(tape,0,2)=='unknown'
    tape['success'][:,0]=[False,True];tape['terminal'][:,0]=[False,True]
    assert evaluation.independent_collection_label(tape,0,2)==1
    tape['terminal'][0,0]=True
    assert evaluation.independent_collection_label(tape,0,2)=='unknown'


def test_expansion_unknown_teacher_cannot_be_adopted_as_absorption():
    for labels in ([None]*3,[1,0,1]):
        c=candidate();c['rows'][-1]['teacher_repeat_labels']=labels
        result=evaluation.select_checkpoint([c],'parent',config())
        assert result['phase']=='no_accepted_improvement'
        assert result['audits'][0]['evaluation']['expansion']['gained']==1


def test_collection_failure_requires_terminal_endpoint():
    import numpy as np
    tape=dict(prefix_mask=np.ones((2,1),bool),finite=np.ones((2,1),bool),
        success=np.zeros((2,1),bool),terminal=np.zeros((2,1),bool),
        physical_failure=np.array([[False],[True]]))
    assert evaluation.independent_collection_label(tape,0,2)=='unknown'
    tape['terminal'][-1,0]=True
    assert evaluation.independent_collection_label(tape,0,2)==0


def test_nominal_seen_training_condition_scoped_gate_is_not_independent_dev():
    rows=candidate()['rows'];rows[0]['role']='TRAIN_RETENTION'
    rows[0]['seen_condition_validation']=dict(schema='frontierbridge_seen_condition_validation_v1',
        training_condition_revisit=True,group='old_nominal',independent_dev=False,source_train_condition_identity='old')
    result=evaluation.evaluate_capability_groups(rows,config())
    assert result['eligible']
    assert result['seen_condition_validation']['old_nominal']['total']==1
    rows[0]['group']='accepted_parent_recovery'
    with pytest.raises(ValueError):evaluation.evaluate_capability_groups(rows,config())
