import pytest
from jit_dvgc.continuation.existence import evidence_status, aggregate_candidate, replay_first_success


def label(value, cost=3, **extra):
    return dict(label=value, physical_failure=False, valid_contact_seen=bool(value),
                environment_interactions=cost, **extra)


def test_unknown_never_becomes_bank_failure_and_success_short_circuits():
    assert aggregate_candidate(['a','b'],{'a':label(0)})['label'] is None
    assert aggregate_candidate(['a','b'],{'a':label(0),'b':label(0)})['label']==0
    row=aggregate_candidate(['a','b'],{'a':label(1)})
    assert row['label']==1 and row['per_policy_outcomes']['b']['status']=='untested'
    assert evidence_status({**label(1),'physical_failure':True})=='conflict'
    assert aggregate_candidate(['a'],{'a':{**label(1),'physical_failure':True}})['label'] is None


def test_replay_charges_only_attempts_needed_and_retains_unknown():
    matrices={'a':[label(1),label(0),{**label(1),'physical_failure':True}],
              'b':[label(0,7),label(1,7),label(0,7)]}
    report=replay_first_success(matrices,['a','b'])
    assert report['hypothetical_useful_interactions']==23
    assert [r['label'] for r in report['entries']]==[1,1,None]
    assert report['hypothetical_evaluator_calls']==5
    with pytest.raises(ValueError):
        replay_first_success(matrices,['b','b'])


def test_resume_charges_later_attempts_before_retry(monkeypatch, tmp_path):
    from jit_dvgc.continuation import existence as module
    from jit_dvgc.jump_evidence_validation import write
    plan = dict(plan_sha256='p', candidate_indices=[0], evaluator_order=['a','b'],
                proposer='a', catalog=str(tmp_path/'catalog.json'), horizon=10,
                budget=20, max_candidates_per_process=1, backend='serial', batch_size=1,
                timeout_seconds=1, seed=1)
    bank = dict(bank_sha256='bank', members=[dict(name=n, frozen_policy=n) for n in ['a','b']])
    monkeypatch.setattr(module, 'load_plan', lambda _: (plan, bank))
    write(tmp_path/'catalog.json', {'entries':[{'state_sha256':'s'}]})
    output=tmp_path/'output'
    for position,name in enumerate(['a','b']):
        attempt=output/f'evaluator_{position:03d}'/'chunk_000000'/'attempt_0000'
        write(attempt/'reservation.json', dict(plan_sha256='p', evaluator=name,
             candidate_indices=[0], maximum_interactions=10))
        write(attempt/'completion.json', {'status':'engineering_error'})
    def forbidden(*args, **kwargs):
        pytest.fail('resume launched despite fully spent budget')
    monkeypatch.setattr(module.subprocess, 'run', forbidden)
    report=module._run('plan', output, gpu='0', python='python')
    assert report['charged_interactions']==20
    assert report['budget_exhausted'] and report['unknown_count']==1


def test_resume_reuses_later_success_without_retry(monkeypatch, tmp_path):
    from jit_dvgc.continuation import existence as module
    from jit_dvgc.jump_evidence_validation import write
    plan = dict(plan_sha256='p', candidate_indices=[0], evaluator_order=['a','b'],
                proposer='a', catalog=str(tmp_path/'catalog.json'), horizon=10,
                budget=30, max_candidates_per_process=1, backend='serial', batch_size=1,
                timeout_seconds=1, seed=1)
    bank = dict(bank_sha256='bank', members=[dict(name=n, frozen_policy=n) for n in ['a','b']])
    monkeypatch.setattr(module, 'load_plan', lambda _: (plan, bank))
    write(tmp_path/'catalog.json', {'entries':[{'state_sha256':'s'}]})
    output=tmp_path/'output'
    for position,name in enumerate(['a','b']):
        attempt=output/f'evaluator_{position:03d}'/'chunk_000000'/'attempt_0000'
        write(attempt/'reservation.json', dict(plan_sha256='p', evaluator=name,
             candidate_indices=[0], maximum_interactions=10))
        write(attempt/'completion.json', {'status':'completed' if name=='b' else 'engineering_error'})
        write(attempt/'exit.json', {'returncode':0})
    monkeypatch.setattr(module, 'validate_subset', lambda *args: ([label(1)], 3))
    def forbidden(*args, **kwargs):
        pytest.fail('resume retried a candidate with an existing valid witness')
    monkeypatch.setattr(module.subprocess, 'run', forbidden)
    report=module._run('plan', output, gpu='0', python='python')
    assert report['charged_interactions']==13 and report['positive_count']==1


@pytest.mark.parametrize('drift', ['schedule','negative_padding','protocol','count'])
def test_subset_rejects_execution_and_accounting_drift(monkeypatch, tmp_path, drift):
    from jit_dvgc.continuation import existence as module
    from jit_dvgc import policy_family_landing as family
    from jit_dvgc.jump_evidence_validation import write, file_sha
    rows=[dict(candidate_index=0,policy_key_candidate_index=0,environment_interactions=3)]
    write(tmp_path/'labels.json',rows)
    write(tmp_path/'catalog.json', {'entries':[{}]})
    plan=dict(backend='vectorized',batch_size=1,catalog=str(tmp_path/'catalog.json'),horizon=10,seed=1)
    member=dict(policy={},frozen_file_sha256='f')
    report=dict(status='completed_subset',selected_candidate_indices=[0],execution_backend='vectorized',
                batch_size=1,environment_interactions=3,inactive_lane_interactions=0,
                useful_label_interactions=3,candidate_count=1,logical_protocol_sha256='p',
                labels_file_sha256=file_sha(tmp_path/'labels.json'))
    execution=dict(status='completed',selected_candidate_indices=[0],execution_backend='vectorized',
                   batch_size=1,logical_protocol_sha256='p',selected_candidate_count=1,
                   maximum_environment_interactions=10,device_step_schedule='vmap_checked_shared_warp_v2')
    if drift=='schedule': execution['device_step_schedule']='vmap_shared_warp_v1'
    if drift=='negative_padding': report.update(environment_interactions=2,inactive_lane_interactions=-1)
    if drift=='protocol': report['logical_protocol_sha256']='other'
    if drift=='count': report['candidate_count']=2
    write(tmp_path/'summary.json',report);write(tmp_path/'execution.json',execution)
    monkeypatch.setattr(family,'_requested_contract',lambda *args: {})
    monkeypatch.setattr(family,'_verify_cached_contract',lambda *args: {'protocol_sha256':'p'})
    monkeypatch.setattr(family,'_verify_cached_rows',lambda *args: None)
    with pytest.raises(ValueError):
        module.validate_subset(tmp_path,plan,member,member,[0])


def test_worker_source_drift_is_charged_and_stops_next_evaluator(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from jit_dvgc.continuation import existence as module
    from jit_dvgc.jump_evidence_validation import write
    plan=dict(plan_sha256='p', candidate_indices=[0], evaluator_order=['a','b'],
              proposer='a',catalog=str(tmp_path/'catalog.json'),horizon=10,budget=20,
              max_candidates_per_process=1,backend='serial',batch_size=1,timeout_seconds=1,seed=1)
    bank=dict(bank_sha256='bank',members=[dict(name=n,frozen_policy=n) for n in ['a','b']])
    calls=[]
    def load(_):
        if calls: raise ValueError('existence plan input/source drift')
        return plan,bank
    monkeypatch.setattr(module,'load_plan',load)
    def worker(*args,**kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess,'run',worker)
    monkeypatch.setattr(module,'validate_subset',lambda *args: pytest.fail('drifted result accepted'))
    write(tmp_path/'catalog.json',{'entries':[{}]})
    output=tmp_path/'output';output.mkdir()
    report=module._run('plan',output,gpu='0',python='python')
    assert len(calls)==1 and report['charged_interactions']==10
    assert report['unknown_count']==1 and report['execution_identity_error']
