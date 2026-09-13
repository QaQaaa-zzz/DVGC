from pathlib import Path
import sys
import pytest
from jit_dvgc import checkpoint_comparison as comparison
from jit_dvgc.jump_evidence_validation import read, write
from jit_dvgc.evidence_integrity import canonical_sha256


def template():
    return dict(action_names=['steer'], signs=[-1, 1], strengths=[.1, .2, .3],
                lookbacks_m=[.1], max_forward_ticks=10, max_candidates_per_attempt=3)


def test_schedule_bounds_covers_uneven_partitions():
    bounds=comparison.schedule_bounds(template(),[{'proposal_family_index':0},{'proposal_family_index':1}],2)
    assert bounds == dict(trajectories=3, acquisition_ceiling=40, candidate_ceiling=9, label_ceiling=7200)


@pytest.mark.parametrize('field,value', [('max_forward_ticks',-1),('max_candidates_per_attempt',False),('strengths',[-.1]),('lookbacks_m',[float('nan')])])
def test_schedule_rejects_invalid_budget_inputs(field,value):
    spec=template();spec[field]=value
    with pytest.raises(ValueError):comparison.schedule_bounds(spec,[{'proposal_family_index':0}],1)


def test_schedule_rejects_invalid_partition_even_if_other_anchor_has_work():
    with pytest.raises(ValueError):comparison.schedule_bounds(template(),[{'proposal_family_index':0},{'proposal_family_index':99}],1)


def test_worker_failure_and_timeout_preserve_reservation(tmp_path):
    for name,code,timeout in [('exit','raise SystemExit(3)',5),('timeout','import time; time.sleep(10)',.05)]:
        output=tmp_path/name
        error=comparison._worker([sys.executable,'-c',code],output,37,timeout)
        assert error
        assert read(output/'reservation.json')['maximum_interactions']==37
        assert read(output/'exit.json')['error']


def setup_runner(tmp_path,monkeypatch):
    import jit_dvgc.probe_bank as probe
    import jit_dvgc.continuation.existence as existence
    baseline=tmp_path/'baseline.csv';baseline.write_text('root_cell,witnessed\n')
    plan=dict(plan_sha256='plan',spec=dict(bank='bank',proposers=['a','b'],acquisition={},
        timeout_seconds=1,per_arm_budget=1000,evaluator_order=['a'],label_seed=1,
        training_surcharges={'a':0,'b':0},baseline_csv=str(baseline)),bounds={'acquisition_ceiling':100,'trajectories':1})
    monkeypatch.setattr(comparison,'verify_plan',lambda _:plan)
    monkeypatch.setattr(probe,'load_probe_bank',lambda _:dict(members=[dict(name=n,policy={}) for n in ['a','b']]))
    def prepare(*args,**kwargs):
        write(args[2],{'plan_sha256':'labels'})
    monkeypatch.setattr(existence,'prepare',prepare)
    return plan


def test_failed_acquisition_keeps_full_charge_and_stops_other_arms(tmp_path,monkeypatch):
    setup_runner(tmp_path,monkeypatch)
    monkeypatch.setattr(comparison,'_worker',lambda *args:'timeout')
    result=comparison._run(tmp_path,'0')
    assert result['status']=='engineering_error'
    assert result['charged_interactions']==100
    assert not (tmp_path/'b').exists()


def test_invalid_acquisition_identity_keeps_reservation(tmp_path,monkeypatch):
    setup_runner(tmp_path,monkeypatch)
    def worker(command,output,*args):
        write(output/'result/catalog.json',dict(status='completed',environment_interactions=5))
    monkeypatch.setattr(comparison,'_worker',worker)
    def project(*args):raise ValueError('catalog identity drift')
    monkeypatch.setattr(comparison,'project_catalog',project)
    result=comparison._run(tmp_path,'0')
    assert result['status']=='engineering_error'
    assert result['charged_interactions']==100


def test_label_supervisor_error_cannot_reduce_reservation_using_existing_index(tmp_path,monkeypatch):
    setup_runner(tmp_path,monkeypatch)
    def worker(command,output,*args):
        if output.name=='acquire':
            write(output/'result/catalog.json',dict(status='completed',environment_interactions=5))
            return None
        index=dict(plan_sha256='labels',charged_interactions=0,status='partial_unknown',entries=[],attempts=[])
        index['index_sha256']=canonical_sha256(index)
        write(output/'result/witness_index.json',index)
        return 'timeout'
    monkeypatch.setattr(comparison,'_worker',worker)
    monkeypatch.setattr(comparison,'project_catalog',lambda *args:[{}])
    result=comparison._run(tmp_path,'0')
    assert result['status']=='engineering_error'
    assert result['charged_interactions']==1000
    assert not (tmp_path/'b').exists()


def test_witness_receipt_rejects_unverified_attempt_and_identity_error(tmp_path):
    assert hasattr(comparison,'validate_witness_receipt'), 'witness receipt validation missing'
    plan=dict(plan_sha256='p',bank_sha256='b',candidate_indices=[0],evaluator_order=['a'],horizon=400)
    catalog={'entries':[dict(candidate_id='c',state_sha256='s',phase='upstream')]}
    for field,value in [('execution_identity_error','source drift'),('charged_interactions',True),('bank_sha256','wrong')]:
        index=dict(schema='jit_first_success_witness_index_v1',plan_sha256='p',bank_sha256='b',
            status='partial_unknown',execution_identity_error=None,charged_interactions=0,attempts=[],entries=[],
            final_test_used=False,training_transitions=0)
        index[field]=value
        with pytest.raises(ValueError):
            comparison.validate_witness_receipt(index,plan,catalog,{}, {},tmp_path,1000)


def test_witness_receipt_keeps_untested_unknown(tmp_path):
    assert hasattr(comparison,'validate_witness_receipt'), 'witness receipt validation missing'
    from jit_dvgc.continuation.existence import aggregate_candidate
    plan=dict(plan_sha256='p',bank_sha256='b',candidate_indices=[0],evaluator_order=['a'],horizon=400)
    catalog={'entries':[dict(candidate_id='c',state_sha256='s',phase='upstream')]}
    entry={**catalog['entries'][0],'candidate_index':0,**aggregate_candidate(['a'],{})}
    index=dict(schema='jit_first_success_witness_index_v1',plan_sha256='p',bank_sha256='b',status='partial_unknown',
        execution_identity_error=None,charged_interactions=0,attempts=[],entries=[entry],final_test_used=False,training_transitions=0)
    costs=comparison.validate_witness_receipt(index,plan,catalog,{}, {},tmp_path,1000)
    assert costs==[0]
    index['entries'][0]['label']=0
    with pytest.raises(ValueError,match='witness'):
        comparison.validate_witness_receipt(index,plan,catalog,{}, {},tmp_path,1000)


def test_alias_arms_resolve_same_base_with_distinct_controllers():
    members={'pi_6':dict(roles=['proposer','evaluator'])}
    spec=dict(proposers=['fixed','residual'],proposer_members={'fixed':'pi_6','residual':'pi_6'},residual_explorers={'residual':'/tmp/explorer.json'})
    assert comparison.resolve_proposer_members(spec,members)=={'fixed':'pi_6','residual':'pi_6'}
    for bad in [{}, {'fixed':'missing','residual':'pi_6'}]:
        with pytest.raises(ValueError):comparison.resolve_proposer_members({**spec,'proposer_members':bad},members)
    with pytest.raises(ValueError):comparison.resolve_proposer_members({**spec,'proposers':['../fixed','residual']},members)


def test_acquisition_arm_override_preserves_shared_schedule():
    spec=dict(bank='b',proposers=['fixed','residual'],proposer_members={'fixed':'pi_6','residual':'pi_6'},residual_explorers={'residual':'model'},acquisition={'strengths':[.15],'record_action_tape':True})
    fixed=comparison.arm_acquisition(spec,'fixed',100)
    learned=comparison.arm_acquisition(spec,'residual',100)
    assert fixed==dict(bank='b',proposer='pi_6',interaction_ceiling=100,strengths=[.15],record_action_tape=True)
    assert learned==dict(**fixed,residual_explorer='model')


def test_default_arms_preserve_existing_acquisition():
    spec=dict(bank='b',proposers=['a','b'],acquisition={'seed':1})
    assert comparison.arm_acquisition(spec,'a',100)==dict(seed=1,bank='b',proposer='a',interaction_ceiling=100)
