"""New campaign reviews must preserve the fixed generated winner and real layout."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jit_dvgc.generative_bridge import production, teacher_worker
from jit_dvgc.generative_bridge.contracts import file_sha
from jit_dvgc.generative_bridge.protocol import atomic_json


def authorized_spec(tmp_path):
    campaign=tmp_path/'campaign'
    output=campaign/'round_001'
    report=dict(schema='jit_teacher_recovery_campaign_authorization_v1',
                implementation_commit='new-code',campaign_output=str(campaign),rounds=100,
                user_authorized=True,formal_acceptance=False,known_limitations=['development only'],
                max_physics_per_round=3500000,teacher_query_count=128,
                teacher_generated_candidates=16,teacher_reviews=2)
    path=tmp_path/'authorization.json';atomic_json(path,report)
    return dict(recovery_training_contract={'schema':'jit_teacher_recovery_training_v1'},
                implementation_commit='new-code',campaign_output=str(campaign),round_index=1,
                output=str(output),budgets={'max_physics':3500000},
                teacher_recovery_authorization={'path':str(path),'sha256':file_sha(path)},
                teacher_layout='source_control_in_candidate_batch',teacher_colored_noise_candidates=0)


def test_new_campaign_authorization_is_explicit_and_does_not_claim_physical_acceptance(tmp_path):
    spec=authorized_spec(tmp_path)
    teacher_worker.validate_teacher_acceptance(spec)
    for fields in ({'round_index':101},{'round_index':0},{'implementation_commit':'other'},
                   {'campaign_output':str(tmp_path/'other')},{'output':str(tmp_path/'outside')},
                   {'budgets':{'max_physics':3500001}}):
        with pytest.raises(ValueError,match='recovery'):
            teacher_worker.validate_teacher_acceptance({**spec,**fields})
    bad=deepcopy(spec);bad.pop('recovery_training_contract')
    with pytest.raises(ValueError,match='physical acceptance'):
        teacher_worker.validate_teacher_acceptance(bad)


def test_new_authorization_hash_and_missing_receipt_cannot_fall_back(tmp_path):
    spec=authorized_spec(tmp_path)
    spec['teacher_recovery_authorization']['sha256']='wrong'
    with pytest.raises(ValueError,match='recovery'):
        teacher_worker.validate_teacher_acceptance(spec)
    spec.pop('teacher_recovery_authorization')
    with pytest.raises(ValueError,match='recovery'):
        teacher_worker.validate_teacher_acceptance(spec)


def test_query_reservation_charges_source_search_and_two_full_reviews(tmp_path):
    spec=authorized_spec(tmp_path)
    assert teacher_worker.teacher_reservation(spec,128)==2662400
    assert teacher_worker.teacher_reservation(spec,3)==62400
    assert teacher_worker.teacher_reservation({},32)==448000
    with pytest.raises(ValueError,match='128'):
        teacher_worker.teacher_reservation(spec,129)


@pytest.mark.parametrize('recovery', [False,True])
def test_dispatch_reserves_actual_queries_and_checks_completed_root_coverage(tmp_path,recovery):
    from jit_dvgc.generative_bridge.production import read
    spec=authorized_spec(tmp_path)
    if not recovery:
        spec.pop('recovery_training_contract')
        path=tmp_path/'physical.json'
        atomic_json(path,dict(schema='jit_b1_physical_acceptance_v1',implementation_commit='new-code',
                             aba_passed=True,baseline_comparison_passed=True,
                             full_search_replay_passed=True,rng_mapping_passed=True))
        spec['teacher_physical_acceptance']={'path':str(path),'sha256':file_sha(path)}
    root=Path(spec['output']);root.mkdir(parents=True)
    count=128 if recovery else 32
    parent=SimpleNamespace(spec=spec,root=root,panels={'new_roots':[{'root_id':str(i)} for i in range(count)]},costs=[])
    reservations=[]

    def child(name,argv,maximum):
        reservations.append(maximum)
        request=Path(argv[-1]);worker=root/'teacher_worker'
        assert read(worker/'production.json')['budgets']['max_physics']==maximum
        atomic_json(worker/'results.json',{str(i):{'teacher_status':'searched_no_solution'} for i in range(count)})
        (worker/'teachers').mkdir()
        atomic_json(worker/'completed.json',dict(phase='completed',results_sha256=file_sha(worker/'results.json'),
            request_sha256=file_sha(request),charged_interactions=10,active_interactions=10))
        cost={};parent.costs.append(cost);return cost
    parent.child=child
    result=teacher_worker.dispatch_teacher(parent,{'checkpoint_manifest_sha256':'G'})
    assert len(result)==count and reservations==[2662400 if recovery else 448000]
    assert parent.costs[0]['charged_interactions']==10


def fixture_runner(tmp_path,monkeypatch,*,original=0,source=0,search=(2,3),reviews=(2,2),
                   partial_review=False,partial_source=False):
    from jit_dvgc.generative_bridge import worker, diffusion
    import jit_dvgc.unified_envelope_snapshot as snapshots
    monkeypatch.setattr(worker,'generator_reference',lambda spec:'generator-reference')
    monkeypatch.setattr(worker,'generator_template',lambda *args:(None,None,None))
    monkeypatch.setattr(diffusion,'restore_state',lambda *args:dict(ema={},normalizer={'mean':0.,'std':1.}))
    monkeypatch.setattr(diffusion,'ddim_sample',lambda *args:np.zeros((16,16,4),np.float32))
    monkeypatch.setattr(snapshots,'load_unified_envelope_snapshot',lambda path:SimpleNamespace(last_action=np.zeros(4)))
    runner=production.ProductionRunner.__new__(production.ProductionRunner)
    runner.root=tmp_path/'teacher';runner.root.mkdir()
    runner.spec={**authorized_spec(tmp_path),'seed':5,'source_runtime':{'horizon':400}}
    runner.source={'actor_sha256':'actor','normalizer_sha256':'normalizer','xml_sha256':'xml'}
    runner.protocol='protocol'
    base=dict(root_id='r',root_episode_id='episode',label=original,snapshot='snapshot',
              snapshot_context_sha256='context',data_role='train',pulse_cells=['fresh'],
              stage_reached=True,pulse_applied_steps=3)
    runner.panels={'new_roots':[base],'splits':{'episode':'generator_train'}}
    calls=[]

    def evaluate(name,rows,*,prefixes=None,source_only=None):
        calls.append(dict(name=name,ids=[r.get('candidate_id') for r in rows],
                          prefixes=None if prefixes is None else prefixes.copy(),
                          source_only=None if source_only is None else source_only.copy()))
        review='review' in name
        ordinal=sum('review' in c['name'] for c in calls)-1
        winners=search if not review else (reviews[ordinal],)
        labels=[source if r.get('candidate_id',0)==0 else int(r['candidate_id'] in winners) for r in rows]
        n=len(rows);terminal=np.ones(n,bool)
        if (review and partial_review) or (name=='teacher_source' and partial_source):terminal[:]=False
        trace=runner.root/(name+'.npz')
        actions=np.zeros((2,n,4),np.float32)
        for lane,row in enumerate(rows):
            if row.get('candidate_id',0)==3:actions[:,lane]=.1
        np.savez_compressed(trace,mask=np.ones((2,n),bool),done=np.stack((np.zeros(n,bool),terminal)),
                            physical_failure=np.stack((np.zeros(n,bool),np.logical_not(labels))),
                            valid_contact=np.stack((np.zeros(n,bool),np.asarray(labels,bool))),
                            timeout=np.zeros((2,n),bool),normalized_action_executed=actions,
                            actor_observation_before=np.zeros((2,n,76)),action_origin_code=np.ones((2,n),int))
        sha=file_sha(trace)
        return [dict(row,label=label,attempts=[dict(trace=str(trace),trace_sha256=sha,trace_lane=lane,
                    actor_sha256='actor',normalizer_sha256='normalizer',model_sha256='xml',
                    snapshot_context_sha256='context',label=label,steps=2,task_return=10.)])
                for lane,(row,label) in enumerate(zip(rows,labels))]
    runner.evaluate=evaluate
    return runner,calls,dict(checkpoint_manifest=str(tmp_path/'G/manifest.json'),checkpoint_manifest_sha256='G-sha')


def test_generated_winner_is_fixed_and_reviewed_twice_in_identical_17_world_layout(tmp_path,monkeypatch):
    runner,calls,incumbent=fixture_runner(tmp_path,monkeypatch)
    result=runner.teacher_search(incumbent)['r']
    assert result['selected_candidate_id']==2
    assert result['teacher_status']=='verified_solution'
    assert result['verification']['repeat_labels']==[1,1]
    assert result['verification']['repeat_complete']==[True,True]
    assert len(calls)==4
    for review in calls[2:]:
        assert review['ids']==list(range(17))
        np.testing.assert_array_equal(review['prefixes'],calls[1]['prefixes'])
        np.testing.assert_array_equal(review['source_only'],[True]+[False]*16)
    assert result['demo']['count']==2 and result['generated_only_search']
    assert result['source_control_consistent']


def test_failed_first_review_never_reselects_another_successful_candidate(tmp_path,monkeypatch):
    runner,calls,incumbent=fixture_runner(tmp_path,monkeypatch,reviews=(3,2))
    result=runner.teacher_search(incumbent)['r']
    assert result['selected_candidate_id']==2
    assert result['verification']['repeat_labels']==[0,1]
    assert result['teacher_status']=='replay_rejected' and 'demo' not in result
    assert len(calls)==4
    assert not list((runner.root/'teachers').glob('*_verified.npz'))


def test_partial_review_stops_without_writing_a_verified_demo(tmp_path,monkeypatch):
    runner,calls,incumbent=fixture_runner(tmp_path,monkeypatch,partial_review=True)
    with pytest.raises(ValueError,match='complete'):
        runner.teacher_search(incumbent)
    assert not list((runner.root/'teachers').glob('*_verified.npz'))


def test_partial_source_evaluation_cannot_become_a_completed_control_outcome(tmp_path,monkeypatch):
    runner,calls,incumbent=fixture_runner(tmp_path,monkeypatch,partial_source=True)
    with pytest.raises(ValueError,match='complete'):
        runner.teacher_search(incumbent)
    assert len(calls)==1


def test_consistent_original_positive_source_does_not_gate_generated_search(tmp_path,monkeypatch):
    runner,calls,incumbent=fixture_runner(tmp_path,monkeypatch,original=1,source=1)
    result=runner.teacher_search(incumbent)['r']
    assert len(calls)==4 and result['teacher_status']=='verified_solution'
    from jit_dvgc.generative_bridge.rewards import teacher_feedback
    reward,eligible,_,_=teacher_feedback(runner.panels['new_roots'],[],{'r':result})
    assert eligible[0] and reward[0]==pytest.approx(2.02)


def test_original_negative_source_recheck_success_is_unknown_but_generated_search_still_runs(tmp_path,monkeypatch):
    runner,calls,incumbent=fixture_runner(tmp_path,monkeypatch,original=0,source=1)
    result=runner.teacher_search(incumbent)['r']
    assert len(calls)==4 and not result['source_control_consistent'] and 'demo' not in result
    from jit_dvgc.generative_bridge.rewards import teacher_feedback
    reward,eligible,_,_=teacher_feedback(runner.panels['new_roots'],[],{'r':result})
    assert not eligible[0] and reward[0]==0.


def test_complete_generated_failures_remain_unsolved_for_consistent_positive_source(tmp_path,monkeypatch):
    runner,calls,incumbent=fixture_runner(tmp_path,monkeypatch,original=1,source=1,search=())
    result=runner.teacher_search(incumbent)['r']
    assert len(calls)==2 and result['teacher_status']=='searched_no_solution'
    from jit_dvgc.generative_bridge.rewards import teacher_feedback
    reward,eligible,_,_=teacher_feedback(runner.panels['new_roots'],[],{'r':result})
    assert eligible[0] and reward[0]==pytest.approx(-.08)


@pytest.mark.parametrize('source', [0,1])
def test_legacy_search_keeps_one_review_and_positive_source_short_circuit(tmp_path,monkeypatch,source):
    runner,calls,incumbent=fixture_runner(tmp_path,monkeypatch,original=source,source=source)
    runner.spec.pop('recovery_training_contract')
    result=runner.teacher_search(incumbent)['r']
    if source:
        assert len(calls)==1 and result['teacher_status']=='not_scheduled'
    else:
        assert len(calls)==3 and result['teacher_status']=='verified_solution'
        assert 'repeat_labels' not in result['verification']
