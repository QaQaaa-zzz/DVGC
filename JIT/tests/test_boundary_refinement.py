import copy
import csv
import pytest
from jit_dvgc import boundary_refinement as b
from jit_dvgc.evidence_integrity import canonical_sha256
from jit_dvgc.jump_evidence_validation import write, read


def source(tmp_path):
    previous=tmp_path/'previous'
    report={'status':'completed','scope':'TRAIN-informed landing/frontier discovery',
            'final_test_used':False,'frontier':{'candidate_status_counts':{'no_success_witness':1}}}
    report['report_sha256']=canonical_sha256(report)
    write(previous/'summary.json',report)
    (previous/'figures').mkdir()
    (previous/'figures/boundary_candidates.csv').write_text(
        'status,proposer,action,strength,sign,trajectory_id\nno_success_witness,pi_1,knee,0.2,1,t\n')
    return previous


def test_profile_locks_nine_legal_action_variants_and_source(tmp_path):
    profile=b.select_profile(source(tmp_path))
    b.validate_refinement(profile)
    assert len(profile['targets'])*len(profile['strengths'])*len(profile['signs'])*len(profile['action_names'])==9
    assert b.CEILING==1_847_700
    changed=copy.deepcopy(profile);changed['signs']=[-1]
    with pytest.raises(ValueError):b.validate_refinement(changed)
    path=next(iter(profile['source_files']))
    write(path,{})
    with pytest.raises(ValueError,match='source'):b.validate_refinement(profile)


def test_low_budget_does_not_launch_gpu_and_failure_is_packaged(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(b,'run_dense',lambda *a,**k:calls.append(k))
    result=b.run(tmp_path,tmp_path/'run',previous=source(tmp_path),budget=1)
    assert result['status']=='engineering_error' and not calls
    assert (tmp_path/'run/results_to_send.zip').exists()


def test_only_one_proposer_and_locked_profile_reaches_supervisor(tmp_path,monkeypatch):
    calls=[]
    def dense(*a,**k):
        calls.append(k)
        return {'status':'completed'}
    monkeypatch.setattr(b,'run_dense',dense)
    monkeypatch.setattr(b,'analyze',lambda p:{'status':'completed'})
    result=b.run(tmp_path,tmp_path/'run',previous=source(tmp_path))
    assert result['status']=='completed' and len(calls)==1
    assert calls[0]['proposer']=='pi_1' and calls[0]['profile']['serial_only']


def test_final_test_cannot_select_local_training_gap(tmp_path):
    previous=source(tmp_path)
    report=read(previous/'summary.json');report.pop('report_sha256');report['final_test_used']=True
    report['report_sha256']=canonical_sha256(report);write(previous/'summary.json',report)
    with pytest.raises(ValueError,match='TRAIN'):b.select_profile(previous)


def test_analysis_preserves_missing_witnesses_and_writes_review_plot(tmp_path,monkeypatch):
    from jit_dvgc import policy_comparison as pc, policy_comparison_runtime as runtime
    names=['pi_0','pi_1','pi_2','pi_3']
    members=[{'policy':{'name':n}} for n in names]
    plan={'members':members,'names':names,'label_seed':9843201}
    monkeypatch.setattr(pc,'verify_plan',lambda p:plan)
    point={'candidate_id':'c','state_sha256':'s','trajectory_id':'t0','phase':'downstream',
           'coordinates':{'root_x_m':4.2,'root_z_m':.5}}
    child=tmp_path/'pi_1'
    write(child/'points.json',[point])
    trails=[dict(trajectory_id=f't{i}',anchor_x_m=2.9,strength=.2,
        stop_reason='first_valid_landing',physical_failure=False,timeout=False,
        truncated=False,environment_interactions=1) for i in range(9)]
    write(child/'catalog.json',{'entries':[dict(point,perturbation={'strength':.2})],
                               'trajectory_receipts':trails,'environment_interactions':9})
    write(child/'analysis_inputs.json',{'projected':str(child/'points.json'),
        'catalog':str(child/'catalog.json'),'merged':{n:n for n in names}})
    write(child/'summary.json',{'charged_interactions':13})
    monkeypatch.setattr(runtime,'complete_output',lambda *a:({},[dict(point,label=0)]))
    result=b.analyze(tmp_path)
    assert result['candidate_status_counts']=={'no_success_witness':1}
    assert result['training_admission_authorized'] is False
    assert (tmp_path/'figures/boundary_refinement.png').exists()
    with (tmp_path/'figures/training_review_candidates.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    assert rows[0]['training_admitted']=='False'
