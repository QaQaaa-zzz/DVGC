import copy
from pathlib import Path
import pytest
from jit_dvgc.frontier_exploration import allocate,validate_profile
from jit_dvgc.evidence_integrity import canonical_sha256
from jit_dvgc.jump_evidence_validation import read,write
from jit_dvgc.analysis.frontier_evidence import classify_labels,summarize_frontier


def source_report():
    r={'status':'completed','scope':'TRAIN development','final_test_used':False,
       'metrics':[{'proposer':n,'charged_interactions':cost,'novel_vs_previous_train_union':gain}
                  for n,cost,gain in [('pi_0',21668,55),('pi_1',29752,307),('pi_2',27675,270),('pi_3',32137,272)]]}
    r['report_sha256']=canonical_sha256(r)
    return r


def test_train_gain_locks_allocation_and_ceilings():
    r=source_report();a=allocate(r)
    assert a['priority_proposers']==['pi_1','pi_2'] and a['total_trajectories']==48
    total=0
    for p in a['profiles'].values():
        validate_profile(p)
        total+=p['acquisition_ceiling']+p['max_trajectories']*p['max_candidates']*4*400
    assert total==9_854_400
    assert all(p['serial_only'] and p['sampling_max_x_m']==8 for p in a['profiles'].values())
    r['metrics'][0]['novel_vs_previous_train_union']=999
    with pytest.raises(ValueError):allocate(r)
    p=copy.deepcopy(a['profiles']['pi_0']);p['label_seed']=123
    with pytest.raises(ValueError):validate_profile(p)


def test_test_results_cannot_guide_allocation():
    r=source_report();r.pop('report_sha256');r['final_test_used']=True
    r['report_sha256']=canonical_sha256(r)
    with pytest.raises(ValueError,match='TEST'):allocate(r)


def test_missing_labels_never_become_boundary_failures():
    assert classify_labels([0,0,0,0])=='no_success_witness'
    assert classify_labels([1,0,0,0])=='policy_disagreement'
    with pytest.raises(ValueError):classify_labels([0,0,0,None])


def test_scope_extension_is_separate_from_common_domain_gain(tmp_path):
    names=['pi_0','pi_1','pi_2','pi_3']
    panels={};labels={}
    for name in names:
        child=tmp_path/name
        write(child/'center.json',{'effective_centerline_max_x_m':3.8})
        write(child/'plan.json',{'centerline':str(child/'center.json')})
        points=[{'candidate_id':str(i),'state_sha256':str(i),'trajectory_id':'t','trajectory_step':i+1,
                 'phase':'downstream','coordinates':{'root_x_m':x,'root_z_m':.4,'root_vz_mps':-1.},'root_cell':cell}
                for i,(x,cell) in enumerate([(3.5,'old'),(4.1,'extended'),(3.7,'failed')])]
        entries=[{'candidate_id':p['candidate_id'],'state_sha256':p['state_sha256'],
                  'perturbation':{'action_name':'hip','sign':1,'strength':.2},
                  'source_bank':'bank','snapshot':p['candidate_id']} for p in points]
        receipt={'trajectory_id':'t','environment_interactions':10,'stop_reason':'candidate_cap' if name=='pi_0' else 'first_valid_landing',
                 'truncated':name=='pi_0','direction':{}}
        write(child/'catalog.json',{'entries':entries,'attempted_candidate_count':1,
                                   'environment_interactions':10,'trajectory_receipts':[receipt]})
        write(child/'analysis_inputs.json',{'catalog':str(child/'catalog.json')})
        panels[name]=points
        labels[name]={n:[{'label':1},{'label':int(n=='pi_1')},{'label':0}] for n in names}
    dest=tmp_path/'figures';dest.mkdir()
    result=summarize_frontier(tmp_path,panels,labels,{'old'},dest)
    assert result['old_corridor_novel_root_cells']==0
    assert result['extended_corridor_only_novel_root_cells']==1
    assert result['total_novel_root_cells']==1
    assert result['candidate_status_counts']['no_success_witness']==4
    assert result['truncated_trajectories']==1 and not result['all_attempts_reached_terminal_outcome']
    assert (dest/'boundary_states.pdf').exists()


def test_frontier_supervisor_locks_allocation_and_rejects_low_budget_before_gpu(tmp_path,monkeypatch):
    import jit_dvgc.discovery_comparison as d
    prior=tmp_path/'prior';write(prior/'summary.json',source_report())
    monkeypatch.setattr(d,'lock_previous',lambda p:{})
    calls=[]
    def dense(repo,output,**kw):
        calls.append(kw)
        return {'status':'completed'}
    monkeypatch.setattr(d,'run_dense',dense)
    monkeypatch.setattr(d,'analyze',lambda *args:{'status':'completed'})
    result=d.run(tmp_path,tmp_path/'run',previous_discovery=prior,budget_per_proposer=4_000_000)
    assert result['status']=='completed' and len(calls)==4
    assert calls[1]['profile']['strengths']==[.1,.2]
    assert calls[0]['profile']['strengths']==[.15]
    assert read(tmp_path/'run/allocation.json')['total_trajectories']==48
    calls.clear()
    failed=d.run(tmp_path,tmp_path/'underbudget',previous_discovery=prior,budget_per_proposer=1_000_000)
    assert failed['status']=='engineering_error' and not calls
    assert (tmp_path/'underbudget/results_to_send.zip').exists()
