from pathlib import Path
import copy
import pytest
from jit_dvgc.discovery_comparison import coverage_events,curve,at_budget,NAMES
from jit_dvgc.jump_evidence_validation import write,read
from jit_dvgc.evidence_integrity import canonical_sha256


def panel():
    points=[{'candidate_id':str(i),'state_sha256':str(i)*64,'phase':'upstream',
             'root_cell':f'r{i}','full_cell':f'f{i}','trajectory_id':'t','coordinates':{'root_x_m':2.8+i*.05,'root_z_m':.4,'root_vz_mps':1.}}
            for i in range(2)]
    labels={n:[{**{k:p[k] for k in ('candidate_id','state_sha256','phase')},'label':int(i==0),
                'success_criterion':'first_valid_landing','environment_interactions':2}
               for i,p in enumerate(points)] for n in NAMES}
    return points,labels


def test_cost_curve_counts_failures_and_never_credits_partial_evaluation():
    points,labels=panel()
    overhead,events=coverage_events(points,labels,charged=26)
    assert overhead==10
    rows=curve(events,overhead,{'r0'})
    assert rows[-1]=={'interactions':26,'root_cells':1,'full_cells':1,'novel_root_cells':0}
    assert at_budget(rows,17)['root_cells']==0
    assert at_budget(rows,18)['root_cells']==1
    assert at_budget(rows,5)['root_cells']==0
    labels['pi_3'][0]['state_sha256']='wrong'
    with pytest.raises(ValueError,match='identity'):coverage_events(points,labels,26)


def test_underaccounted_and_incomplete_panels_refuse():
    points,labels=panel()
    with pytest.raises(ValueError,match='undercounts'):coverage_events(points,labels,1)
    labels['pi_2'].pop()
    with pytest.raises(ValueError,match='incomplete'):coverage_events(points,labels,26)


def test_discovery_analysis_preserves_proposer_identity_and_exports(tmp_path,monkeypatch):
    import jit_dvgc.discovery_comparison as d
    import jit_dvgc.policy_comparison_runtime as runtime
    points,base_labels=panel()
    old=tmp_path/'old'
    write(old/'plan.json',{'panels':{'train':{'projected':str(old/'points.json')}}})
    write(old/'points.json',[])
    write(old/'figures/train/summary.json',{'masks':{'union':[]}})
    members=[{'path':n,'file_sha256':n,'policy':{'name':n}} for n in NAMES]
    for n in NAMES:
        child=tmp_path/n
        write(child/'plan.json',{'proposer':n,'baseline':str(old),'members':members})
        write(child/'summary.json',{'status':'completed','charged_interactions':26})
        # Distinct reached cells by proposer; suffix policies share most support.
        ps=copy.deepcopy(points)
        for p in ps:
            p['root_cell']=n+p['root_cell'];p['full_cell']=n+p['full_cell']
        write(child/'points.json',ps)
        write(child/'analysis_inputs.json',{'projected':str(child/'points.json'),'catalog':n,
                                          'merged':{e:e for e in NAMES}})
        report={'resolution':{'x':.1}};report['report_sha256']=canonical_sha256(report)
        write(child/'figures/summary.json',report)
    monkeypatch.setattr(d,'verify_plan',read)
    def complete(path,catalog,acquisition,member,*args):
        assert acquisition['policy']['name']==catalog
        return {},base_labels[member['policy']['name']]
    monkeypatch.setattr(runtime,'complete_output',complete)
    report=d.analyze(tmp_path)
    assert report['pooled_root_cells']==4 and report['charged_interactions']==104
    assert all(r['unique_proposer_root_cells']==1 for r in report['metrics'])
    # All setup costs upfront means pooled schedule has no witness at B=26.
    assert report['matched_budget'][-1]['root_cells']==0
    assert len(list((tmp_path/'figures').glob('*.png')))==3
    assert (tmp_path/'figures/matched_budget.csv').exists()


def test_supervisor_packages_child_failure_and_routes_all_proposers(tmp_path,monkeypatch):
    import jit_dvgc.discovery_comparison as d
    calls=[]
    monkeypatch.setattr(d,'lock_previous',lambda p:{})
    def dense(repo,output,**kw):
        calls.append(kw['proposer'])
        return {'status':'engineering_error' if kw['proposer']=='pi_2' else 'completed'}
    monkeypatch.setattr(d,'run_dense',dense)
    result=d.run(tmp_path,tmp_path/'run')
    assert result['status']=='engineering_error'
    assert calls==NAMES[:3]
    assert (tmp_path/'run/results_to_send.zip').exists()


def test_non_pi0_worker_uses_own_proposer_but_pi0_benchmark(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import jax
    import jit_dvgc.dense_tube_runtime as runtime
    import jit_dvgc.policy_family_landing as family
    plan={'members':[{'path':n,'policy':{'name':n}} for n in NAMES],
          'proposer':'pi_2','benchmark_catalog':str(tmp_path/'bench.json')}
    write(tmp_path/'bench.json',{})
    monkeypatch.setattr(runtime,'verify_plan',lambda p:plan)
    monkeypatch.setattr(jax,'default_backend',lambda:'gpu')
    monkeypatch.setattr(jax,'local_devices',lambda:[1])
    observed=[]
    def shard(**kwargs):
        observed.append(str(kwargs['acquisition_frozen_policy']))
        write(kwargs['output_dir']/'labels.json',[])
        return {'environment_interactions':0,'elapsed_seconds':1}
    monkeypatch.setattr(family,'run_policy_family_evaluator_shard',shard)
    for kind in ('benchmark','label'):
        args=SimpleNamespace(output_dir=tmp_path,destination=tmp_path/kind,worker=kind,
             policy='pi_3',catalog=tmp_path/'new.json',shard_index=0,shard_count=1,backend='serial')
        assert runtime.worker(args)==0
    assert observed==['pi_0','pi_2']


def test_empty_arrivals_still_validate_actor_identity():
    from jit_dvgc.unified_continuation_labels import validate_unified_boundary_catalog
    policy={'iteration':2,'name':'pi_2','actor_sha256':'a','payload_sha256':'b'}
    catalog={'schema':'jit_unified_boundary_catalog_v1','status':'completed',
             'artifact_role':'unlabeled_policy_conditioned_frontier_candidates',
             'split':'train','training_transitions':0,'expert_switching_used':False,
             'test_data_used':False,'validation_data_used':False,'final_evaluation_data_used':False,
             'frozen_unified_manifest_sha256':'f','iteration':2,'policy_name':'pi_2',
             'policy_actor_sha256':'a','policy_payload_sha256':'b','entries':[],'candidate_count':0,
             'claim_boundary':{'unlabeled_acquisition_only':True,'tube_expansion_claim':False,
                               'jce_jel_claim':False,'certified_safe_set_claim':False}}
    assert validate_unified_boundary_catalog(catalog,policy_record=policy,frozen_manifest_sha256='f',allow_empty=True)==()
    with pytest.raises(ValueError,match='no candidates'):
        validate_unified_boundary_catalog(catalog,policy_record=policy,frozen_manifest_sha256='f')
    catalog['policy_actor_sha256']='wrong'
    with pytest.raises(ValueError,match='actor'):
        validate_unified_boundary_catalog(catalog,policy_record=policy,frozen_manifest_sha256='f',allow_empty=True)
