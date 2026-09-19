from pathlib import Path
import pytest
from jit_dvgc.jump_evidence_validation import write,read,file_sha
from jit_dvgc.analysis.dense_coverage import coverage_rows,compare_coverage


def panels():
    def point(cell,x):return {'root_cell':cell,'full_cell':cell,'coordinates':{'root_x_m':x,'root_z_m':.4,'root_vz_mps':-.5}}
    old=[point('a',2.5)];new=[point('a',2.5),point('b',2.6)]
    before={'resolution':{'fixed':True},'policy_names':['pi_0'],'masks':{'pi_0':[True],'union':[True]}}
    after={'resolution':{'fixed':True},'policy_names':['pi_0','pi_4'],'masks':{'pi_0':[True,False],'pi_4':[False,True],'union':[True,True]}}
    return old,before,new,after


def test_untested_policy_is_unknown_but_union_novelty_is_measurable(tmp_path):
    old,before,new,after=panels()
    rows=coverage_rows(old,before,new,after)
    assert rows[1]['old_panel_evaluated'] is False
    for suffix in ('old','novel','cumulative'):assert rows[1]['root_cell_'+suffix] is None
    assert rows[1]['root_cell_new_panel']==1
    assert rows[2]['root_cell_novel']==1 and rows[2]['root_cell_cumulative']==2
    compare_coverage(old,before,new,after,tmp_path)
    assert (tmp_path/'old_new_support.png').stat().st_size>500


def test_malformed_existing_mask_still_rejected():
    old,before,new,after=panels();before['masks']['pi_0']=[]
    with pytest.raises(ValueError):coverage_rows(old,before,new,after)


@pytest.mark.parametrize('missing_labels',[False,True])
def test_analysis_only_recovers_verified_data_without_rewriting_source(tmp_path,monkeypatch,missing_labels):
    from jit_dvgc import campaign_analysis as a
    from jit_dvgc import unified_policy_freeze as f
    source=tmp_path/'source';seed=tmp_path/'seed';out=tmp_path/'review'
    write(seed/'analysis_inputs.json',{})
    write(source/'seed_inputs.json',{str(seed/'analysis_inputs.json'):file_sha(seed/'analysis_inputs.json')})
    write(source/'summary.json',{'status':'engineering_error'})
    rd=source/'round_000';attempt=rd/'training_attempt_000';discovery=rd/'discovery'
    policy=attempt/'frozen.json';write(policy,{'test':True})
    write(attempt/'reservation.json',{'maximum_interactions':27200})
    write(attempt/'completion.json',{'policy':str(policy),'charged_interactions':25661,'artifacts':{str(policy):file_sha(policy)}})
    write(discovery/'cost_ledger.json',{'charged_interactions':73612})
    original={str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    def child(path,**kwargs):
        if path==discovery:
            assert kwargs['recover_figures_failure']
            if missing_labels:raise ValueError('incomplete campaign source labels')
        cell='a' if path==seed else 'b'
        return [{'root_cell':cell,'witnessed':True}],{}, {'proposer':'pi_4','members':[{'path':str(policy),'policy':{'name':'pi_4'}}]}
    monkeypatch.setattr(a,'read_child',child)
    monkeypatch.setattr(f,'load_frozen_unified_manifest',lambda p:{'policy':{'name':'pi_4','source_training_transitions':25600}})
    rendered=[];monkeypatch.setattr(a,'analyze',lambda *args,**kw:rendered.append(kw['figures_dir']))
    monkeypatch.setattr(a,'render_progress',lambda *args:None)
    monkeypatch.setattr(a,'bundle',lambda p:p/'results_to_send.zip')
    result=a.run(source,out)
    assert result['environment_interactions']==result['training_transitions']==0
    assert all(Path(p).read_bytes()==data for p,data in original.items())
    if missing_labels:assert result['status']=='engineering_error' and not rendered
    else:
        assert result['status']=='analysis_completed'
        assert result['rounds'][0]['novel_root_cells']==1
        assert result['source_charged_interactions']==99273
        assert rendered==[out/'round_000/figures']
