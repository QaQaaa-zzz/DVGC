import pytest
from jit_dvgc.generative_bridge.campaign import nominal_evidence, acceptance_v12


def test_nominal_gate_requires_every_completed_repeat():
    assert nominal_evidence({'phase':'completed'},[{'prefix_label':1}]*4)['nominal_success']
    assert not nominal_evidence({'phase':'completed'},[{'prefix_label':1},{'prefix_label':0}])['nominal_success']
    with pytest.raises(ValueError,match='not completed'):
        nominal_evidence({'phase':'waiting'},[])


def test_acceptance_counts_integer_loss_and_rejects_unknown():
    old={'core':[1,1,0], 'protected':[1]*32,'new_roots':[0,0],'nominal':[1]*4}
    new={'core':[1,1,0], 'protected':[1]*32,'new_roots':[1,0],'nominal':[1]*4}
    result=acceptance_v12(old,new,{'actor_sha256':'s','normalizer_sha256':'n'})
    assert result['adopted'] and result['protected_allowed_losses']==0
    new['protected'][0]=0
    assert not acceptance_v12(old,new,{})['adopted']
    new['protected'][0]=None
    assert not acceptance_v12(old,new,{})['adopted']


def test_teacher_layout_conflict_cannot_count_as_new_actor_gain():
    from jit_dvgc.generative_bridge.campaign import source_root_labels
    rows=[{'root_id':'r'}]
    assert source_root_labels(rows,{'r':{'source_recheck_label':1,'reason':'same_layout_source_recheck'}})==[None]
    assert source_root_labels(rows,{'r':{'source_recheck_label':1,'reason':'source_recheck_succeeded'}})==[1]


def test_tensorboard_export_keeps_supervised_step_unit(tmp_path):
    import importlib.util,json
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('export_tb',Path(__file__).parents[1]/'cli/export_tensorboard.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    path=tmp_path/'g.jsonl';path.write_text(json.dumps({'update':3,'metrics':{'generator/noise_mse':.2}})+'\n')
    assert module.metric_rows({'path':str(path),'step':'update'})==[{'update':3,'generator/noise_mse':.2}]


def test_campaign_rejects_ambiguous_process_restart(tmp_path):
    from jit_dvgc.generative_bridge.campaign import CampaignRunner
    (tmp_path/'started.json').write_text('{}')
    with pytest.raises(ValueError,match='already started'):
        CampaignRunner({'output':str(tmp_path)})
