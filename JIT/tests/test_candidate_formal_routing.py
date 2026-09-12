"""Candidate runtime dispatch is checked with CPU mocks and no simulator steps."""
from types import SimpleNamespace
import pytest


@pytest.mark.parametrize('schema',['jit_iterative_probe_training_v1','jit_iterative_candidate_training_v1'])
def test_all_formal_and_freeze_loaders_route_iterative_schema(tmp_path,monkeypatch,schema):
    from jit_dvgc.jump_evidence_validation import write
    import jit_dvgc.unified_formal as formal
    import jit_dvgc.iterative_probe_training as probe
    from jit_dvgc.unified_policy_freeze import _load_policy_formal_config
    path=tmp_path/'config.json'
    write(path,{'schema':schema,'initialization':{'actor':'warm_start_frozen_unified'}})
    sentinel=object();restored=object()
    monkeypatch.setattr(probe,'load_config',lambda p:sentinel)
    monkeypatch.setattr(probe,'restore_params',lambda p:restored)
    for loader in (formal.load_unified_formal_config,formal.load_unified_actor_warm_start_config,
                   formal.load_unified_policy_formal_config,_load_policy_formal_config):
        assert loader(path) is sentinel
    assert formal.load_frozen_actor_restore_params(path) is restored
    monkeypatch.setattr(probe,'build_environment',lambda c:('candidate artifact','candidate env'))
    assert formal._build_unified_formal_environment(SimpleNamespace(schema=schema)) == ('candidate artifact','candidate env')


def test_legacy_panel_reuses_existing_environment():
    from jit_dvgc.unified_formal import _build_train_panel_environment
    artifact,env=object(),object()
    assert _build_train_panel_environment(SimpleNamespace(schema='jit_iterative_probe_training_v1'),artifact,env)==(artifact,env)


def test_candidate_run_callback_receives_only_witnessed_panel(jit_root,tmp_path,monkeypatch):
    from dataclasses import replace
    import jit_dvgc.unified_formal as formal
    import jit_dvgc.iterative_probe_training as probe
    config=replace(formal.load_unified_formal_config(jit_root/'configs/pi_unified_formal.json'),
                   schema='jit_iterative_candidate_training_v1')
    config.raw['initialization']=formal.actor_only_warm_start_initialization('source.json')
    monkeypatch.setattr(formal,'load_unified_policy_formal_config',lambda path:config)
    training_artifact=object()
    training_env=SimpleNamespace(_bundle=SimpleNamespace(xml_sha256='a'*64),
        resolved_config=SimpleNamespace(model={'reference_sha256':'b'*64}))
    panel_artifact,panel_env=object(),object()
    calls=[]
    def build(c,*,panel=False):
        calls.append(panel)
        return (panel_artifact,panel_env) if panel else (training_artifact,training_env)
    monkeypatch.setattr(probe,'build_environment',build)
    writes={}
    monkeypatch.setattr(formal,'_write_json',lambda path,value:writes.update({path.name:value}))
    for name in ('predeclare_run','mark_run_running','close_run'):
        monkeypatch.setattr(formal,name,lambda *a,**k:None)
    monkeypatch.setattr(formal,'checkpoint_identity',lambda *a:object())
    monkeypatch.setattr(formal,'load_frozen_actor_restore_params',lambda p:(object(),object(),object()))
    callbacks=[]
    controller=SimpleNamespace(on_progress=lambda *a:None,on_policy_params=lambda *a:None,
                               segment_training_transitions=0,train_panel_interactions=0)
    monkeypatch.setattr(formal,'UnifiedFormalController',lambda **kwargs:callbacks.append(kwargs['evaluate_train_panel']) or controller)
    evaluations=[]
    monkeypatch.setattr(formal,'_evaluate_train_panel',lambda *args:evaluations.append(args))
    def trainer(**kwargs):
        callbacks[0](32000,object(),object())
        raise RuntimeError('CPU mock stop before simulation')
    with pytest.raises(RuntimeError,match='CPU mock stop'):
        formal.run_unified_formal(tmp_path/'config.json','fixture',run_root=tmp_path,
                                  backend_name=lambda:'gpu',trainer=trainer)
    assert calls==[False,True]
    assert evaluations[0][0] is panel_env and evaluations[0][1] is panel_artifact
    provenance=writes['formal_provenance.json']
    assert provenance['tube_rsi_smoke_report_sha256'] is None
    assert provenance['iterative_training_support_sha256']==config.soft_tube_manifest_sha256


def test_cli_routes_candidate_schema_without_launch(tmp_path,jit_root,monkeypatch,capsys):
    import importlib.util
    import sys
    from jit_dvgc.jump_evidence_validation import write
    path=tmp_path/'candidate.json';write(path,{'schema':'jit_iterative_candidate_training_v1'})
    spec=importlib.util.spec_from_file_location('candidate_cli_fixture',jit_root/'cli/train_unified.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    calls=[]
    monkeypatch.setattr(module,'run_unified_formal',lambda *args:calls.append(args) or {'mock':True})
    monkeypatch.setattr(sys,'argv',['train_unified.py','--config',str(path),'--run-id','fixture'])
    assert module.main()==0
    assert calls==[(path,'fixture')]
    assert 'true' in capsys.readouterr().out


def test_canonical_cli_preflight_routes_candidate_support(tmp_path,monkeypatch):
    import json
    import jit_dvgc.training.formal as formal
    support=tmp_path/'support.json'
    support.write_text(json.dumps({'entries':[{'snapshot':'fixture'}],'selection':'pending quota'}))
    config=SimpleNamespace(schema='jit_iterative_candidate_training_v1',soft_tube_path=str(support),
        soft_tube_manifest_sha256='locked',raw={'jump_start_probability':.2})
    monkeypatch.setattr(formal,'load_unified_policy_formal_config',lambda _:config)
    monkeypatch.setattr(formal,'_tube_points',lambda artifact: tuple(artifact.entries))
    monkeypatch.setattr(formal,'load_soft_tube',lambda _:pytest.fail('JSON support is not a legacy directory'))
    result=formal.preflight_unified_formal_tube(tmp_path/'config.json')
    assert result['entry_count']==1
    assert result['environment_interactions']==0
