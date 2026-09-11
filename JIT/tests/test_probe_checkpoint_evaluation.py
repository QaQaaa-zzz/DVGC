"""Bounded checkpoint scheduling uses the existing single-run PPO callbacks."""
from types import SimpleNamespace
import pytest


def test_explicit_checkpoints_reserve_every_panel_and_forbid_extension():
    from jit_dvgc.iterative_probe_training import checkpoint_evaluation_plan
    plan = checkpoint_evaluation_plan(128000, [32000, 64000, 128000])
    assert plan['checkpoint_transitions'] == [0, 32000, 64000, 128000]
    assert plan['train_panel_transitions'] == [32000, 64000, 128000]
    assert plan['maximum_panel_interactions'] == 4800
    assert plan['maximum_total_interactions'] == 132800
    assert plan['maximum_training_transitions'] == 128000
    assert plan['automatic_extension'] is False
    assert plan['optimizer_continuation'] == 'same_live_trainer_only'


@pytest.mark.parametrize('steps,schedule', [(128000,[32001,128000]), (128000,[64000,32000,128000]),
    (128000,[32000,32000,128000]), (128000,[32000]), (25600,[32000,64000,128000]), (128000,[])])
def test_invalid_or_unbounded_schedule_rejected(steps,schedule):
    from jit_dvgc.iterative_probe_training import checkpoint_evaluation_plan
    with pytest.raises(ValueError): checkpoint_evaluation_plan(steps,schedule)


def test_legacy_budget_and_generic_small_aligned_budget():
    from jit_dvgc.iterative_probe_training import checkpoint_evaluation_plan
    legacy=checkpoint_evaluation_plan(128000)
    assert legacy['checkpoint_transitions']==[0,128000]
    assert legacy['maximum_total_interactions']==129600
    short=checkpoint_evaluation_plan(6400,[3200,6400],samples_per_phase=1,horizon=20)
    assert short['maximum_total_interactions']==6480


def test_fixed_panel_identity_detects_selected_context_and_schedule_drift():
    from jit_dvgc.iterative_probe_training import checkpoint_evaluation_plan, fixed_train_panel_identity
    support={'support_sha256':'a'*64,'entries':[
        {'phase':phase,'key':str(i),'snapshot':f'{phase}-{i}.npz','witnessed':True}
        for phase in ('upstream','downstream') for i in range(3)]}
    plan=checkpoint_evaluation_plan(6400,[3200,6400])
    first=fixed_train_panel_identity(support,plan)
    assert len(first['selected_entries'])==4
    assert first['role']=='train' and first['final_test_used'] is False
    changed={**support,'support_sha256':'b'*64}
    assert fixed_train_panel_identity(changed,plan)['panel_sha256']!=first['panel_sha256']
    assert fixed_train_panel_identity(support,{**plan,'horizon':200})['panel_sha256']!=first['panel_sha256']


def test_runtime_panel_uses_declared_horizon_identity_and_reserves_failure(tmp_path, monkeypatch):
    import jit_dvgc.unified_formal as formal
    from jit_dvgc.iterative_probe_training import checkpoint_evaluation_plan
    plan=checkpoint_evaluation_plan(6400,[3200,6400],samples_per_phase=1,horizon=20)
    artifact=SimpleNamespace(manifest={'manifest_sha256':'a'*64},entries=[
        {'phase':phase,'key':'0','snapshot':phase+'.npz'} for phase in ('upstream','downstream')])
    from jit_dvgc.iterative_probe_training import fixed_train_panel_identity
    identity=fixed_train_panel_identity({'support_sha256':'a'*64,'entries':artifact.entries},plan)
    config=SimpleNamespace(raw={'checkpoint_evaluation':plan,'fixed_train_panel':identity},
        formal=SimpleNamespace(samples_per_phase=1),ppo=SimpleNamespace(episode_horizon=400))
    seen=[]
    def fail(env,policy,**kwargs):
        seen.append(kwargs)
        assert (tmp_path/'train_panels/transition_3200/reservation.json').exists()
        raise RuntimeError('device failed during panel')
    monkeypatch.setattr(formal,'rollout_fixed_tube_panel',fail)
    with pytest.raises(RuntimeError,match='device failed'):
        formal._evaluate_train_panel(None,artifact,tmp_path,config,3200,lambda *a,**k:None,None)
    assert seen[0]['horizon']==20
    from jit_dvgc.iterative_probe_training import charged_train_panel_interactions
    assert charged_train_panel_interactions(tmp_path)==40


def test_panel_success_charges_actual_interactions(tmp_path, monkeypatch):
    import json
    import jit_dvgc.unified_formal as formal
    from jit_dvgc.iterative_probe_training import checkpoint_evaluation_plan, charged_train_panel_interactions
    plan=checkpoint_evaluation_plan(6400,[3200,6400],samples_per_phase=1,horizon=20)
    artifact=SimpleNamespace(manifest={'manifest_sha256':'a'*64},entries=[
        {'phase':phase,'key':'0','snapshot':phase+'.npz'} for phase in ('upstream','downstream')])
    from jit_dvgc.iterative_probe_training import fixed_train_panel_identity
    identity=fixed_train_panel_identity({'support_sha256':'a'*64,'entries':artifact.entries},plan)
    config=SimpleNamespace(raw={'checkpoint_evaluation':plan,'fixed_train_panel':identity},
        formal=SimpleNamespace(samples_per_phase=1),ppo=SimpleNamespace(episode_horizon=400))
    monkeypatch.setattr(formal,'rollout_fixed_tube_panel',lambda *a,**k:({'environment_interactions':7},()))
    monkeypatch.setattr(formal,'_tube_points',lambda a:[])
    monkeypatch.setattr(formal,'plot_xz_visitation',lambda *a:None)
    result=formal._evaluate_train_panel(None,artifact,tmp_path,config,3200,lambda *a,**k:None,None)
    assert result.environment_transitions==7
    assert charged_train_panel_interactions(tmp_path)==7
    report=json.loads((tmp_path/'train_panels/transition_3200/report.json').read_text())
    assert report['fixed_train_panel']==identity
    artifact.entries[0]['key']='changed'
    with pytest.raises(ValueError,match='runtime TRAIN panel identity'):
        formal._evaluate_train_panel(None,artifact,tmp_path,config,6400,lambda *a,**k:None,None)


def test_generated_config_locks_panel_and_drives_existing_callbacks(tmp_path, jit_root):
    import json
    from jit_dvgc.evidence_integrity import canonical_sha256
    from jit_dvgc.iterative_probe_training import make_config, load_config
    from jit_dvgc.unified_formal import UnifiedFormalController, build_unified_formal_trainer_kwargs
    from jit_dvgc.checkpoint import CheckpointIdentity
    from jit_dvgc.formal_training import PanelResult
    support={'schema':'jit_iterative_witnessed_support_v1','role':'train','final_test_used':False,
        'inputs':{},'entries':[{'phase':phase,'key':str(i),'snapshot':f'{phase}-{i}.npz','witnessed':True}
            for phase in ('upstream','downstream') for i in range(2)]}
    support['support_sha256']=canonical_sha256(support)
    support_path=tmp_path/'support.json';support_path.write_text(json.dumps(support))
    init=tmp_path/'frozen.json';init.write_text('{}')
    path=tmp_path/'config.json'
    make_config(support_path,init,jit_root/'configs/pi_unified_formal.json',path,'fixture',1,6400,123,
        checkpoints=[3200,6400],panel_samples_per_phase=1,panel_horizon=20)
    config=load_config(path)
    assert config.formal.checkpoint_transitions==(0,3200,6400)
    saved=[];evaluated=[]
    def panel(step,make_policy,params):
        evaluated.append((step,params))
        return PanelResult(step,5,{})
    controller=UnifiedFormalController(config=config,run_dir=tmp_path/'training',
        identity=CheckpointIdentity('cfg','xml',(),(),()),evaluate_train_panel=panel,
        checkpoint_saver=lambda path,payload:saved.append(payload.training_transitions))
    kwargs=build_unified_formal_trainer_kwargs(config,object(),controller)
    assert kwargs['num_timesteps']==6400 and kwargs['num_evals']==3
    assert kwargs['run_evals'] is False
    params=(object(),object(),object())
    for step in (0,3200,6400):controller.on_policy_params(step,object(),params)
    assert saved==[0,3200,6400]
    assert [step for step,_ in evaluated]==[3200,6400]
    assert all(p is params for _,p in evaluated)
    assert controller.train_panel_interactions==10
    raw=json.loads(path.read_text());raw['fixed_train_panel']['horizon']=21
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='panel identity'):load_config(path)
    # Historical make_config call keeps the legacy final-only schema and defaults.
    raw=make_config(support_path,init,jit_root/'configs/pi_unified_formal.json',path,'legacy',1,6400,123)
    assert 'checkpoint_evaluation' not in raw and 'fixed_train_panel' not in raw
    assert load_config(path).formal.checkpoint_transitions==(0,6400)


@pytest.mark.parametrize('mutation', ['identity','count','negative_reserve','bool_reserve','missing_report'])
def test_completed_panel_receipt_must_join_reservation_and_report(tmp_path, mutation):
    from jit_dvgc.iterative_probe_training import charged_train_panel_interactions
    from jit_dvgc.jump_evidence_validation import write
    panel=tmp_path/'train_panels/transition_3200'
    reserve={'maximum_interactions':40,'panel_sha256':'a'*64,'training_checkpoint_transition':3200}
    receipt={'charged_interactions':7,'panel_sha256':'a'*64}
    report={'environment_interactions':7,'training_checkpoint_transition':3200,
            'fixed_train_panel':{'panel_sha256':'a'*64}}
    if mutation=='identity':receipt['panel_sha256']='b'*64
    if mutation=='count':receipt['charged_interactions']=6
    if mutation=='negative_reserve':reserve['maximum_interactions']=-1
    if mutation=='bool_reserve':reserve['maximum_interactions']=True; receipt['charged_interactions']=1
    write(panel/'reservation.json',reserve)
    write(panel/'completion.json',receipt)
    if mutation!='missing_report':write(panel/'report.json',report)
    with pytest.raises(ValueError,match='panel'):
        charged_train_panel_interactions(tmp_path)
