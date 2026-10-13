"""Calibration selection is factual finite DEV evidence, not a universal budget."""
import pytest
from jit_dvgc import retention_guard_calibration as calibration


def point(step,eligible=True,mse=.01,p99=.1):
    return dict(attempt=step,finite=True,capabilities={'eligible':eligible},
        guard={'groups':{'old':{'mse':[mse]*4,'p99':[p99]*4}}})


def test_frozen_budgets_are_max_actual_drift_of_qualified_nonzero_candidates():
    rows=[point(0,mse=0,p99=0),point(5,mse=.02,p99=.2),point(10,False,mse=1,p99=1),point(20,mse=.01,p99=.3)]
    result=calibration.freeze_thresholds(rows,['old'])
    assert result['mse_thresholds']=={'old':[.01]*4}
    assert result['p99_thresholds']=={'old':[.3]*4}
    assert result['qualified_attempts']==[20]
    assert result['threshold_rule']==calibration.SELECTION_RULE


def test_no_qualifying_nonzero_candidate_blocks_instead_of_inventing_budget():
    with pytest.raises(ValueError,match='no qualifying'):
        calibration.freeze_thresholds([point(0),point(5,True),point(10,True),point(20,False)],['old'])


@pytest.mark.parametrize('change',['nonfinite','missing_group','wrong_schedule'])
def test_invalid_candidate_evidence_fails_closed(change):
    rows=[point(0),point(5),point(10),point(20)]
    if change=='nonfinite':rows[1]['guard']['groups']['old']['p99'][0]=float('nan')
    elif change=='missing_group':rows[1]['guard']['groups']={}
    else:rows[1]['attempt']=7
    with pytest.raises((ValueError,FloatingPointError)):calibration.freeze_thresholds(rows,['old'])


def reservation_fixture(tmp_path,monkeypatch):
    import json
    from jit_dvgc.generative_bridge.contracts import file_sha
    from jit_dvgc import retention_consolidation as stage
    spec=dict(horizon=400,num_envs=64,proposer='parent',bank='bank')
    cases=[dict(case_id=str(i),role='TRAIN_RETENTION' if i==0 else 'DEV_RETENTION',mode='collect',capacity=64,horizon=400,repeat_seeds=[1,1,1],spec=spec) for i in range(4)]
    cases.append(dict(case_id='solver',role='DEV_RETENTION',mode='evaluate',capacity=4,horizon=400,repeat_seeds=[2,2,2],spec={'candidates':'roots'}))
    panel=tmp_path/'panel.json';panel.write_text(json.dumps({'cases':cases,'split_records':[]}))
    plan=tmp_path/'plan.json';plan.write_text(json.dumps({'config':{'evaluation_panel':{'path':str(panel),'sha256':file_sha(panel)}}}))
    monkeypatch.setattr(stage,'audit',lambda p:dict(unresolved=['guard_config.calibration','guard_config.mse_thresholds']))
    grant=dict(schema=calibration.AUTH_SCHEMA,stage_plan_sha256=file_sha(plan),execute=True,user_authorization_record='explicit_fixture_only',
        selection_rule=calibration.SELECTION_RULE,checkpoint_steps=calibration.STEPS,physical_checkpoint_steps=calibration.PHYSICAL_STEPS,
        calibration_attempts=20,max_supervised_attempts=22,output=str(tmp_path/'output'),environment_transitions=153600,wall_seconds=7200,tensorboard_port=6031)
    auth=tmp_path/'authorization.json';auth.write_text(json.dumps(grant))
    return plan,auth,grant


def test_calibration_reserves_shared64_job_only_physical0_and20(tmp_path,monkeypatch):
    plan,auth,grant=reservation_fixture(tmp_path,monkeypatch)
    _,parsed,panel=calibration.require_reservation(plan,auth)
    assert parsed['environment_transitions']==153600
    assert len(panel['cases'])==4
    assert {c['mode'] for c in panel['cases']}=={'collect'}
    assert panel['cases'][0]['role']=='TRAIN_RETENTION'


@pytest.mark.parametrize('change',['too_low_physics','no_authorization','other_blocker','changed_plan'])
def test_calibration_fails_closed_before_backend_or_optimizer(tmp_path,monkeypatch,change):
    import json
    from jit_dvgc import retention_consolidation as stage
    plan,auth,grant=reservation_fixture(tmp_path,monkeypatch)
    if change=='too_low_physics':grant['environment_transitions']=100
    elif change=='no_authorization':grant['user_authorization_record']=''
    elif change=='other_blocker':monkeypatch.setattr(stage,'audit',lambda p:dict(unresolved=['data_roles.TRAIN_GUARD']))
    else:plan.write_text('{}')
    auth.write_text(json.dumps(grant))
    with pytest.raises(ValueError):calibration.require_reservation(plan,auth)
    assert not (tmp_path/'output').exists()


def test_micro_uses_real_toy_proposal_and_full_checkpoint_roundtrip(tmp_path):
    from .test_retention_guard import setup_warm
    import json,pickle
    net,initializer,reference,demo,keep,guard=setup_warm(tmp_path)
    fit=dict(observations=guard['bank']['observations'],weights=guard['bank']['weights']/guard['bank']['weights'].sum())
    result=calibration._gpu_micro(net,initializer,demo,fit,guard['bank'],guard['config'],4,tmp_path)
    assert result['phase']=='passed' and result['physical_transitions']==0
    assert result['optimizer_proposals']==1 and result['guard_checks']==2
    assert result['accepted']['accepted'] and not result['rejected']['accepted']
    saved=pickle.load((tmp_path/'gpu_micro_learner.pkl').open('rb'))
    assert saved['attempts']==saved['accepted_updates']==1
    assert int(saved['optimizer_state'][1][0].count)==1


def test_calibration_watcher_defers_completion_to_overall_lineage(tmp_path):
    overall=tmp_path/'overall_status.json';overall.write_text('{"phase":"running"}')
    root=tmp_path/'calibration';root.mkdir()
    pointer=calibration.monitoring_pointer(root,{'overall_lineage_status':str(overall)})
    assert pointer['execution']==str(root/'status.json')
    assert pointer['lineage']==str(overall.resolve())
