"""No physical environment is needed to prepare or refuse execution."""
import json
from pathlib import Path
import pytest


def config():
    return dict(schema='frontierbridge_retention_consolidation_v1',
        parent=None, pi0_reference=None, data_roles={}, guard_config=None,
        selection_config=None, seed=13, max_attempted_updates=1000,
        checkpoint_steps=[0,100,250,500,1000], cost_limits=None,
        execution_code_identity=None, automatic_next_stage=False,
        arms=['P0','P1','P2'], evaluation_panel=None,
        tensorboard=dict(host='127.0.0.1',port=None))


def prepare(tmp_path):
    from jit_dvgc.retention_consolidation import prepare
    c=tmp_path/'config.json';c.write_text(json.dumps(config()))
    repo=Path(__file__).resolve().parents[2]
    return prepare(c,tmp_path/'output',repo)


def test_prepare_dry_run_no_optimizer_or_physics(tmp_path, monkeypatch):
    import optax
    monkeypatch.setattr(optax,'adam',lambda *a,**k:pytest.fail('optimizer created'))
    from jit_dvgc.retention_consolidation import dry_run
    plan=prepare(tmp_path)
    report=dry_run(plan)
    assert report['status']=='blocked'
    assert {'parent','guard_config','cost_limits'}<=set(report['unresolved'])
    assert report['environment_transitions']==report['optimizer_updates']==0
    assert (tmp_path/'output/parent_manifest.json').exists()
    assert (tmp_path/'output/input_lock.json').exists()


def test_run_and_worker_cannot_bypass_unresolved_or_authorization(tmp_path):
    from jit_dvgc.retention_consolidation import run,worker
    plan=prepare(tmp_path)
    for fn in (run,worker):
        with pytest.raises(ValueError,match='blocked|authorization'):
            fn(plan,None)
    assert json.loads((tmp_path/'output/status.json').read_text())['phase']=='prepared'


def test_config_drift_fails_closed(tmp_path):
    from jit_dvgc.retention_consolidation import audit
    plan=prepare(tmp_path)
    (tmp_path/'config.json').write_text('{}')
    with pytest.raises(ValueError,match='drift'):
        audit(plan)


def test_no_historical_budget_or_automatic_stage(tmp_path):
    from jit_dvgc.retention_consolidation import validate_config
    c=config();c['automatic_next_stage']=True
    with pytest.raises(ValueError,match='automatic'):validate_config(c)
    c=config();c['max_attempted_updates']=1001
    with pytest.raises(ValueError,match='attempt'):validate_config(c)
    c=config();c['cost_limits']={'environment_transitions':1,'wall_seconds':1,
        'supervised_attempts':1,'prior_charged':1}
    with pytest.raises(ValueError,match='attempt|budget'):validate_config(c)


def test_panel_rejects_test_composite_and_missing_repeats():
    from jit_dvgc.retention_consolidation import validate_panel
    case=dict(case_id='one',group='old_nominal',role='TEST',mode='evaluate',
        lane=0,capacity=1,horizon=400,spec={'bridge_action_plan':{}},repeat_seeds=[1])
    with pytest.raises(ValueError):validate_panel({'cases':[case]})


def test_report_without_training_does_not_claim_improvement(tmp_path):
    from jit_dvgc.retention_consolidation import report
    result=report(prepare(tmp_path))
    assert result['published_policy'] is None
    assert result['training']=='not_run'
    assert result['conclusion']=='engineering_blocked'


def test_completed_execution_with_guard_rejection_is_not_method_success(tmp_path):
    from jit_dvgc.retention_consolidation import report
    plan=prepare(tmp_path);root=tmp_path/'output'
    (root/'status.json').write_text(json.dumps({'phase':'completed'}))
    (root/'P2').mkdir();(root/'P2/warmup_status.json').write_text(json.dumps({'status':'blocked_by_guard','attempts':1,'accepted_updates':0}))
    result=report(plan)
    assert result['conclusion']=='blocked_by_guard'
    assert result['arms']['P2']['accepted_updates']==0


def test_diagnostic_behavior_receipt_requires_explicit_diagnostic_mode(tmp_path):
    from jit_dvgc.retention_consolidation import _audit_thresholds,REQUIRED_GROUPS
    from jit_dvgc.generative_bridge.contracts import file_sha
    thresholds={g:[.01]*4 for g in REQUIRED_GROUPS}
    receipt=dict(schema='frontierbridge_guard_behavior_calibration_diagnostic_v1',
        behavior_calibrated=True,closed_loop_qualified=False,data_role='TRAIN_GUARD',
        guard_bank_sha256='bank',parent_actor_sha256='actor',
        cost_ledger={'path':'ledger'},candidate_thresholds=[{'attempt':20}],
        mandatory_group_status={'old_nominal':'UNVERIFIED_BASELINE_MIXED'},
        mse_thresholds=thresholds,p99_thresholds=thresholds)
    path=tmp_path/'receipt.json';path.write_text(json.dumps(receipt))
    guard=dict(behavior_calibrated=True,closed_loop_qualified=False,
        max_consecutive_rejections=10,mse_thresholds=thresholds,p99_thresholds=thresholds,
        calibration_receipt=dict(path=str(path),sha256=file_sha(path)))
    missing=[]
    _audit_thresholds(guard,{'sha256':'bank'},{'actor_sha256':'actor'},missing,diagnostic_training=True)
    assert missing==[]
    with pytest.raises(ValueError,match='diagnostic'):
        _audit_thresholds(guard,{'sha256':'bank'},{'actor_sha256':'actor'},[],diagnostic_training=False)
    receipt['closed_loop_qualified']=True;path.write_text(json.dumps(receipt))
    guard['calibration_receipt']['sha256']=file_sha(path)
    with pytest.raises(ValueError,match='diagnostic'):
        _audit_thresholds(guard,{'sha256':'bank'},{'actor_sha256':'actor'},[],diagnostic_training=True)


def test_diagnostic_completed_report_keeps_qualification_unverified(tmp_path):
    from jit_dvgc.retention_consolidation import report,read
    plan=prepare(tmp_path);p=read(plan)
    p['config']['selection_config']={'diagnostic_training':True}
    Path(plan).write_text(json.dumps(p))
    root=tmp_path/'output';(root/'status.json').write_text(json.dumps({'phase':'completed'}))
    (root/'P2').mkdir();(root/'P2/warmup_status.json').write_text(json.dumps({'status':'completed','attempts':1000,'accepted_updates':1000}))
    result=report(plan)
    assert result['conclusion']=='diagnostic_training_only_retention_unverified'
    assert result['closed_loop_qualified'] is False and result['automatic_adoption'] is False
    assert result['deterministic_policy_retention']=='UNVERIFIED'
    assert result['published_policy'] is None


def test_record_only_execution_requires_matching_user_grant(tmp_path,monkeypatch):
    from jit_dvgc import retention_consolidation as stage
    from jit_dvgc.generative_bridge.contracts import file_sha
    plan=prepare(tmp_path);p=stage.read(plan)
    p['config']['selection_config']={'diagnostic_training':True,'retention_gate_mode':'record_only'}
    Path(plan).write_text(json.dumps(p));monkeypatch.setattr(stage,'audit',lambda _:dict(status='ready'))
    grant=dict(schema='frontierbridge_execution_authorization_v1',execute=True,plan_sha256=file_sha(plan),
        cost_limits=p['config']['cost_limits'],arms=['P0','P1','P2'],user_authorization_record='actual user',
        diagnostic_training=True,automatic_adoption=False,nominal_retention_status='UNVERIFIED_BASELINE_MIXED')
    auth=tmp_path/'auth.json';auth.write_text(json.dumps(grant))
    with pytest.raises(ValueError,match='record.only'):stage.require_execution(plan,auth)
    grant['retention_gate_mode']='record_only';auth.write_text(json.dumps(grant))
    stage.require_execution(plan,auth)
