"""Fully bound CPU preflight fixtures, no production payload/backend/physics."""
import copy
import json
from types import SimpleNamespace as NS
from pathlib import Path
import numpy as np
import pytest
from .test_retention_consolidation import config
from .test_retention_bank import IDENTITY,records
from .test_generative_bridge_student_v12 import teacher
from jit_dvgc.generative_bridge.contracts import file_sha
from jit_dvgc.generative_bridge.retention_bank import build_retention_bank
from jit_dvgc.generative_bridge.data import export_student_demonstrations
from jit_dvgc.retention_consolidation import REQUIRED_GROUPS


def dump(path,value):
    path.write_text(json.dumps(value));return dict(path=str(path),sha256=file_sha(path))


def fixture(tmp_path,monkeypatch):
    import jit_dvgc.generative_bridge.worker as worker
    import jit_dvgc.handoff_bank as handoff
    import jit_dvgc.unified_policy_freeze as freeze
    import jit_dvgc.retention_consolidation as stage
    # No JAX pytree conversion or serialized production policy in this audit fixture.
    monkeypatch.setattr(handoff,'pytree_sha256',lambda x:x)
    monkeypatch.setattr(stage,'code_identity',lambda repo:{'commit':'fixture','dirty':False})
    cp=tmp_path/'checkpoint';cp.mkdir();(cp/'payload.pkl').write_bytes(b'bound payload')
    from jit_dvgc.generative_bridge.contracts import digest
    upstream=tmp_path/'up.json';upstream.write_text(json.dumps({'action':{'fixture':True}}))
    cfg=tmp_path/'source_config.json';cfg.write_text(json.dumps({'inputs':{'up_config_path':str(upstream)}}))
    xml=tmp_path/'model.xml';xml.write_text('<mujoco/>')
    from jit_dvgc.config import canonical_sha256
    identity=NS(config_sha256=canonical_sha256(json.loads(cfg.read_text())),xml_sha256=file_sha(xml),actor_frame_fields=tuple('f'+str(i) for i in range(25)),
        actor_task_fields=('goal',),action_order=('steer','rear','hip','knee'))
    actor='a'*64;norm='b'*64;critic='d'*64
    policy=dict(checkpoint=str(cp),actor_sha256=actor,normalizer_sha256=norm,
        config_sha256=identity.config_sha256,xml_sha256=identity.xml_sha256)
    dump(cp/'identity.json',dict(payload_sha256=file_sha(cp/'payload.pkl')))
    frozen=dump(tmp_path/'frozen.json',{'policy':policy})
    descriptor={**frozen, 'checkpoint':dict(path=str(cp/'payload.pkl'),sha256=file_sha(cp/'payload.pkl')),
        'config':dict(path=str(cfg),sha256=file_sha(cfg)), 'physics':dict(path=str(xml),sha256=file_sha(xml)),
        'actor_sha256':actor,'normalizer_sha256':norm,'critic_sha256':critic,
        'actor_schema':dict(observation_dim=76,frame_fields=list(identity.actor_frame_fields),task_fields=['goal'],history_window=3),
        'critic_schema':dict(observation_dim=106),
        'action_transform':dict(action_order=list(identity.action_order),transform_identity=digest({'fixture':True}))}
    payload=NS(identity=identity,actor_params=actor,observation_normalizer=norm,critic_params=critic)
    monkeypatch.setattr(worker,'source_payload',lambda frozen:(policy,payload))
    import jit_dvgc.unified_policy_freeze as freeze
    monkeypatch.setattr(freeze,'load_frozen_unified_manifest',lambda path:{'policy':policy})
    import jit_dvgc.unified_envelope_snapshot as snapshots
    monkeypatch.setattr(snapshots,'load_unified_envelope_snapshot',lambda path:NS(qpos=np.zeros(2),qvel=np.zeros(2)))
    monkeypatch.setattr(snapshots,'snapshot_context_sha256',lambda snapshot:'context')
    c=config();c.update(parent=copy.deepcopy(descriptor),pi0_reference=copy.deepcopy(descriptor),
        cost_limits=dict(environment_transitions=100000,wall_seconds=300,supervised_attempts=2000),
        tensorboard=dict(host='127.0.0.1',port=6006))
    task=dict(schema='frontierbridge_task_identity_v1',physics_sha256=identity.xml_sha256,task='fixture',reward='fixture',termination='stable_forward_recovery',
        actor_schema=descriptor['actor_schema'],critic_schema=descriptor['critic_schema'],action_transform=descriptor['action_transform'])
    c['task_identity']=dump(tmp_path/'task.json',task)
    bank_identity={**IDENTITY,'protocol_sha256':digest(task)}
    banks={}
    for role in ('TRAIN_FIT','TRAIN_GUARD'):
        rs=[]
        for i,g in enumerate(REQUIRED_GROUPS):
            r=records()[0];r['role']=role;r['capability_group']=g
            r.update(root_id=role+str(i),ancestor_ids=[role+str(i)],condition_identity=role+str(i))
            r['trace']['metadata']['root_id']=r['root_id']
            r['trace']['metadata']['condition_identity']=r['condition_identity']
            if g=='accepted_parent_recovery':
                r.update(source_policy_sha=actor,normalizer_sha=norm)
                r['trace']['metadata'].update(actor_sha256=actor,normalizer_sha256=norm)
            rs.append(r)
        manifest=build_retention_bank(rs,tmp_path/role,identity=bank_identity,reference_action_fn=lambda obs:np.zeros((len(obs),4)),group_weights={g:.25 for g in REQUIRED_GROUPS})
        banks[role]=manifest
        c['data_roles'][role]=dict(path=str(tmp_path/role/'manifest.json'),sha256=file_sha(tmp_path/role/'manifest.json'))
    from jit_dvgc.generative_bridge.contracts import digest
    t=teacher('teacher')
    trace_path=tmp_path/'teacher_trace.npz';np.savez_compressed(trace_path,**t['arrays'])
    verification=dict(label=1,root_id='teacher',root_context_sha256=t['metadata']['root_context_sha256'],
        source_trace_sha256=file_sha(trace_path),attempts=[dict(label=1,outcome='stable_forward_recovery')])
    t['metadata']['verification_receipt_sha256']=digest(verification)
    verification_descriptor=dump(tmp_path/'teacher_verification.json',verification)
    demo=export_student_demonstrations([t],tmp_path/'demo')
    demo['entries'][0].update(source_trace=dict(path=str(trace_path),sha256=file_sha(trace_path)),verification_receipt=verification_descriptor)
    demo['entries'][0].update(parent_repeat_labels=[0,0,0],teacher_repeat_labels=[1,1,1],ancestor_ids=['teacher'],condition_identity='teacher')
    c['data_roles']['TEACHER']=dump(tmp_path/'demo/manifest.json',demo)
    rows=[]
    for i,g in enumerate(REQUIRED_GROUPS):
        rows.append(dict(case_id=f'dev{i}',group=g,role='DEV_RETENTION',mode='collect',lane=0,
            capacity=1,horizon=400,repeat_seeds=[i,i,i],condition_identity=f'dev{i}',
            spec=dict(success_criterion='stable_forward_recovery',horizon=400,
                full_episode_rollout=True,controller_mode='fixed_random',delta_limit=0,num_envs=1,budget=400)))
    dev=teacher('new');dev['metadata'].update(role='DEV_EXPANSION',condition_identity='new',ancestor_ids=['new'],parent_repeat_labels=[0,0,0],teacher_repeat_labels=[1,1,1])
    dev_path=tmp_path/'dev_teacher.npz';np.savez_compressed(dev_path,**dev['arrays'])
    dev_receipt=dict(label=1,root_id='new',root_context_sha256='context',source_trace_sha256=file_sha(dev_path),attempts=[dict(label=1,outcome='stable_forward_recovery')])
    from jit_dvgc.generative_bridge.data import trajectory_identity
    evidence=dict(metadata=dev['metadata'],source_trace=dict(path=str(dev_path),sha256=file_sha(dev_path)),
        verification_receipt=dump(tmp_path/'dev_receipt.json',dev_receipt),trajectory_sha256=trajectory_identity(dev))
    snapshot=tmp_path/'snapshot';snapshot.mkdir();(snapshot/'snapshot.pkl').write_bytes(b'fixture');dump(snapshot/'identity.json',{})
    time_path=tmp_path/'source_time.npz';np.savez(time_path,prefix_mask=np.ones((1,1),bool),time_after=np.zeros((1,1)),qpos=np.zeros((1,1,2)),qvel=np.zeros((1,1,2)))
    time_provenance=dict(path=str(time_path),sha256=file_sha(time_path),tick=0,lane=0,time_field='time_after')
    candidates=dump(tmp_path/'candidates.json',[dict(time_provenance=time_provenance,candidate_id=0,role='DEV_EXPANSION',snapshot=str(snapshot),snapshot_time=0.,snapshot_context_sha256='context')])
    expansion=copy.deepcopy(rows[0]);expansion.update(case_id='new',condition_identity='new',role='DEV_EXPANSION',group='new_recovery',mode='evaluate',teacher_evidence=dump(tmp_path/'dev_evidence.json',evidence))
    expansion['spec']=dict(horizon=400,budget=400,success_criterion='stable_forward_recovery',preserve_snapshot_episode_context=True,candidates=candidates['path'])
    rows.append(expansion)
    panel=dict(cases=rows,split_records=[dict(root_id=f'dev{i}',ancestor_ids=[f'dev{i}'],
        condition_identity=f'dev{i}',role='DEV_RETENTION') for i in range(4)]+[dict(root_id='new',ancestor_ids=['new'],condition_identity='new',role='DEV_EXPANSION')])
    c['evaluation_panel']=dump(tmp_path/'panel.json',panel)
    thresholds={g:[.01]*4 for g in REQUIRED_GROUPS}
    receipt=dict(schema='frontierbridge_guard_calibration_v1',calibrated=True,data_role='TRAIN_GUARD',
        cost_ledger={'environment_transitions':0,'forward_evaluations':8},candidate_thresholds=[thresholds],
        guard_bank_sha256=banks['TRAIN_GUARD']['sha256'],parent_actor_sha256=actor,
        mse_thresholds=thresholds,p99_thresholds=thresholds)
    c['guard_config']=dict(calibrated=True,max_consecutive_rejections=10,mse_thresholds=thresholds,p99_thresholds=thresholds,
        calibration_receipt=dump(tmp_path/'calibration.json',receipt))
    c['selection_config']=dict(required_groups={g:dict(max_lost=0,max_rate_drop=0.) for g in REQUIRED_GROUPS},
        min_expansion_gain=1,mixed_policy='block',unknown_policy='block',parent_pi0_gap_decision='preflight_required')
    return c,policy,payload


def prepare_fixture(tmp_path,monkeypatch,change=None):
    from jit_dvgc.retention_consolidation import prepare
    c,policy,payload=fixture(tmp_path,monkeypatch)
    if change:change(c,policy,payload)
    d=dump(tmp_path/'config.json',c)
    plan=prepare(d['path'],tmp_path/'output',tmp_path)
    return plan,c


def test_complete_identity_fixture_ready_and_payloads_recursively_locked(tmp_path,monkeypatch):
    from jit_dvgc.retention_consolidation import audit
    plan,c=prepare_fixture(tmp_path,monkeypatch)
    result=audit(plan)
    assert result['status']=='ready',result
    locked=json.loads(Path(plan).read_text())['input_locks']
    for name in ('TRAIN_FIT','TRAIN_GUARD','demo'):
        payload=tmp_path/name/('demonstrations.npz' if name=='demo' else 'bank.npz')
        assert str(payload) in locked
    assert str(tmp_path/'checkpoint/identity.json') in locked
    assert result['optimizer_updates']==result['environment_transitions']==0


@pytest.mark.parametrize('key', ['normalizer_sha256','critic_sha256','actor_schema','physics','action_transform'])
def test_parent_payload_and_protocol_binding(tmp_path,monkeypatch,key):
    from jit_dvgc.retention_consolidation import audit
    def change(c,policy,payload):
        if key in ('normalizer_sha256','critic_sha256'):c['parent'][key]='0'*64
        elif key=='actor_schema':c['parent'][key]['observation_dim']=106
        elif key=='physics':payload.identity.xml_sha256='0'*64
        else:c['parent'][key]['action_order'].reverse()
    plan,c=prepare_fixture(tmp_path,monkeypatch,change)
    with pytest.raises(ValueError,match='mismatch'):audit(plan)


def test_recursively_locked_npz_mutation_detected(tmp_path,monkeypatch):
    from jit_dvgc.retention_consolidation import audit
    plan,c=prepare_fixture(tmp_path,monkeypatch)
    with (tmp_path/'TRAIN_GUARD/bank.npz').open('ab') as f:f.write(b'drift')
    with pytest.raises(ValueError,match='input.*drift'):audit(plan)


def test_calibration_boolean_without_bound_actual_receipt_is_not_ready(tmp_path,monkeypatch):
    from jit_dvgc.retention_consolidation import audit
    def change(c,*args):
        c['guard_config']['calibration_receipt']=dump(tmp_path/'invalid_calibration.json',{'calibrated':True})
    plan,c=prepare_fixture(tmp_path,monkeypatch,change)
    with pytest.raises(ValueError,match='calibrated receipt'):audit(plan)


def test_group_channel_threshold_nan_fails(tmp_path,monkeypatch):
    from jit_dvgc.retention_consolidation import audit
    def change(c,*args):c['guard_config']['mse_thresholds']['old_nominal'][0]=float('nan')
    with pytest.raises(ValueError,match='JSON compliant|finite nonnegative'):
        plan,c=prepare_fixture(tmp_path,monkeypatch,change)
        audit(plan)


def test_candidate_snapshot_dependencies_locked(tmp_path):
    from jit_dvgc.retention_consolidation import _lock_descriptors
    snapshot=tmp_path/'snapshot';snapshot.mkdir()
    (snapshot/'snapshot.pkl').write_bytes(b'full context')
    dump(snapshot/'identity.json',{'snapshot_sha256':file_sha(snapshot/'snapshot.pkl')})
    candidates=dump(tmp_path/'candidates.json',[{'snapshot':str(snapshot)}])
    spec=dump(tmp_path/'spec.json',dict(candidates=candidates['path']))
    locks={};missing=[]
    _lock_descriptors(spec,locks,missing)
    assert not missing
    assert str(snapshot/'snapshot.pkl') in locks and str(snapshot/'identity.json') in locks


def test_teacher_condition_leakage_into_dev_is_rejected(tmp_path,monkeypatch):
    from jit_dvgc.retention_consolidation import audit
    def change(c,*args):
        desc=c['data_roles']['TEACHER'];m=json.loads(Path(desc['path']).read_text())
        m['entries'][0]['condition_identity']='dev0'
        c['data_roles']['TEACHER']=dump(Path(desc['path']),m)
    plan,c=prepare_fixture(tmp_path,monkeypatch,change)
    with pytest.raises(ValueError,match='leakage'):audit(plan)


def test_missing_bank_parent_hash_or_guard_group_not_ready(tmp_path,monkeypatch):
    from jit_dvgc.retention_consolidation import audit
    def change(c,*args):c['parent'].pop('critic_sha256')
    plan,c=prepare_fixture(tmp_path,monkeypatch,change)
    report=audit(plan)
    assert report['status']=='blocked' and 'parent.critic_sha256' in report['unresolved']


def test_missing_expansion_or_panel_split_never_ready(tmp_path,monkeypatch):
    from jit_dvgc.retention_consolidation import audit
    def change(c,*args):
        desc=c['evaluation_panel'];panel=json.loads(Path(desc['path']).read_text())
        panel['cases']=[x for x in panel['cases'] if x['role']=='DEV_RETENTION'];panel['split_records']=[]
        c['evaluation_panel']=dump(Path(desc['path']),panel)
    plan,c=prepare_fixture(tmp_path,monkeypatch,change)
    report=audit(plan)
    assert report['status']=='blocked'
    assert {'evaluation_panel.DEV_EXPANSION','evaluation_panel.split_records'}<=set(report['unresolved'])


def test_existing_pi0_read_only_manifest_contract():
    """Local artifact integration audit; no optimizer/environment construction."""
    import os
    from jit_dvgc.retention_consolidation import _audit_policy
    from jit_dvgc.generative_bridge.contracts import digest
    path=Path(os.environ.get('FRONTIERBRIDGE_REAL_PI0', '/home/qy/DVGC/JIT/runs/experiments/generative_bridge_v1_2_20260928/audit/binding/frozen_unified_policy.json'))
    if not path.exists():pytest.skip('external production pi0 artifact unavailable; CPU fixtures are separate')
    policy=json.loads(path.read_text())['policy']
    cfg=Path(policy['formal_config']);raw=json.loads(cfg.read_text())
    upstream=Path(raw['inputs']['up_config_path']);up=json.loads(upstream.read_text())
    xml=Path('/home/qy/DVGC')/up['model']['xml_path']
    descriptor=dict(path=str(path),sha256=file_sha(path),
        checkpoint=dict(path=str(Path(policy['checkpoint'])/'payload.pkl'),sha256=policy['payload_sha256']),
        config=dict(path=str(cfg),sha256=file_sha(cfg)),physics=dict(path=str(xml),sha256=file_sha(xml)),
        actor_sha256=policy['actor_sha256'],normalizer_sha256=policy['normalizer_sha256'],critic_sha256=policy['critic_sha256'],
        actor_schema=dict(observation_dim=76,frame_fields=policy['actor_frame_fields'],task_fields=policy['actor_task_fields'],history_window=3),
        critic_schema=dict(observation_dim=106),action_transform=dict(action_order=policy['action_order'],transform_identity=digest(up['action'])))
    missing=[];result=_audit_policy(descriptor,'pi0_reference',missing)
    assert not missing
    assert result['actor_sha256']==policy['actor_sha256']


def test_stage_parent_manifest_uses_the_same_real_binding_contract(tmp_path,monkeypatch):
    from jit_dvgc.generative_bridge.retention_bank import audit_parent_manifest
    plan,c=prepare_fixture(tmp_path,monkeypatch)
    manifest=json.loads((tmp_path/'output/parent_manifest.json').read_text())
    assert audit_parent_manifest(manifest)['training_ready'] is True


def test_null_threshold_channels_are_unresolved_not_calibrated(tmp_path,monkeypatch):
    from jit_dvgc.retention_consolidation import _audit_thresholds
    missing=[]
    guard=dict(calibrated=False,max_consecutive_rejections=10,
        mse_thresholds={g:[None]*4 for g in REQUIRED_GROUPS},p99_thresholds={g:[None]*4 for g in REQUIRED_GROUPS})
    _audit_thresholds(guard,{}, {}, missing)
    assert 'guard_config.mse_thresholds.old_nominal' in missing
    assert 'guard_config.calibration' in missing


def test_unrelated_teacher_verification_receipt_root_rejected(tmp_path,monkeypatch):
    from jit_dvgc.retention_consolidation import audit
    from jit_dvgc.generative_bridge.contracts import digest
    def change(c,*args):
        desc=c['data_roles']['TEACHER'];teacher_manifest=json.loads(Path(desc['path']).read_text())
        entry=teacher_manifest['entries'][0];receipt=json.loads(Path(entry['verification_receipt']['path']).read_text())
        receipt['root_id']='unrelated'
        entry['verification_receipt']=dump(Path(entry['verification_receipt']['path']),receipt)
        entry['verification_receipt_sha256']=digest(receipt)
        c['data_roles']['TEACHER']=dump(Path(desc['path']),teacher_manifest)
    plan,c=prepare_fixture(tmp_path,monkeypatch,change)
    with pytest.raises(ValueError,match='source root/context/trace'):audit(plan)


def test_forged_snapshot_clock_fails_before_physics(tmp_path,monkeypatch):
    from jit_dvgc.retention_consolidation import audit
    def change(c,*args):
        panel_desc=c['evaluation_panel'];panel=json.loads(Path(panel_desc['path']).read_text())
        path=Path(panel['cases'][-1]['spec']['candidates']);rows=json.loads(path.read_text());rows[0]['snapshot_time']=1e9
        dump(path,rows)
    plan,c=prepare_fixture(tmp_path,monkeypatch,change)
    with pytest.raises(ValueError,match='active recorded source frame'):audit(plan)


def test_only_declared_repository_model_assets_use_repository_fallback(tmp_path):
    from jit_dvgc.retention_consolidation import _lock_descriptors
    repo=tmp_path/'repo';(repo/'assets').mkdir(parents=True)
    asset=repo/'assets/model.xml';asset.write_text('<mujoco/>')
    old_config=tmp_path/'old/run';old_config.mkdir(parents=True)
    locks={};missing=[]
    _lock_descriptors(dict(model=dict(xml_path='assets/model.xml',xml_sha256=file_sha(asset))),
        locks,missing,prefix='source',base=old_config,repository_root=repo)
    assert not missing and str(asset) in locks
    generic_locks={};generic_missing=[]
    _lock_descriptors(dict(snapshot='assets/model.xml'),generic_locks,generic_missing,
        prefix='source',base=old_config,repository_root=repo)
    assert generic_missing and not generic_locks
    with pytest.raises(ValueError,match='hash drift'):
        _lock_descriptors(dict(model=dict(xml_path='assets/model.xml',xml_sha256='0'*64)),
            {},[],prefix='source',base=old_config,repository_root=repo)


def test_real_parent_config_tree_prepare_locks_actual_repository_assets(tmp_path):
    """Regression for nested historical source configs with relative model paths."""
    import os
    from jit_dvgc.retention_consolidation import _lock_descriptors
    path=Path(os.environ.get('FRONTIERBRIDGE_REAL_PI0', '/home/qy/DVGC/JIT/runs/experiments/generative_bridge_v1_2_20260928/audit/binding/frozen_unified_policy.json'))
    if not path.exists():pytest.skip('external production pi0 unavailable')
    root=Path(__file__).resolve().parents[2]
    locked={};missing=[]
    _lock_descriptors(dict(parent=dict(path=str(path),sha256=file_sha(path))),locked,missing,
        base=tmp_path,repository_root=root)
    assert missing==[]
    policy=json.loads(path.read_text())['policy']
    raw=json.loads(Path(policy['formal_config']).read_text())
    for key in ('up_config_path','down_config_path'):
        phase=json.loads(Path(raw['inputs'][key]).read_text())
        assert str((root/phase['model']['xml_path']).resolve()) in locked


def test_full_real_prepared_stage_has_only_uncalibrated_guard_blocker(tmp_path,monkeypatch):
    from jit_dvgc.retention_consolidation import prepare,audit
    import jit_dvgc.retention_consolidation as stage
    source=Path('/home/qy/DVGC/JIT/runs/experiments/frontierbridge_pi0_pilot_20261013/calibration_stage_config.json')
    if not source.exists():pytest.skip('external real prepared pilot inputs unavailable')
    c=json.loads(source.read_text());c['execution_code_identity']=None;c['guard_config']=None
    path=tmp_path/'actual_config.json';path.write_text(json.dumps(c))
    # This test concerns actual frozen inputs; current dirty-code drift is tested
    # independently and does not depend on other team edits during pytest.
    monkeypatch.setattr(stage,'code_identity',lambda repo:{'commit':'fixture_code_audit_scope'})
    repo=Path(__file__).resolve().parents[2]
    plan=prepare(path,tmp_path/'prepared',repo)
    result=audit(plan)
    assert result['status']=='blocked'
    assert result['unresolved']==['guard_config'],result['unresolved']
    actual=json.loads(Path(plan).read_text())
    assert actual['preparation_missing']==[]
    assert str((repo/'assets/orange_bike_4kg_horizontal.xml').resolve()) in actual['input_locks']
