"""Independent metadata/serialization regressions; zero physical environment steps."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from jit_dvgc import retention_consolidation as stage
from jit_dvgc.generative_bridge.contracts import file_sha,digest


def dump(path,value):
    path.write_text(json.dumps(value));return dict(path=str(path),sha256=file_sha(path))


def snapshot_panel(tmp_path):
    from jit_dvgc.unified_envelope_snapshot import (UnifiedEnvelopeSnapshot,
        UP_EVENT_FIELDS,DOWN_EVENT_FIELDS,save_unified_envelope_snapshot,snapshot_context_sha256)
    events=lambda names:{k:np.asarray(0) for k in names}
    snap=UnifiedEnvelopeSnapshot(qpos=np.arange(9,dtype=np.float32),qvel=np.arange(8,dtype=np.float32),
        observation_fifo=np.arange(75,dtype=np.float32).reshape(3,25),history_valid_count=3,
        observation=np.arange(76,dtype=np.float32),last_action=np.zeros(4),ctrl=np.zeros(4),rng=np.array([1,2],np.uint32),
        up_events=events(UP_EVENT_FIELDS),down_events=events(DOWN_EVENT_FIELDS),active_phase=1,start_phase=0,
        phase_transitioned=True,episode_step=17,phase_episode_step=4,episode_return=3.25,reset_from_soft_tube=True,
        source_tick=9,parent_group_index=2,tube_entry_index=3,tube_global_index=8,parent_trajectory='ancestor',
        parent_state_sha256='1'*64,config_sha256='2'*64,xml_sha256='3'*64,policy_actor_sha256='4'*64,
        policy_payload_sha256='5'*64,policy_iteration=0,compatibility_identity={'contract':'metadata_fixture'})
    path=tmp_path/'snapshot';save_unified_envelope_snapshot(path,snap)
    source_time=tmp_path/'original_tape.npz';np.savez_compressed(source_time,mask=np.ones((1,1),bool),time_after=np.array([[.34]],float),qpos=snap.qpos[None,None,:],qvel=snap.qvel[None,None,:])
    row=dict(time_provenance=dict(path=str(source_time),sha256=file_sha(source_time),tick=0,lane=0,time_field='time_after'),candidate_id='root',snapshot=str(path),snapshot_context_sha256=snapshot_context_sha256(snap),
        snapshot_time=.34,role='DEV_RETENTION')
    rows_path=tmp_path/'roots.json';dump(rows_path,[row])
    case=dict(case_id='root',condition_identity='condition',group='old_recovery',role='DEV_RETENTION',
        repeat_seeds=[13,13,13],mode='evaluate',horizon=400,capacity=1,lane=0,
        spec=dict(success_criterion='stable_forward_recovery',horizon=400,budget=400,
            preserve_snapshot_episode_context=True,candidates=str(rows_path)))
    return {'cases':[case]},row,rows_path,snap


def test_canonical_snapshot_without_time_attribute_preflights(tmp_path):
    panel,row,path,snap=snapshot_panel(tmp_path)
    assert not hasattr(snap,'time')
    sidecar=json.loads((tmp_path/'snapshot/identity.json').read_text())
    assert 'snapshot_context_sha256' not in sidecar and 'context_sha256' not in sidecar
    assert stage.validate_panel(panel)==panel['cases']


@pytest.mark.parametrize('change',['missing_time','nonfinite_time','wrong_context','payload_drift'])
def test_snapshot_context_time_and_payload_fail_before_dispatch(tmp_path,change):
    panel,row,path,snap=snapshot_panel(tmp_path)
    if change=='missing_time':row.pop('snapshot_time')
    elif change=='nonfinite_time':row['snapshot_time']=float('inf')
    elif change=='wrong_context':row['snapshot_context_sha256']='0'*64
    else:
        with (tmp_path/'snapshot/snapshot.pkl').open('ab') as f:f.write(b'drift')
    dump(path,[row])
    with pytest.raises(ValueError,match='time|context|SHA'):stage.validate_panel(panel)


def verified_teacher(tmp_path):
    from .test_generative_bridge_student_v12 import teacher
    from jit_dvgc.generative_bridge.data import export_student_demonstrations
    t=teacher('teacher_root')
    source=tmp_path/'teacher_trace.npz';np.savez_compressed(source,**t['arrays'])
    verification=dict(root_id=t['metadata']['root_id'],root_context_sha256=t['metadata']['root_context_sha256'],
        source_trace_sha256=file_sha(source),label=1,attempts=[dict(label=1,outcome='stable_forward_recovery')])
    t['metadata']['verification_receipt_sha256']=digest(verification)
    demo=export_student_demonstrations([t],tmp_path/'demo')
    row=demo['entries'][0]
    row.update(source_trace={'path':str(source),'sha256':file_sha(source)},
        verification_receipt=dump(tmp_path/'verification.json',verification))
    return demo,verification


def test_verified_teacher_receipt_exact_frame_source(tmp_path):
    demo,receipt=verified_teacher(tmp_path)
    unresolved=[];stage._audit_teacher_sources(demo,unresolved)
    assert unresolved==[]


@pytest.mark.parametrize('change',['root','context','source_trace','dataset_actions'])
def test_teacher_receipt_cannot_be_borrowed_or_frames_replaced(tmp_path,change):
    demo,receipt=verified_teacher(tmp_path);row=demo['entries'][0]
    if change=='dataset_actions':
        with np.load(demo['path'],allow_pickle=False) as raw:arrays={k:raw[k] for k in raw.files}
        arrays['actions'][0,0]=.1;np.savez_compressed(demo['path'],**arrays)
        demo['sha256']=file_sha(demo['path'])
    else:
        key={'root':'root_id','context':'root_context_sha256','source_trace':'source_trace_sha256'}[change]
        receipt[key]='other_root' if change=='root' else '0'*64
        row['verification_receipt']=dump(Path(row['verification_receipt']['path']),receipt)
        row['verification_receipt_sha256']=digest(receipt)
    with pytest.raises(ValueError,match='root/context/trace|dataset differs'):
        stage._audit_teacher_sources(demo,[])


def test_test_bank_refused_before_payload_hash_or_array_open(tmp_path,monkeypatch):
    from jit_dvgc.generative_bridge import retention_bank
    payload=tmp_path/'TEST.npz';np.savez_compressed(payload,observations=np.zeros((1,76)))
    manifest=tmp_path/'TEST_manifest.json';dump(manifest,dict(records=[dict(role='TEST')],path=str(payload),sha256=file_sha(payload)))
    def forbidden(*args,**kwargs):pytest.fail('locked TEST payload bytes were opened')
    monkeypatch.setattr(retention_bank,'file_sha',forbidden);monkeypatch.setattr(np,'load',forbidden)
    with pytest.raises(ValueError,match='role|TEST'):retention_bank.load_retention_bank(manifest)


def test_prepare_rejects_test_data_role_before_hashing_its_bank(tmp_path,monkeypatch):
    from .test_retention_consolidation import config
    payload=tmp_path/'TEST.npz';np.savez_compressed(payload,observations=np.zeros((1,76)))
    manifest=dump(tmp_path/'TEST_manifest.json',dict(records=[dict(role='TEST')],path=str(payload),sha256=file_sha(payload)))
    c=config();c['data_roles']['TEST']=manifest
    cpath=tmp_path/'config.json';dump(cpath,c)
    actual=stage.file_sha
    def checked(path):
        assert Path(path).resolve()!=payload.resolve(),'TEST bytes opened during prepare'
        return actual(path)
    monkeypatch.setattr(stage,'file_sha',checked)
    with pytest.raises(ValueError,match='TEST'):stage.prepare(cpath,tmp_path/'output',tmp_path)
    assert not (tmp_path/'output').exists()
