import copy
import json
import numpy as np
import pytest
from jit_dvgc.generative_bridge.retention_bank import (build_retention_bank,
    load_retention_bank, validate_retention_bank, audit_split_isolation,
    audit_parent_manifest, environment_provenance)
from .test_generative_bridge_student_v12 import teacher

IDENTITY = dict(parent_actor_sha256='a'*64, normalizer_sha256='b'*64,
                protocol_sha256='c'*64, actor_observation_dim=76, action_dim=4)

def records():
    out=[]
    for i, n in enumerate((2, 4)):
        t=teacher(root=f'r{i}',n=n)
        t['arrays']['action_origin']=np.array(['actor_only']*n)
        t['metadata']['execution']='independent_complete'
        t['metadata']['condition_identity']=f'c{i}'
        for key, value in (('actor_sha256','d'*64),('normalizer_sha256','e'*64),('protocol_sha256','f'*64)):
            t['metadata'][key]=value
        # Use existing valid real-frame fixture without constructing fake history.
        out.append(dict(root_id=f'r{i}', ancestor_ids=[f'a{i}'], condition_identity=f'c{i}',
            role='TRAIN_FIT', capability_group='old_nominal', phase='UNKNOWN',
            source_policy_sha=t['metadata']['actor_sha256'], normalizer_sha=t['metadata']['normalizer_sha256'], protocol_sha=t['metadata']['protocol_sha256'],
            trace=t, valid_frame_range=[0, len(t['arrays']['valid_mask'])],
            validation_label='success'))
    return out

def build(tmp_path, rs=None):
    return build_retention_bank(rs or records(),tmp_path/'bank',identity=IDENTITY,
        reference_action_fn=lambda obs: np.zeros((len(obs),4)),group_weights={'old_nominal':1.})

def test_build_hash_bound_real_observation_labels_and_balance(tmp_path):
    m=build(tmp_path);m,a,audit=load_retention_bank(tmp_path/'bank/manifest.json')
    assert audit['post_contact_observations']=='UNKNOWN'
    assert a['observations'].shape[1]==76
    assert a['reference_actions'].shape[1]==4
    for rid in ('r0','r1'):
        assert a['weights'][a['episode_ids']==rid].sum()==pytest.approx(.5)
    assert m['identity']==IDENTITY
    with open(m['path'],'ab') as f:f.write(b'drift')
    with pytest.raises(ValueError,match='hash'):load_retention_bank(tmp_path/'bank/manifest.json')

@pytest.mark.parametrize('field,value', [('observations', np.zeros((4,106))),
    ('reference_actions',np.ones((4,4))*1.01),('weights', np.array([np.nan]*4))])
def test_illegal_arrays_fail_closed(tmp_path,field,value):
    m=build(tmp_path);_,a,_=load_retention_bank(tmp_path/'bank/manifest.json')
    a[field]=value
    with pytest.raises(ValueError):validate_retention_bank(m,a)

def test_test_role_never_loaded(tmp_path,monkeypatch):
    m=build(tmp_path);m['records'][0]['role']='TEST'
    p=tmp_path/'bank/manifest.json';p.write_text(json.dumps(m))
    monkeypatch.setattr(np,'load',lambda *a,**kw:pytest.fail('TEST arrays opened'))
    with pytest.raises(ValueError,match='role'):load_retention_bank(p)
    with pytest.raises(ValueError,match='TEST'):load_retention_bank(p,allowed_roles=('TEST',))

def test_ancestor_and_condition_cross_split_fail():
    base=dict(root_id='r',ancestor_ids=['a'],condition_identity='c',role='TRAIN_FIT')
    for key,val in [('ancestor_ids',['a']),('condition_identity','c')]:
        other=dict(root_id='other',ancestor_ids=['z'],condition_identity='z',role='DEV_RETENTION');other[key]=val
        with pytest.raises(ValueError,match='leakage'):audit_split_isolation([base,other])
    audit=audit_split_isolation([base,{**base,'role':'TRAIN_GUARD'}])
    assert audit['train_guard_reuse'] is True

def test_input_identity_and_padding_checked(tmp_path):
    rs=records();rs[0]['trace']['arrays']['valid_mask'][0]=0
    with pytest.raises(ValueError,match='padding'):build(tmp_path,rs)
    m=build(tmp_path)
    with pytest.raises(ValueError,match='identity'):
        load_retention_bank(tmp_path/'bank/manifest.json',expected_identity={**IDENTITY,'parent_actor_sha256':'0'*64})

def test_unknown_parent_manifest_is_blocked_and_environment_does_not_import_jax():
    report=audit_parent_manifest({'parent_candidate':None,'source_pi0':None})
    assert report['status']=='unresolved' and report['training_ready'] is False
    env=environment_provenance()
    assert env['devices']=='UNKNOWN_NOT_INITIALIZED'

@pytest.mark.parametrize('change', ['source_dev','reset','phase','source_hash','weights','parent_labels'])
def test_provenance_and_balancing_fail_closed(tmp_path,change):
    rs=records()
    if change=='source_dev':
        rs[0]['trace']['metadata']['role']='student_dev'
        with pytest.raises(ValueError,match='TRAIN'):build(tmp_path,rs)
    elif change=='reset':
        rs[0]['trace']['arrays']['done'][0]=True
        with pytest.raises(ValueError,match='terminal'):build(tmp_path,rs)
    elif change=='phase':
        rs[0]['phase']='post_contact'
        with pytest.raises(ValueError,match='phase'):build(tmp_path,rs)
    elif change=='source_hash':
        rs[0]['source_policy_sha']='0'*64
        with pytest.raises(ValueError,match='source trace identity'):build(tmp_path,rs)
    else:
        m=build(tmp_path);_,a,_=load_retention_bank(tmp_path/'bank/manifest.json')
        if change=='weights':
            a['weights']=np.full(len(a['weights']),1/len(a['weights']))
            with pytest.raises(ValueError,match='balance'):validate_retention_bank(m,a)
        else:
            m['reference_identity']={**IDENTITY,'parent_actor_sha256':'0'*64}
            with pytest.raises(ValueError,match='reference labels'):validate_retention_bank(m,a)


def test_parent_audit_checks_artifact_bytes_without_percentage_gate(tmp_path):
    from jit_dvgc.generative_bridge.contracts import file_sha
    p=tmp_path/'artifact';p.write_bytes(b'fixture')
    refs={k:dict(path=str(p),sha256=file_sha(p)) for k in ('payload','actor','normalizer','critic','config')}
    manifest=dict(source_pi0=copy.deepcopy(refs),parent_candidate=copy.deepcopy(refs),
        initializer='parent',retention_reference='parent',task_identity={'schema':'fixture'},
        execution_code_identity={'commit':'fixture'})
    assert audit_parent_manifest(manifest)['training_ready'] is True
    p.write_bytes(b'drift')
    with pytest.raises(ValueError,match='hash drift'):audit_parent_manifest(manifest)


def test_composite_teacher_cannot_be_an_old_independent_skill(tmp_path):
    rs=records();rs[0]['trace']['arrays']['action_origin']=np.array(['bridge_prefix','actor_only'])
    with pytest.raises(ValueError,match='independent complete'):build(tmp_path,rs)


def test_accepted_parent_recovery_requires_the_actual_parent_source(tmp_path):
    rs=records();rs[0]['capability_group']='accepted_parent_recovery'
    with pytest.raises(ValueError,match='parent recovery source'):
        build_retention_bank(rs,tmp_path/'bank',identity=IDENTITY,
            reference_action_fn=lambda obs:np.zeros((len(obs),4)),
            group_weights={'accepted_parent_recovery':.5,'old_nominal':.5})
