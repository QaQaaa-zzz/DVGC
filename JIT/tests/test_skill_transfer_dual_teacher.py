import copy
import importlib
import json
import numpy as np
import pytest


def api():
    assert importlib.util.find_spec('jit_dvgc.skill_transfer_dual_teacher'), 'dual-teacher module missing'
    return importlib.import_module('jit_dvgc.skill_transfer_dual_teacher')


def fixture(tmp_path):
    from jit_dvgc.generative_bridge.contracts import file_sha,digest
    ctx=tmp_path/'context.json';ctx.write_text('{"complete_context":true}')
    query=dict(sample_id='s1',root_id='train1',ancestor_root_id='train1',split='TRAIN',
        student_actor_sha256='s'*64,context_sha256=file_sha(ctx),context_receipt=dict(path=str(ctx),sha256=file_sha(ctx)),
        observation_before=[0.]*76,control_step=10,timestamp='2026-10-14T00:00:00Z',time_before=.2,remaining_horizon=390,
        model_sha256='m'*64,protocol_sha256='p'*64,source_type='student_visited_query',
        student_closed_loop=True,teacher_used_as_driver=False,task_group='new_recovery')
    teachers={key:dict(actor_sha256=key[0]*64,normalizer_sha256='n'*64) for key in ('pi0','R')}
    references={}
    for name,label in [('pi0',0),('R',1)]:
        record=dict(teacher=name,**teachers[name],context_sha256=query['context_sha256'],
            root_id=query['root_id'],ancestor_root_id=query['ancestor_root_id'],split='TRAIN',
            observation_sha256=digest(query['observation_before']),time_before=.2,remaining_horizon=390,
            model_sha256='m'*64,protocol_sha256='p'*64,control_step=10,
            execution_mode='independent_frozen_teacher',success_criterion='stable_forward_recovery',
            repeat_labels=[label]*3,normalized_action=[.2 if name=='pi0' else .7]*4)
        path=tmp_path/(name+'.json');path.write_text(json.dumps(record))
        references[name]=dict(path=str(path),sha256=file_sha(path))
    return query,teachers,references


def test_strict_teacher_selection():
    select=api().choose_verified_teacher
    assert select(1,1)=='pi0'
    assert select(1,0)=='pi0'
    assert select(0,1)=='R'
    for a,b in [(0,0),(1,'unknown'),('mixed',1),('execution_error',1),(1,'timeout')]:assert select(a,b) is None


def test_same_context_verified_label_and_provenance(tmp_path):
    query,teachers,refs=fixture(tmp_path)
    row=api().label_verified_query(query,refs,teachers)
    assert row['chosen_teacher']=='R' and row['target_action']==[.7]*4
    assert row['pi0_outcome']==0 and row['R_outcome']==1
    assert row['verification_receipts']==refs and row['context_receipt']==query['context_receipt']
    assert row['source_type']=='student_visited_query'
    assert api().audit_online_label(row,teachers)==row


def test_context_mismatch_dev_and_offline_sources_rejected(tmp_path):
    query,teachers,refs=fixture(tmp_path)
    for field,value in [('split','DEV'),('source_type','offline_G_composite'),('teacher_used_as_driver',True)]:
        bad=copy.deepcopy(query);bad[field]=value
        with pytest.raises(ValueError):api().label_verified_query(bad,refs,teachers)
    path=tmp_path/'R.json';record=json.loads(path.read_text());record['remaining_horizon']=400;path.write_text(json.dumps(record))
    from jit_dvgc.generative_bridge.contracts import file_sha
    refs['R']['sha256']=file_sha(path)
    with pytest.raises(ValueError,match='same context'):api().label_verified_query(query,refs,teachers)


def test_unqueried_actions_and_receipt_tampering_rejected(tmp_path):
    query,teachers,refs=fixture(tmp_path)
    row=api().label_verified_query(query,refs,teachers);row['target_action'][0]=-.9
    with pytest.raises(ValueError,match='target'):api().audit_online_label(row,teachers)
    (tmp_path/'R.json').write_text('{}')
    with pytest.raises(ValueError,match='hash'):api().label_verified_query(query,refs,teachers)


def test_bank_quarantines_unidentifiable_actions(tmp_path):
    query,teachers,refs=fixture(tmp_path)
    row=api().label_verified_query(query,refs,teachers)
    second=copy.deepcopy(row);second['sample_id']='s2';second['target_action']=[-.7]*4
    conflicts=api().observation_conflicts([row,second],observation_tolerance=1e-6,action_tolerance=.1)
    assert conflicts==[dict(sample_ids=['s1','s2'],observation_max_abs=0.,action_max_abs=1.4)]


def test_group_balanced_s_sampling():
    rows=[dict(root_id='n1',task_group='new_recovery')]*4+[dict(root_id='n2',task_group='new_recovery')]+[
        dict(root_id='o1',task_group='old_nominal')]*2+[dict(root_id='o2',task_group='old_recovery')]
    np.testing.assert_allclose(api().consolidation_weights(rows),[.0625]*4+[.25]+[.125]*2+[.25])


def test_aggregate_preserves_only_verified_targets_and_rejects_tampering(tmp_path):
    q,teachers,refs=fixture(tmp_path)
    row=api().label_verified_query(q,refs,teachers)
    bank=tmp_path/'bank.json';bank.write_text(json.dumps(dict(schema=api().SCHEMA,teachers=teachers,labels=[row])))
    manifest=api().merge_verified_labels([bank],tmp_path/'merged')
    with np.load(manifest['path']) as data:
        np.testing.assert_allclose(data['actions'],[[.7]*4])
    assert manifest['sample_origins']==['student_visited_query']
    row['target_action'][0]=-.7
    bank.write_text(json.dumps(dict(schema=api().SCHEMA,teachers=teachers,labels=[row])))
    with pytest.raises(ValueError,match='target'):api().merge_verified_labels([bank],tmp_path/'tampered')
