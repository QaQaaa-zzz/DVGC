import pytest
from jit_dvgc.skill_transfer_evaluation import summarize_recovery


def test_summary_preserves_full_root_denominator_and_numeric_repeat_semantics():
    results={'a':{'labels':[1,1,1],'receipts':[]},'b':{'labels':[1,0,1],'receipts':[]}}
    summary=summarize_recovery(results,['a','b','c'],3)
    assert summary['full_train_root_denominator']==3
    assert summary['selected_root_denominator']==2
    assert summary['stable_successes']==1
    assert summary['mixed']==1
    assert summary['unselected_roots']==['c']
    assert summary['repeat_semantics']=='numerical_label_stability_not_independent_training_seeds'


def test_summary_rejects_wrong_repeat_count():
    with pytest.raises(ValueError):summarize_recovery({'a':{'labels':[1]}},['a'],3)


def test_prepare_has_no_prefix_and_caps_original_remaining_time(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    import jit_dvgc.skill_transfer_evaluation as m
    import jit_dvgc.probe_bank as bank_module
    import jit_dvgc.unified_envelope_snapshot as snapshot_module
    policy=dict(name='parent',actor_sha256='actor',normalizer_sha256='norm',xml_sha256='xml')
    parent=tmp_path/'parent.json';parent.write_text(json.dumps(dict(policy=policy)))
    bank=tmp_path/'bank.json';bank.write_text('{}')
    teacher=tmp_path/'teacher.json';teacher.write_text('{}')
    source=tmp_path/'source.json';source.write_text('[]')
    monkeypatch.setattr(bank_module,'load_probe_bank',lambda p:dict(members=[dict(name='parent',policy=policy)],max_candidates_per_process=8,task=dict(success_criterion='stable_forward_recovery')))
    locked=[]
    monkeypatch.setattr(bank_module,'lock_probe_bank',lambda spec,path:locked.append(spec))
    def row(root):return dict(root_id=root,root_episode_id=root,role='TRAIN',snapshot='snapshot',state_sha256='physical',snapshot_context_sha256='context',snapshot_time=.06)
    def entry(root):return dict(root_id=root,actor_sha256='actor',normalizer_sha256='norm',model_sha256='xml')
    monkeypatch.setattr(m,'resolve_teacher_sources',lambda *args:[(entry('a'),row('a')),(entry('b'),row('b'))])
    monkeypatch.setattr(snapshot_module,'load_unified_envelope_snapshot',lambda p:SimpleNamespace(episode_step=3))
    monkeypatch.setattr(snapshot_module,'snapshot_context_sha256',lambda p:'context')
    result=m.prepare_recovery_panel(tmp_path/'out',parent,teacher,source,roots=['a'],bank_path=bank)
    panel=result['panel'];spec=panel['cases'][0]['spec']
    assert spec['horizon']==397 and spec['budget']==397
    assert locked[0]['max_ticks']==397
    assert panel['full_train_root_ids']==['a','b']
    assert panel['selected_root_ids']==['a']
    assert spec['preserve_snapshot_episode_context'] is True
    assert not any(k in spec for k in ('bridge_action_plan','closed_loop_prefix_policy','closed_loop_prefix_initializer'))
    assert panel['preparation_environment_transitions']==0
