"""Explicit campaign continuation starts after the last complete outer round."""
import json
import shutil
from pathlib import Path
import pytest
from jit_dvgc.generative_bridge.protocol import atomic_json
from jit_dvgc.generative_bridge.contracts import file_sha
from .test_teacher_recovery_storage import completed_fixture


def later_fixture(tmp_path,monkeypatch,count=12):
    root,bundle=completed_fixture(tmp_path/'first',monkeypatch);bundles=[bundle];done=[]
    for i in range(1,count+1):
        if i>1:
            source,bundle=completed_fixture(tmp_path/f'source{i}',monkeypatch)
            mpath=Path(bundle['learner']['path']);m=json.loads(mpath.read_text());m['lifetime_transitions']=i*32000
            atomic_json(mpath,m);bundle['learner']['sha256']=file_sha(mpath);bundle['round']=i
            atomic_json(source/'round_0001/current_source.json',bundle)
            shutil.copytree(source/'round_0001',root/f'round_{i:04d}');bundles.append(bundle)
        done.append(dict(round=i,learner=bundle['learner'],charged_interactions=38800))
    atomic_json(root/'completed_rounds.json',done[:-1]);atomic_json(root/'current_source.json',bundles[-2])
    atomic_json(root/'status.json',dict(phase='running',completed_rounds=count-1,current_round=count))
    atomic_json(root/'started.json',{})
    monitor=root/'teacher_monitor/G0';monitor.mkdir(parents=True)
    atomic_json(monitor/'status.json',dict(phase='completed',round=0,charged_interactions=17))
    atomic_json(monitor/'report.json',dict(phase='completed',round=0))
    for i in (5,10):
        if i>count:continue
        out=root/f'teacher_monitor/G{i}';out.mkdir()
        atomic_json(out/'status.json',dict(phase='completed',round=i,charged_interactions=23))
        atomic_json(out/'report.json',dict(phase='completed',round=i))
        done[i-1]['teacher_monitor']=str(out/'report.json')
    atomic_json(root/'completed_rounds.json',done[:-1])
    old=json.loads((root/'declaration.json').read_text());old['monitor_initial']=str(monitor/'report.json');atomic_json(root/'declaration.json',old)
    return root,bundles,done


def test_later_boundary_requires_full_lifetime_and_preserves_sequential_history(tmp_path,monkeypatch):
    from jit_dvgc.generative_bridge.recovery_campaign import verify_storage_completed_round
    root,bundles,done=later_fixture(tmp_path,monkeypatch)
    proof=verify_storage_completed_round(root,completed_round=12)
    assert proof['round']==12 and proof['next_round']==13
    assert proof['completed_training_transitions']==384000
    assert proof['learner']['lifetime_transitions']==384000
    assert proof['completed_journal']==done
    assert [m['round'] for m in proof['teacher_monitors']]==[0,5,10]
    assert proof['campaign_charged_interactions']==12*38800+17+23*2
    (root/'round_0013').mkdir()
    with pytest.raises(ValueError,match='successor'):verify_storage_completed_round(root,completed_round=12)


def test_missing_history_or_incomplete_child_cannot_be_adopted(tmp_path,monkeypatch):
    from jit_dvgc.generative_bridge.recovery_campaign import verify_storage_completed_round
    root,_,done=later_fixture(tmp_path,monkeypatch)
    atomic_json(root/'completed_rounds.json',done[:9]+done[10:11])
    with pytest.raises(ValueError,match='sequential'):verify_storage_completed_round(root,completed_round=12)
    atomic_json(root/'completed_rounds.json',done[:-1])
    atomic_json(root/'round_0012/status.json',dict(phase='running',training_transitions=32000))
    with pytest.raises(ValueError,match='completed'):verify_storage_completed_round(root,completed_round=12)


def test_completed_resume_archives_parent_and_starts13_without_repeat(tmp_path,monkeypatch):
    from jit_dvgc.generative_bridge.recovery_campaign import prepare_completed_resume,campaign_start_state,monitoring_sources
    import jit_dvgc.generative_bridge.production as production
    import jit_dvgc.generative_bridge.recovery_campaign as campaign
    root,bundles,done=later_fixture(tmp_path,monkeypatch)
    old={name:(root/name).read_bytes() for name in ('declaration.json','current_source.json','completed_rounds.json','status.json')}
    monkeypatch.setattr(production,'implementation_identity',lambda repo:'newcommit')
    monkeypatch.setattr(campaign,'record_round_metrics_once',lambda *args:None)
    engineering=root/'engineering.json';atomic_json(engineering,dict(phase='passed',measured_new_charged_control_steps=19,new_optimizer_updates=0))
    d=prepare_completed_resume(root,tmp_path/'repo',completed_round=12,engineering_receipt=engineering)
    assert (root/'declaration.json').read_bytes()==old['declaration.json']
    transition=root/'storage_transition/round_0012'
    for name,data in old.items():assert (transition/'parent'/name).read_bytes()==data
    assert d['active_authorization_path']==str(root/'declaration_round_0012_resume.json')
    source,journal,next_round=campaign_start_state(d,root)
    assert source==bundles[-1] and journal==done and next_round==13
    assert d['storage_continuation']['engineering_checks'][0]['charged_interactions']==19
    assert [s['name'] for s in monitoring_sources(root,start_round=next_round)][:2]==['round13_loss','round13_reward']
    assert not (root/'round_0013').exists()
    bad=dict(d);bad['storage_continuation']=dict(d['storage_continuation'],next_round=12)
    with pytest.raises(ValueError,match='cursor'):campaign_start_state(bad,root)


def test_engineering_receipt_limits_physics_and_forbids_updates(tmp_path,monkeypatch):
    from jit_dvgc.generative_bridge.recovery_campaign import prepare_completed_resume
    import jit_dvgc.generative_bridge.production as production
    root,_,_=later_fixture(tmp_path,monkeypatch);receipt=root/'engineering.json'
    monkeypatch.setattr(production,'implementation_identity',lambda repo:'newcommit')
    atomic_json(receipt,dict(phase='failed',measured_new_charged_control_steps=50001,new_optimizer_updates=0))
    with pytest.raises(ValueError,match='engineering'):prepare_completed_resume(root,tmp_path/'repo',completed_round=12,engineering_receipt=receipt)
    assert not (root/'declaration_round_0012_resume.json').exists()
    atomic_json(receipt,dict(phase='passed',measured_new_charged_control_steps=1,new_optimizer_updates=1))
    with pytest.raises(ValueError,match='engineering'):prepare_completed_resume(root,tmp_path/'repo',completed_round=12,engineering_receipt=receipt)


def test_round13_uses_saved12_learner_and_successor_authorization(tmp_path,monkeypatch):
    from jit_dvgc.generative_bridge.recovery_campaign import prepare_completed_resume,round_spec,campaign_parameters
    import jit_dvgc.generative_bridge.production as production
    import jit_dvgc.generative_bridge.recovery_campaign as campaign
    import jit_dvgc.probe_bank as probe_bank
    root,bundles,_=later_fixture(tmp_path,monkeypatch);bundle=bundles[-1]
    bank=root/'source_bank.json';atomic_json(bank,dict(task='stable_forward_recovery',max_ticks=400,
        label_interaction_budget=400,max_candidates_per_process=17))
    baseline=root/'pi0.json';atomic_json(baseline,{'policy':dict(actor_sha256='pi0actor',normalizer_sha256='pi0stats',xml_sha256='xml')})
    dev=root/'dev.json';atomic_json(dev,[])
    old=json.loads((root/'declaration.json').read_text());old.update(parameters=campaign_parameters(),
        source_runtime=dict(bank=str(bank),reward_mode='jump_ori'),baseline_frozen_policy=str(baseline),
        retention_ref={},complete_start={},max_wall_seconds=123)
    atomic_json(root/'declaration.json',old)
    bundle['actor']['policy']['name']='student12';bundle.update(dev_fixture=str(dev),dev_fixture_sha256=file_sha(dev),
        generator_lifetime_charged_updates=196000)
    atomic_json(root/'round_0012/current_source.json',bundle)
    monkeypatch.setattr(production,'implementation_identity',lambda repo:'newcommit')
    monkeypatch.setattr(production,'implementation_files',lambda repo:{})
    monkeypatch.setattr(campaign,'record_round_metrics_once',lambda *args:None)
    monkeypatch.setattr(probe_bank,'lock_probe_bank',lambda value,path:atomic_json(path,value))
    d=prepare_completed_resume(root,tmp_path/'repo',completed_round=12);spec=round_spec(d,bundle,13)
    assert spec['teacher_recovery_authorization']==dict(path=d['active_authorization_path'],sha256=file_sha(d['active_authorization_path']))
    assert spec['continuation']['learner']==bundle['learner']
    assert spec['continuous_learning'] and not spec['allow_legacy_optimizer_bootstrap']
    assert spec['seed']==old['seed']+13 and spec['round_index']==13
    assert spec['recovery_training_contract']==old['parameters']['student']
