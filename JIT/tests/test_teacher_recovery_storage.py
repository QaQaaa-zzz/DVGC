"""Storage handoff continues a serialized complete round without another update."""
import io
import json
import pickle
from pathlib import Path
import numpy as np
import pytest
from jit_dvgc.generative_bridge.protocol import atomic_json
from jit_dvgc.generative_bridge.contracts import file_sha


def completed_fixture(tmp_path,monkeypatch):
    import jax.numpy as jp
    import optax,torch
    from flax.serialization import msgpack_serialize
    from brax.training.agents.ppo.train import TrainingState
    from brax.training.agents.ppo.losses import PPONetworkParams
    from brax.training import types
    from jit_dvgc.handoff_bank import pytree_sha256
    import jit_dvgc.generative_bridge.worker as worker
    root=tmp_path/'physical';rd=root/'round_0001';rd.mkdir(parents=True)
    params=PPONetworkParams(policy={'w':jp.asarray([1.])},value={'w':jp.asarray([2.])})
    optimizer=optax.adam(3e-5);optstate=optimizer.init(params)
    _,optstate=optimizer.update(__import__('jax').tree.map(jp.ones_like,params),optstate,params)
    norm={'mean':np.zeros(2,np.float32)};key=np.array([3,4],np.uint32)
    state=TrainingState(params=params,optimizer_state=optstate,normalizer_params=norm,env_steps=types.UInt64(hi=0,lo=32000))
    learner=rd/'learner';learner.mkdir();payload=learner/'state.pkl'
    with payload.open('wb') as f:pickle.dump(dict(training_state=state,rng=key),f)
    policy=dict(actor_sha256=pytree_sha256(params.policy),critic_sha256=pytree_sha256(params.value),normalizer_sha256=pytree_sha256(norm))
    manifest=dict(schema='jit_bridge_full_ppo_learner_v1',state_path=str(payload),state_sha256=file_sha(payload),
        local_transitions=32000,lifetime_transitions=32000,contract_sha256='contract',trainer_sha256='trainer',
        **policy,optimizer_sha256=pytree_sha256(optstate),rng_sha256=pytree_sha256(key))
    atomic_json(learner/'manifest.json',manifest)
    frozen=rd/'frozen.json';atomic_json(frozen,{'policy':policy})
    actor=dict(frozen_policy=str(frozen),policy=policy,actor_sha256=policy['actor_sha256'],normalizer_sha256=policy['normalizer_sha256'])
    monkeypatch.setattr(worker,'source_payload',lambda path:(policy,None))
    esaved=dict(policy={'w':torch.tensor([1.])},optimizer={'state':{0:dict(step=torch.tensor(1.),exp_avg=torch.zeros(1),exp_avg_sq=torch.ones(1))},
        'param_groups':[dict(lr=1e-4,params=[0])]},rng=torch.get_rng_state(),learning_rate=1e-4)
    stream=io.BytesIO();torch.save(esaved,stream)
    ep=rd/'E.msgpack';ep.write_bytes(msgpack_serialize(dict(backend='rsl_rl_3.2.0',params={'w':np.ones(1)},
        normalizer_mean=np.zeros(106),normalizer_std=np.ones(106),rng=np.array([5,6],np.uint32),
        torch_state=stream.getvalue(),total_updates=1)))
    gp=rd/'G';gp.mkdir();gstate=dict(params={'w':np.ones(1)},ema={'w':np.ones(1)},optimizer={'count':np.array(1)},
        rng=np.array([7,8],np.uint32),normalizer=dict(mean=np.zeros(76),std=np.ones(76)),updates=np.array(2000))
    (gp/'state.msgpack').write_bytes(msgpack_serialize(gstate));atomic_json(gp/'manifest.json',dict(state_sha256=file_sha(gp/'state.msgpack'),updates=2000,inference_parameters='ema'))
    bundle=dict(round=1,actor=actor,learner=dict(path=str(learner/'manifest.json'),sha256=file_sha(learner/'manifest.json')),
        explorer=str(ep),explorer_sha256=file_sha(ep),generator=dict(checkpoint_manifest=str(gp/'manifest.json'),
            checkpoint_manifest_sha256=file_sha(gp/'manifest.json'),state_updates=2000,inference_parameters='ema'),stress={})
    atomic_json(rd/'current_source.json',bundle)
    costs=[dict(stage='teacher_worker_prior_attempt',phase='failed',charged_interactions=6800),
        dict(stage='student',phase='completed',charged_interactions=32000)]
    atomic_json(rd/'costs.json',costs);atomic_json(rd/'status.json',dict(phase='completed',training_transitions=32000,charged_interactions=38800))
    (root/'monitoring').mkdir()
    atomic_json(root/'initial_bundle.json',dict(generator_lifetime_charged_updates=172000))
    atomic_json(root/'declaration.json',dict(schema='jit_teacher_recovery_campaign_authorization_v1',campaign_output='/old/home',repository='/old/repo',
        implementation_commit='old',rounds=100,seed=8113001,tensorboard_port=6039,locks={},parameters={'physics_cap_per_round':3500000}))
    return root,bundle


def test_full_completed_boundary_checks_adam_rng_E_G_and_retains_failed_fees(tmp_path,monkeypatch):
    from jit_dvgc.generative_bridge.recovery_campaign import verify_storage_completed_round
    root,bundle=completed_fixture(tmp_path,monkeypatch)
    proof=verify_storage_completed_round(root)
    assert proof['completed_training_transitions']==32000
    assert proof['next_round']==2 and proof['charged_interactions']==38800
    assert proof['learner']['optimizer_updates']==1
    assert proof['learner']['rng_sha256'] and proof['explorer']['optimizer_sha256'] and proof['generator']['rng_sha256']
    assert proof['simulator_frame_continuity'] is False
    p=Path(bundle['learner']['path']);m=json.loads(p.read_text());m['rng_sha256']='wrong';atomic_json(p,m)
    bundle['learner']['sha256']=file_sha(p);atomic_json(root/'round_0001/current_source.json',bundle)
    with pytest.raises(ValueError,match='RNG'):verify_storage_completed_round(root)


def test_incomplete_round_cannot_prepare_storage_resume(tmp_path):
    from jit_dvgc.generative_bridge.recovery_campaign import verify_storage_completed_round
    root=tmp_path/'physical';(root/'round_0001').mkdir(parents=True)
    atomic_json(root/'round_0001/status.json',dict(phase='running',training_transitions=0))
    with pytest.raises(ValueError,match='completed'):verify_storage_completed_round(root)


def test_storage_successor_preserves_original_declaration_and_cursor(tmp_path,monkeypatch):
    from jit_dvgc.generative_bridge.recovery_campaign import prepare_storage_resume,campaign_start_state
    import jit_dvgc.generative_bridge.production as production
    import jit_dvgc.generative_bridge.recovery_campaign as campaign
    root,bundle=completed_fixture(tmp_path,monkeypatch);original=(root/'declaration.json').read_bytes()
    monkeypatch.setattr(production,'implementation_identity',lambda repo:'newcommit')
    monkeypatch.setattr(campaign,'record_round_metrics_once',lambda *a:None)
    d=prepare_storage_resume(root,tmp_path/'newrepo')
    assert (root/'declaration.json').read_bytes()==original
    assert d['active_authorization_path']==str(root/'declaration_storage.json')
    assert d['repository']==str((tmp_path/'newrepo').resolve()) and d['implementation_commit']=='newcommit'
    assert d['seed']==8113001 and d['rounds']==100
    source,done,next_round=campaign_start_state(d,root)
    assert source==bundle and len(done)==1 and next_round==2
    assert (root/'round_0001/current_source.json').read_bytes()==json.dumps(bundle,sort_keys=True,indent=2).encode()+b'\n'
    assert not (root/'round_0002').exists()
    with pytest.raises(ValueError,match='already'):prepare_storage_resume(root,tmp_path/'newrepo')


def test_storage_export_manifest_appends_to_existing_event_run_paths(tmp_path):
    from jit_dvgc.generative_bridge.recovery_campaign import monitoring_sources
    sources=monitoring_sources(tmp_path,start_round=2,existing_event_runs=True)
    assert len(sources)==99*2+1
    assert sources[0]['step_offset']==32000
    assert sources[0]['run']==str(tmp_path/'monitoring/events/student_loss')
    assert '/round_0001/' not in sources[0]['path']


def test_round2_binds_successor_authorization_and_completed_learner(tmp_path,monkeypatch):
    from jit_dvgc.generative_bridge.recovery_campaign import prepare_storage_resume,round_spec,campaign_parameters
    import jit_dvgc.generative_bridge.production as production
    import jit_dvgc.generative_bridge.recovery_campaign as campaign
    import jit_dvgc.probe_bank as probe_bank
    root,bundle=completed_fixture(tmp_path,monkeypatch)
    bank=root/'source_bank.json';atomic_json(bank,dict(task='stable_forward_recovery',max_ticks=400,
        label_interaction_budget=400,max_candidates_per_process=17))
    baseline=root/'pi0.json';atomic_json(baseline,{'policy':dict(actor_sha256='pi0actor',normalizer_sha256='pi0stats',xml_sha256='xml')})
    dev=root/'dev.json';atomic_json(dev,[])
    old=json.loads((root/'declaration.json').read_text());old.update(parameters=campaign_parameters(),
        source_runtime=dict(bank=str(bank),reward_mode='jump_ori'),baseline_frozen_policy=str(baseline),
        retention_ref={},complete_start={},max_wall_seconds=123)
    atomic_json(root/'declaration.json',old)
    bundle['actor']['policy']['name']='student';bundle.update(dev_fixture=str(dev),dev_fixture_sha256=file_sha(dev),
        generator_lifetime_charged_updates=174000)
    atomic_json(root/'round_0001/current_source.json',bundle)
    monkeypatch.setattr(production,'implementation_identity',lambda repo:'newcommit')
    monkeypatch.setattr(production,'implementation_files',lambda repo:{})
    monkeypatch.setattr(campaign,'record_round_metrics_once',lambda *a:None)
    monkeypatch.setattr(probe_bank,'lock_probe_bank',lambda bank,path:atomic_json(path,bank))
    d=prepare_storage_resume(root,tmp_path/'newrepo');spec=round_spec(d,bundle,2)
    assert spec['teacher_recovery_authorization']==dict(path=d['active_authorization_path'],sha256=file_sha(d['active_authorization_path']))
    assert spec['continuation']['learner']==bundle['learner']
    assert spec['continuous_learning'] and not spec['allow_legacy_optimizer_bootstrap']
    assert spec['seed']==old['seed']+2 and spec['round_index']==2
    assert spec['recovery_training_contract']==old['parameters']['student']
