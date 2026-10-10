"""CPU numerical fixtures; no physical or learning-performance claims."""
from copy import deepcopy
from types import SimpleNamespace as NS
import numpy as np
import pytest
from .test_generative_bridge_contracts import trace
from jit_dvgc.generative_bridge.contracts import file_sha


def teacher(root='r',n=18):
    t=trace(origin='verified_teacher',n=n)
    t['metadata'].update(root_id=root,teacher_status='verified_solution',verification_receipt_sha256='ok')
    t['arrays']['action_origin']=np.array(['bridge_prefix']*min(n,16)+['source_tail']*max(n-16,0))
    return t


def test_cumulative_bank_root_segment_group_weights_and_identity(tmp_path):
    from jit_dvgc.generative_bridge.student_demo_bank import build_student_demo_bank
    from jit_dvgc.generative_bridge.student import load_optional_demo
    identity=dict(actor_sha256='old',normalizer_sha256='norm',model_sha256='model',protocol_sha256='protocol')
    first=build_student_demo_bank([teacher('a',18),teacher('b',20)],tmp_path/'first',source_identity=identity,round_id=0)
    path=tmp_path/'first/manifest.json';previous=dict(path=str(path),sha256=file_sha(path))
    second=build_student_demo_bank([teacher('c',24)],tmp_path/'second',source_identity=identity,previous=previous,round_id=1)
    _,_,weights=load_optional_demo(second);r=np.array(second['sample_roots']);s=np.array(second['sample_origins'])
    assert set(r)=={'a','b','c'}
    assert weights[r=='a'].sum()==pytest.approx(.25)
    assert weights[r=='c'].sum()==pytest.approx(.5)
    assert weights[(r=='c')&(s=='source_tail')].sum()==pytest.approx(.25)
    duplicate=build_student_demo_bank([teacher('a')],tmp_path/'duplicate',source_identity=identity,previous=previous,round_id=2)
    assert duplicate['count']==first['count'] and duplicate['new_roots']==[]
    bad=teacher('a');bad['arrays']['normalized_action_executed'][0,0]=.1
    with pytest.raises(ValueError,match='conflicting'):build_student_demo_bank([bad],tmp_path/'bad',source_identity=identity,previous=previous,round_id=2)
    with pytest.raises(ValueError,match='identity'):build_student_demo_bank([],tmp_path/'drift',source_identity={**identity,'actor_sha256':'other'},previous=previous,round_id=2)


def test_bank_rejects_dev_and_empty(tmp_path):
    from jit_dvgc.generative_bridge.student_demo_bank import build_student_demo_bank
    from jit_dvgc.generative_bridge.student import load_optional_demo
    identity=dict(actor_sha256='old',normalizer_sha256='norm',model_sha256='model',protocol_sha256='protocol')
    empty=build_student_demo_bank([],tmp_path/'empty',source_identity=identity,round_id=0)
    assert load_optional_demo(empty) is None
    t=teacher();t['metadata']['role']='student_dev'
    with pytest.raises(ValueError,match='TRAIN'):build_student_demo_bank([t],tmp_path/'dev',source_identity=identity,round_id=0)


def test_explicit_retention_reference_is_not_warm_initializer(monkeypatch):
    import jax
    import jax.numpy as jp
    from brax.training.agents.ppo import losses
    from jit_dvgc.generative_bridge.student import make_joint_student_trainer
    monkeypatch.setattr(losses,'compute_ppo_loss',lambda *a,**kw:(jp.array(0.),{}))
    net=NS(policy_network=NS(apply=lambda norm,p,obs:jp.ones((len(obs['state']),4))*p),parametric_action_distribution=NS(mode=lambda x:x))
    norm={'count':jp.array(0.)}
    # Use actual JAX tree normalizer rather than an opaque test namespace.
    from brax.training.acme import running_statistics as rs
    normalizer=rs.init_state(jp.zeros(76))
    def trainer(**kwargs):
        (_,m),grad=jax.value_and_grad(lambda p:losses.compute_ppo_loss(NS(policy=p),normalizer,None,jax.random.PRNGKey(1),net),has_aux=True)(jp.array(.5))
        assert float(m['retention_action_mse'])==pytest.approx(.25)
        assert float(grad)==pytest.approx(.2)
    make_joint_student_trainer(trainer,None,retention=(np.ones((2,76)),np.ones(2)),retention_reference=(normalizer,jp.array(0.)))(restore_params=(normalizer,jp.array(.5),None))


def test_warmup_actor_only_frozen_normalizer_and_bounded(tmp_path):
    import jax
    import jax.numpy as jp
    from brax.training.acme import running_statistics as rs
    from jit_dvgc.generative_bridge.data import export_student_demonstrations
    from jit_dvgc.generative_bridge.warmup import warmup_actor,select_warmup_checkpoint
    demo=export_student_demonstrations([teacher()],tmp_path/'demo')
    normalizer=rs.init_state(jp.zeros(76));actor=jp.array(.5);critic=jp.array(7.)
    net=NS(policy_network=NS(apply=lambda norm,p,obs:jp.ones((len(obs['state']),4))*p),parametric_action_distribution=NS(mode=lambda x:x))
    checkpoints=[]
    result=warmup_actor(net,(normalizer,actor,critic),(normalizer,jp.array(0.)),demo,(np.ones((2,76)),np.ones(2)),tmp_path/'warm',updates=2,batch_size=2,checkpoint_callback=lambda step,p:checkpoints.append(step))
    assert result[0] is normalizer and result[2] is critic
    assert float(result[1])<float(actor) and checkpoints==[0,2]
    full=warmup_actor(net,(normalizer,actor,critic),(normalizer,jp.array(0.)),demo,(np.ones((2,76)),np.ones(2)),tmp_path/'full',updates=2,batch_size=2,full_learner=True,actual_gradient_audit=True)
    import pickle
    with (tmp_path/'full/learner_update_0002.pkl').open('rb') as f:saved=pickle.load(f)
    assert saved['completed_supervised_updates']==2 and 'optimizer_state' in saved and 'rng' in saved
    assert select_warmup_checkpoint({0:3,500:3,1000:2})==0
    with pytest.raises(ValueError,match='budget'):warmup_actor(net,(normalizer,actor,critic),(normalizer,actor),demo,None,tmp_path/'bad',updates=2001)


def test_retention_full_trace_loader(tmp_path):
    from jit_dvgc.generative_bridge.student import load_retention_traces
    path=tmp_path/'retention.npz';np.savez(path,actor_observation_before=np.ones((80,76)))
    ref=dict(path=str(path),sha256=file_sha(path),role='train',full_success=True)
    assert load_retention_traces(ref)[0].shape==(80,76)
    with pytest.raises(ValueError,match='TRAIN'):load_retention_traces({**ref,'role':'student_dev'})


def test_actual_installed_brax_fixed_gradient_audit():
    import jax
    import jax.numpy as jp
    from brax.training import types
    from brax.training.acme import running_statistics as rs
    from brax.training.agents.ppo import losses
    from jit_dvgc.ppo import make_network_factory
    from jit_dvgc.generative_bridge.learning_audit import gradient_audit
    key=jax.random.PRNGKey(19)
    net=make_network_factory()({'state':76,'privileged_state':106},4,preprocess_observations_fn=rs.normalize)
    actor=net.policy_network.init(key);critic=net.value_network.init(jax.random.fold_in(key,1))
    norm=rs.init_state({'state':jp.zeros(76),'privileged_state':jp.zeros(106)})
    obs={'state':jp.ones((2,3,76))*.1,'privileged_state':jp.ones((2,3,106))*.2}
    logits=net.policy_network.apply(norm,actor,obs)
    raw=net.parametric_action_distribution.sample_no_postprocessing(logits,key)
    logp=net.parametric_action_distribution.log_prob(logits,raw)
    data=types.Transition(observation=obs,action=jp.tanh(raw),reward=jp.ones((2,3)),discount=jp.ones((2,3)),next_observation=obs,
        extras={'state_extras':{'truncation':jp.zeros((2,3))},'policy_extras':{'raw_action':raw,'log_prob':logp,'distribution_params':logits}})
    def demo(p):return jp.mean(jp.square(net.parametric_action_distribution.mode(net.policy_network.apply(norm,p,obs))-.2))
    report=gradient_audit(net,losses.PPONetworkParams(policy=actor,value=critic),norm,data,key,demo_loss=demo,keep_loss=demo)
    assert report['installed_kl_mean']==pytest.approx(report['installed_self_kl'],abs=1e-7)
    assert report['installed_self_kl']>0  # Installed Brax adds 1e-5 inside the log.
    assert report['behavior_logprob_max_abs_difference']==0
    assert report['actor_gradient_norms']['demo']>0
    assert report['actor_gradient_cosines']['demo/keep']==pytest.approx(1.,abs=1e-6)
    assert report['critic_value_gradient_norm']>0
    from jit_dvgc.generative_bridge.student import make_joint_student_trainer
    original_loss=losses.compute_ppo_loss
    def trainer(**kw):
        (value,metrics),grads=jax.value_and_grad(lambda p:losses.compute_ppo_loss(p,norm,data,key,net),has_aux=True)(losses.PPONetworkParams(policy=actor,value=critic))
        assert np.isfinite(value)
        assert all(np.isfinite(v).all() for v in jax.tree.leaves(grads))
        assert 'audit/actor_gradient_norm/ppo_policy' in metrics
        assert metrics['audit/behavior_logprob_max_abs_difference']==0
        assert float(metrics['audit/critic_value_gradient_norm'])==pytest.approx(report['critic_value_gradient_norm'],rel=1e-5)
        assert metrics['audit/critic_explained_variance_defined']==1
        assert float(metrics['audit/critic_explained_variance'])==pytest.approx(report['critic_explained_variance'],abs=1e-6)
        return metrics
    metrics=make_joint_student_trainer(trainer,None,retention=(np.ones((2,76),np.float32),np.ones(2)),retention_reference=(norm,actor),audit_enabled=True,retention_batch_size=2)(restore_params=(norm,actor,critic))
    assert losses.compute_ppo_loss is original_loss
    assert metrics['fixed_probe/demo_mse']==0


def test_warmup_inference_override_preserves_source_provenance(tmp_path):
    import pickle
    from jit_dvgc.checkpoint import CheckpointPayload,CheckpointIdentity
    from jit_dvgc.handoff_bank import pytree_sha256
    from jit_dvgc.generative_bridge.learning_audit import warmup_inference_override
    norm={'state':np.ones(76)};actor={'w':np.zeros((76,4))};critic={'v':np.ones(2)}
    payload=CheckpointPayload(CheckpointIdentity('config','xml',(),(),()),4988928,norm,actor,critic)
    policy=dict(actor_sha256=pytree_sha256(actor),normalizer_sha256=pytree_sha256(norm),checkpoint='original')
    path=tmp_path/'warm.pkl'
    candidate_actor={'w':np.ones((76,4))*.01}
    with path.open('wb') as stream:pickle.dump((norm,candidate_actor,{'ignored':0}),stream)
    ref=dict(path=str(path),sha256=file_sha(path),source_actor_sha256=policy['actor_sha256'],normalizer_sha256=policy['normalizer_sha256'])
    candidate,metadata=warmup_inference_override(payload,policy,ref)
    assert candidate.identity is payload.identity and candidate.critic_params is critic
    assert candidate.training_transitions==4988928
    assert metadata['checkpoint']=='original' and metadata['optimizer_restored'] is False
    assert metadata['actor_sha256']!=policy['actor_sha256'] and metadata['adopted'] is False
    assert np.all(payload.actor_params['w']==0)
    with pytest.raises(ValueError,match='source'):warmup_inference_override(payload,policy,{**ref,'source_actor_sha256':'wrong'})


def test_worker_tensorboard_does_not_precreate_exclusive_warmup_output(monkeypatch,tmp_path):
    import json
    import sys
    import jax
    from jit_dvgc.generative_bridge import worker,student,warmup
    from jit_dvgc import ppo
    output=tmp_path/'warm'
    events=[]
    class Writer:
        def __init__(self,path):
            assert output.exists()
            from pathlib import Path
            Path(path).mkdir(parents=True)
            events.append('writer')
        def add_scalar(self,*args):pass
        def flush(self):pass
        def close(self):events.append('closed')
    monkeypatch.setitem(sys.modules,'torch.utils.tensorboard',NS(SummaryWriter=Writer))
    monkeypatch.setattr(jax,'default_backend',lambda:'gpu')
    policy={'actor_sha256':'actor','normalizer_sha256':'norm'}
    norm=NS(mean={'state':np.zeros(76),'privileged_state':np.zeros(106)})
    payload=NS(observation_normalizer=norm,actor_params=np.array(1.),critic_params=np.array(2.))
    monkeypatch.setattr(worker,'source_payload',lambda path:(policy,payload))
    monkeypatch.setattr(student,'load_retention_reference',lambda ref:((norm,payload.actor_params),policy))
    monkeypatch.setattr(student,'load_retention_traces',lambda ref:(np.zeros((2,76)),np.ones(2)/2))
    monkeypatch.setattr(ppo,'make_network_factory',lambda:lambda *args,**kwargs:None)
    def fake_warmup(*args,**kwargs):
        assert not output.exists()
        output.mkdir(exist_ok=False)
        events.append('warmup')
        kwargs['metrics_callback']({'update':1,'loss':.2})
        (output/'warmup_status.json').write_text(json.dumps({'status':'completed','completed_updates':1,'checkpoints':[]}))
    monkeypatch.setattr(warmup,'warmup_actor',fake_warmup)
    demo=tmp_path/'demo.json';demo.write_text('{}')
    result=worker.run_warmup(dict(source_frozen_policy='source',retention_reference_actor={},
        demo_manifest={'path':str(demo),'sha256':file_sha(demo)},retention_trace_observations={},
        output=str(output),result=str(tmp_path/'result.json'),seed=1,updates=1))
    assert events==['warmup','writer','closed'] and result['status']=='completed'


def test_fixed_weighted_telemetry_probe_bounded_and_stratified():
    from jit_dvgc.generative_bridge.learning_audit import fixed_weighted_probe_indices
    weights=np.concatenate([np.ones(100)*.5/100,np.ones(49900)*.5/49900])
    ids=fixed_weighted_probe_indices(weights)
    assert len(ids)==256 and np.sum(ids<100)==128
    assert np.array_equal(ids,fixed_weighted_probe_indices(weights))


def test_bank_rejects_normalizer_drift(tmp_path):
    from jit_dvgc.generative_bridge.student_demo_bank import build_student_demo_bank
    t=teacher();t['metadata']['normalizer_sha256']='other'
    identity=dict(actor_sha256='old',normalizer_sha256='norm',model_sha256='model',protocol_sha256='protocol')
    with pytest.raises(ValueError,match='normalizer'):build_student_demo_bank([t],tmp_path/'badnorm',source_identity=identity,round_id=0)


def test_first_loss_declared_kl_gate_preserves_evidence_and_restores_loss(monkeypatch,tmp_path):
    import json
    import jax
    import jax.numpy as jp
    from brax.training.agents.ppo import losses
    from jit_dvgc.generative_bridge import learning_audit
    from jit_dvgc.generative_bridge.student import make_joint_student_trainer
    base=lambda *a,**kw:(jp.array(0.),{})
    monkeypatch.setattr(losses,'compute_ppo_loss',base)
    monkeypatch.setattr(learning_audit,'critic_loss_metrics',lambda *a,**kw:{})
    monkeypatch.setattr(learning_audit,'joint_loss_metrics',lambda *a,**kw:{
        'audit/installed_behavior_kl':jp.array(.8),'audit/installed_self_kl':jp.array(.00004),
        'audit/behavior_logprob_max_abs_difference':jp.array(2.)})
    net=NS(policy_network=NS(apply=lambda norm,p,obs:jp.ones((len(obs['state']),4))*p),parametric_action_distribution=NS(mode=lambda x:x))
    normalizer=NS(count=jp.array(0.))
    def trainer(**kw):return losses.compute_ppo_loss(NS(policy=jp.array(0.),value=jp.array(0.)),normalizer,None,jax.random.PRNGKey(0),net)
    path=tmp_path/'first_loss.json'
    wrapped=make_joint_student_trainer(trainer,None,retention=(np.ones((2,76)),np.ones(2)),
        audit_enabled=True,max_first_behavior_kl=.05,first_update_audit_path=path)
    with pytest.raises(Exception,match='first-loss behavior KL'):
        wrapped(restore_params=(normalizer,jp.array(0.),None))
    report=json.loads(path.read_text())
    assert report['status']=='diagnostic_stop' and report['maximum_excess_behavior_kl']==.05
    assert report['metrics']['audit/behavior_logprob_max_abs_difference']==2
    assert losses.compute_ppo_loss is base


def test_empty_demo_action_probe_preserves_normalizer_drift():
    import jax.numpy as jp
    from brax.training.acme import running_statistics as rs
    from jit_dvgc.generative_bridge.learning_audit import action_probe
    norm=rs.init_state({'state':jp.zeros(76),'privileged_state':jp.zeros(106)})
    changed=norm.replace(mean={**norm.mean,'state':jp.ones(76)*.25})
    result=action_probe(None,changed,jp.array(0.),None,reference=(norm,jp.array(0.)))
    assert result['groups']=={} and result['count']==0
    assert result['normalizer_mean_change']==[.25]*76
    assert result['normalizer_sha256']!=result['reference_normalizer_sha256']
