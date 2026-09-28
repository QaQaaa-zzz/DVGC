import importlib
from types import SimpleNamespace as NS
import numpy as np
import pytest


def test_empty_demo_real_joint_grad_keep_and_scoped_restore(monkeypatch, tmp_path):
    import jax
    import jax.numpy as jp
    from brax.training.agents.ppo import losses
    m=importlib.import_module('jit_dvgc.generative_bridge.student')
    def base(params,normalizer,data,rng,network,**kw):
        return jp.square(params.policy-3),{'base':jp.square(params.policy-3)}
    monkeypatch.setattr(losses,'compute_ppo_loss',base)
    network=NS(policy_network=NS(apply=lambda norm,p,obs:jp.ones((len(obs['state']),4))*p/norm.scale),
               parametric_action_distribution=NS(mode=lambda x:x))
    normalizer=NS(count=jp.array(0.),scale=jp.array(2.))
    source_normalizer=NS(count=jp.array(0.),scale=jp.array(1.))
    def trainer(**kwargs):
        def loss(p):
            return losses.compute_ppo_loss(NS(policy=p),normalizer,None,jax.random.PRNGKey(1),network)
        (v,metrics),grad=jax.value_and_grad(loss,has_aux=True)(jp.array(2.))
        assert float(metrics['effective_lambda_demo'])==0
        assert float(metrics['retention_action_mse'])==1
        assert float(grad)==pytest.approx(-2.2)
        return v,metrics
    wrapped=m.make_joint_student_trainer(trainer,None,
        retention=(np.ones((3,76),np.float32),np.ones(3)/3),transitions=128000,
        demo_sampler=lambda *args:pytest.fail('empty demo sampler called'))
    wrapped(restore_params=(source_normalizer,jp.array(2.),jp.array(0.)))
    assert losses.compute_ppo_loss is base
    def broken(**kw):raise RuntimeError('trainer failure')
    with pytest.raises(RuntimeError):
        m.make_joint_student_trainer(broken,None,retention=None,keep_coefficient=0)(
            restore_params=(source_normalizer,jp.array(2.),jp.array(0.)))
    assert losses.compute_ppo_loss is base


def test_zero_coefficients_do_not_evaluate_auxiliary_targets(monkeypatch):
    from brax.training.agents.ppo import losses
    m=importlib.import_module('jit_dvgc.generative_bridge.student')
    sentinel=object()
    monkeypatch.setattr(losses,'compute_ppo_loss',sentinel)
    def trainer(**kw):
        assert losses.compute_ppo_loss is sentinel
        return 17
    wrapped=m.make_joint_student_trainer(trainer,None,retention=None,keep_coefficient=0,
                                       demo_coefficient_start=0,demo_coefficient_end=0)
    assert wrapped(restore_params=(None,None,None))==17


def test_joint_config_rejects_nested_wrappers_and_fresh_actor():
    m=importlib.import_module('jit_dvgc.generative_bridge.student')
    for raw in [dict(generative_bridge_student={},action_retention={}),
                dict(generative_bridge_student={},initialization={'actor':'fresh'})]:
        with pytest.raises(ValueError):m.trainer_from_config(None,raw,None)


def test_empty_demo_real_brax_ppo_optimizer_step(tmp_path):
    import jax
    import jax.numpy as jp
    import optax
    from brax.training import types, gradients
    from brax.training.acme import running_statistics as rs
    from brax.training.agents.ppo import losses
    from jit_dvgc.ppo import make_network_factory
    from jit_dvgc.ppo_numerics import guard_ppo_updates
    from jit_dvgc.generative_bridge.student import make_joint_student_trainer
    key=jax.random.PRNGKey(101)
    network=make_network_factory()({'state':76,'privileged_state':106},4,preprocess_observations_fn=rs.normalize)
    actor=network.policy_network.init(key);critic=network.value_network.init(jax.random.fold_in(key,1))
    normalizer=rs.init_state({'state':jp.zeros(76),'privileged_state':jp.zeros(106)})
    obs={'state':jp.ones((2,3,76))*.1,'privileged_state':jp.ones((2,3,106))*.2}
    logits=network.policy_network.apply(normalizer,actor,obs)
    raw=network.parametric_action_distribution.sample_no_postprocessing(logits,key)
    logp=network.parametric_action_distribution.log_prob(logits,raw)
    data=types.Transition(observation=obs,action=jp.tanh(raw),reward=jp.ones((2,3)),
        discount=jp.ones((2,3)),next_observation=obs,
        extras={'state_extras':{'truncation':jp.zeros((2,3))},'policy_extras':{'raw_action':raw,'log_prob':logp,'distribution_params':logits}})
    params=losses.PPONetworkParams(policy=actor,value=critic)
    original=losses.compute_ppo_loss
    def trainer(**kwargs):
        calc=gradients.loss_and_pgrad(lambda p:losses.compute_ppo_loss(p,normalizer,data,key,network),None,has_aux=True)
        optimizer=optax.chain(optax.clip_by_global_norm(.75),optax.adam(3e-5))
        state=optimizer.init(params)
        (value,metrics),grad=calc(params)
        updates,new_state=optimizer.update(grad,state,params)
        trained=optax.apply_updates(params,updates)
        assert np.isfinite(value) and metrics['effective_lambda_demo']==0
        assert any(not np.array_equal(a,b) for a,b in zip(jax.tree.leaves(params),jax.tree.leaves(trained)))
        assert all(np.isfinite(v).all() for v in jax.tree.leaves(new_state))
        return metrics
    with guard_ppo_updates(tmp_path):
        result=make_joint_student_trainer(trainer,None,retention=(np.ones((2,76),np.float32),np.ones(2)/2),
            retention_batch_size=2,demo_sampler=lambda *a,**kw:pytest.fail('empty demo RNG used'))(
            restore_params=(normalizer,actor,critic))
    assert losses.compute_ppo_loss is original
    assert result['demo_samples_per_loss']==0


def test_actual_demo_usage_counts_execution_not_jit_trace(monkeypatch,tmp_path):
    import jax
    import jax.numpy as jp
    from brax.training.agents.ppo import losses
    from jit_dvgc.generative_bridge.data import export_student_demonstrations
    from jit_dvgc.generative_bridge.student import make_joint_student_trainer
    from .test_generative_bridge_contracts import trace
    t=trace(origin='verified_teacher');t['metadata'].update(teacher_status='verified_solution',verification_receipt_sha256='ok')
    t['arrays']['action_origin']=np.array(['bridge_prefix']*16+['source_tail']*2)
    demo=export_student_demonstrations([t],tmp_path/'data');usage={}
    monkeypatch.setattr(losses,'compute_ppo_loss',lambda *a,**kw:(jp.array(1.),{}))
    net=NS(policy_network=NS(apply=lambda norm,p,obs:jp.ones((len(obs['state']),4))*p),
           parametric_action_distribution=NS(mode=lambda x:x))
    normalizer=NS(count=jp.array(0.))
    def trainer(**kw):
        fn=jax.jit(lambda p:losses.compute_ppo_loss(NS(policy=p),normalizer,None,jax.random.PRNGKey(1),net)[0])
        jax.eval_shape(fn,jp.array(.2))
        assert not usage.get('demo_samples_by_root')
        jax.block_until_ready(fn(jp.array(.2)));jax.block_until_ready(fn(jp.array(.3)))
    make_joint_student_trainer(trainer,demo,retention=None,keep_coefficient=0,
        demo_batch_size=4,usage_sink=usage)(restore_params=(normalizer,jp.array(0.),None))
    assert usage['demo_samples_by_root']=={'r':8} and usage['demo_loss_calls']==2
