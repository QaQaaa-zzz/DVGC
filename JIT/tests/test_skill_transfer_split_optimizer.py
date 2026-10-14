"""CPU tests for split Adam independence, atomic rejection and exact resume."""
import pickle
import jax
import jax.numpy as jp
import numpy as np
import optax
import pytest
from brax.training.agents.ppo.losses import PPONetworkParams as PPOParams
from jit_dvgc.generative_bridge.split_optimizer import (
    split_optimizer, instrument_split_optimizer, PINNED_BRAX_TRAIN_SHA256,
)
from jit_dvgc.generative_bridge.actual_update import adam_update_counts


def params():
    return PPOParams(policy={'w':jp.array([1.,2.])},value={'w':jp.array([3.,4.])})


def equal(a,b):
    assert jax.tree.structure(a)==jax.tree.structure(b)
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
        np.testing.assert_array_equal(x,y)


def test_critic_scale_does_not_shrink_actor():
    p=params(); tx=split_optimizer(); state=tx.init(p)
    g=p.replace(policy={'w':jp.array([.2,.4])},value={'w':jp.array([.1,.3])})
    a,sa=tx.update(g,state,p)
    b,sb=tx.update(g.replace(value=jax.tree.map(lambda x:1000*x,g.value)),state,p)
    equal(a.policy,b.policy);equal(sa.actor,sb.actor)
    assert {k:int(v) for k,v in adam_update_counts(sa).items()}==dict(actor_adam_count=1,critic_adam_count=1)


@pytest.mark.parametrize('branch', ['policy','value'])
def test_nonfinite_gradient_rejects_both_branches(branch):
    p=params();tx=split_optimizer();state=tx.init(p)
    g=p.replace(**{branch:{'w':jp.array([jp.nan,0.])}})
    u,new=jax.jit(tx.update)(g,state,p)
    equal(state,new);equal(p,optax.apply_updates(p,u))


def test_overflowed_proposal_rejects_both_branches():
    p=params();tx=split_optimizer(actor_learning_rate=1e100);state=tx.init(p)
    u,new=tx.update(p,state,p)
    equal(state,new);equal(p,optax.apply_updates(p,u))


def test_pickled_resume_next_step_exact():
    p=params();tx=split_optimizer();state=tx.init(p)
    u,state=tx.update(p,state,p);p=optax.apply_updates(p,u)
    loaded_p,loaded_state=jax.device_put(pickle.loads(pickle.dumps(jax.device_get((p,state)))))
    equal(tx.update(p,state,p),tx.update(p,loaded_state,loaded_p))


def test_locked_instrumentation_and_unknown_source_rejected():
    from brax.training.agents.ppo import train
    wrapped=instrument_split_optimizer(train.train)
    source=wrapped._jit_source_text
    assert source.count('optimizer = _recovery_split_optimizer()')==1
    assert 'optimizer.update(grads, optimizer_state, params, batch=data)' in source
    with pytest.raises(ValueError,match='source SHA'):
        instrument_split_optimizer(train.train,expected_sha256='0'*64)


def test_outer_guard_does_not_guard_each_branch(tmp_path):
    from jit_dvgc.ppo_numerics import guard_ppo_updates
    with guard_ppo_updates(tmp_path):
        tx=split_optimizer()
        p=params();state=tx.init(p)
        bad=p.replace(value={'w':jp.array([jp.inf,0.])})
        u,new=tx.update(bad,state,p)
        equal(state,new);equal(p,optax.apply_updates(p,u))


def test_full_learner_resume_restores_two_clocks_rng_and_next_update(tmp_path):
    from brax.training.agents.ppo.train import TrainingState
    from brax.training import types
    from jit_dvgc.generative_bridge.learner_continuation import LearnerHooks
    p=params();tx=split_optimizer();st=tx.init(p)
    u,st=tx.update(p,st,p);p=optax.apply_updates(p,u)
    state=TrainingState(params=p,optimizer_state=st,normalizer_params={'x':jp.zeros(1)},
        env_steps=types.UInt64(hi=0,lo=8))
    key=jax.random.PRNGKey(7)
    first=LearnerHooks(tmp_path/'first',contract='c',trainer_sha='s',parent=None,preserve_stage_clock=True)
    first.initialize(state,key);first.save(8,state,key)
    second=LearnerHooks(tmp_path/'second',contract='c',trainer_sha='s',parent=first.latest_receipt(),preserve_stage_clock=True)
    restored,rng=second.initialize(state,jax.random.PRNGKey(99))
    restored,rng=jax.device_put((restored,rng))
    equal(key,rng);assert int(restored.env_steps)==8
    equal(tx.update(p,st,p),tx.update(p,restored.optimizer_state,restored.params))


def test_recovery_composes_all_pinned_hooks_without_keep_access(tmp_path):
    from brax.training.agents.ppo import train
    from jit_dvgc.generative_bridge.learner_continuation import LearnerHooks
    from jit_dvgc.generative_bridge.split_optimizer import make_recovery_trainer
    hooks=LearnerHooks(tmp_path,contract='c',trainer_sha='s',parent=None)
    assert callable(make_recovery_trainer(train.train,hooks,None,PINNED_BRAX_TRAIN_SHA256,
        np.zeros((2,76),np.float32),lambda row:None,transitions=128))


def test_rl_only_loss_still_records_completed_transitions(monkeypatch):
    from types import SimpleNamespace
    from brax.training.agents.ppo import losses
    from jit_dvgc.generative_bridge import student_clock
    from jit_dvgc.generative_bridge.student import make_joint_student_trainer
    monkeypatch.setattr(student_clock,'instrument_frozen_normalizer',lambda trainer:trainer)
    monkeypatch.setattr(losses,'compute_ppo_loss',lambda *args,**kwargs:(jp.asarray(1.),{}))
    normalizer=SimpleNamespace(count=jp.asarray(0.))
    net=SimpleNamespace(policy_network=SimpleNamespace(apply=lambda *a:None),
                        parametric_action_distribution=SimpleNamespace(mode=lambda a:a))
    data=SimpleNamespace(extras={'policy_extras':{'completed_training_transitions':jp.asarray([128.])}})
    def trainer(**kwargs):
        return losses.compute_ppo_loss(params(),normalizer,data,jax.random.PRNGKey(0),net)[1]
    metrics=make_joint_student_trainer(trainer,None,retention=None,keep_coefficient=0,
        demo_coefficient_start=0,demo_coefficient_end=0,freeze_actor_normalizer=True,
        demo_clock='completed_transitions',record_update_clock=True)(restore_params=(normalizer,None,None))
    assert float(metrics['actual/completed_training_transitions'])==128
    assert float(metrics['demo_samples_per_loss'])==0
