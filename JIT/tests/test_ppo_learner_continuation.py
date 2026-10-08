import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from brax.training.agents.ppo.train import TrainingState
from brax.training.agents.ppo.losses import PPONetworkParams
from brax.training import types


def state_after_updates():
    params=PPONetworkParams(policy={'w':jnp.array([1.,2.])},value={'w':jnp.array([3.,4.])})
    opt=optax.adam(.01);state=opt.init(params)
    for _ in range(3):
        updates,state=opt.update(jax.tree.map(jnp.ones_like,params),state,params)
        params=optax.apply_updates(params,updates)
    return opt,TrainingState(optimizer_state=state,params=params,normalizer_params={'mean':jnp.array([.3])},env_steps=types.UInt64(hi=0,lo=3200))


def test_resume_preserves_adam_critic_normalizer_rng_and_next_update(tmp_path):
    from jit_dvgc.generative_bridge.learner_continuation import LearnerHooks
    opt,s=state_after_updates();key=jax.random.PRNGKey(55)
    first=LearnerHooks(tmp_path/'first',contract='contract',trainer_sha='trainer',parent=None)
    first.initialize(s,key);first.save(3200,s,key)
    parent=first.latest_receipt()
    second=LearnerHooks(tmp_path/'second',contract='contract',trainer_sha='trainer',parent=parent)
    template=s.replace(optimizer_state=opt.init(s.params),env_steps=types.UInt64(hi=0,lo=0))
    restored,rng=second.initialize(template,jax.random.PRNGKey(3))
    np.testing.assert_array_equal(rng,key)
    assert int(restored.env_steps)==0 and second.offset==3200
    grads=jax.tree.map(lambda x:jnp.full_like(x,.2),s.params)
    actual=opt.update(grads,restored.optimizer_state,restored.params)
    expected=opt.update(grads,s.optimizer_state,s.params)
    for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(expected)):np.testing.assert_array_equal(a,b)
    for a,b in zip(jax.tree.leaves(restored.params),jax.tree.leaves(s.params)):np.testing.assert_array_equal(a,b)
    second.save(3200,restored,key)
    assert second.latest_receipt()['lifetime_transitions']==6400
    wrong=LearnerHooks(tmp_path/'bad',contract='changed',trainer_sha='trainer',parent=parent)
    with pytest.raises(ValueError,match='contract'):wrong.initialize(template,key)


def test_adapter_instruments_installed_brax_without_mutating_it(tmp_path):
    import inspect
    from brax.training.agents.ppo import train
    from jit_dvgc.generative_bridge.learner_continuation import instrument_trainer
    original=inspect.getsource(train.train)
    patched=instrument_trainer(train.train,lambda s,k:(s,k),lambda *a:None)
    assert patched is not train.train
    assert inspect.getsource(train.train)==original


def test_continuous_decision_has_no_nominal_or_performance_veto():
    from jit_dvgc.generative_bridge.closed_loop import continuous_decision
    result=continuous_decision({'actor_sha256':'new'},True)
    assert result['adopted'] and not result['performance_gate']
    assert result['nominal_evaluation']=='disabled_by_user'
    with pytest.raises(ValueError):continuous_decision({},False)


def test_real_brax_cpu_two_blocks_resume_full_learner(tmp_path):
    import functools,pickle
    from brax.envs.fast import Fast
    from brax.training.agents.ppo import train,networks
    from jit_dvgc.generative_bridge.learner_continuation import LearnerHooks,instrument_trainer
    first=LearnerHooks(tmp_path/'a',contract='fixed',trainer_sha='pinned',parent=None)
    kwargs=dict(environment=Fast(),num_timesteps=16,num_envs=2,episode_length=4,unroll_length=2,
        batch_size=2,num_minibatches=1,num_updates_per_batch=1,num_evals=2,run_evals=False,
        normalize_observations=True,network_factory=functools.partial(networks.make_ppo_networks,
            policy_hidden_layer_sizes=(8,),value_hidden_layer_sizes=(8,)))
    instrument_trainer(train.train,first.initialize,first.save)(**kwargs)
    parent=first.latest_receipt()
    import json
    m=json.loads(__import__('pathlib').Path(parent['path']).read_text())
    with open(m['state_path'],'rb') as f:saved=pickle.load(f)['training_state']
    second=LearnerHooks(tmp_path/'b',contract='fixed',trainer_sha='pinned',parent=parent)
    instrument_trainer(train.train,second.initialize,second.save)(**{**kwargs,'environment':Fast(),
        'restore_params':(saved.normalizer_params,saved.params.policy,saved.params.value),'restore_value_fn':True})
    assert second.latest_receipt()['lifetime_transitions']==32
    initial=json.loads((tmp_path/'b/initialization.json').read_text())
    assert initial['optimizer_sha256']==m['optimizer_sha256']
    assert initial['critic_sha256']==m['critic_sha256']
