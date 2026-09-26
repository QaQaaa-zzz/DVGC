from pathlib import Path
import numpy as np
import pytest
from jit_dvgc.policy_distillation import load_dataset, distill
from jit_dvgc.checkpoint import CheckpointIdentity, load_checkpoint


def test_dataset_rejects_invalid_actions_and_weights(tmp_path):
    path = tmp_path / 'data.npz'
    np.savez(path, observations=np.zeros((4, 3)), actions=np.ones((4, 4))*1.1, weights=np.ones(4))
    with pytest.raises(ValueError, match='actions'):
        load_dataset(path)
    np.savez(path, observations=np.zeros((4, 3)), actions=np.zeros((4, 4)), weights=np.zeros(4))
    with pytest.raises(ValueError, match='weights'):
        load_dataset(path)


def test_distills_fresh_actor_and_restores_standard_checkpoint(tmp_path):
    path = tmp_path / 'data.npz'
    rng = np.random.default_rng(3)
    obs = rng.normal(size=(48, 6)).astype(np.float32)
    actions = np.tile([.6, -.4, .2, -.1], (48, 1)).astype(np.float32)
    np.savez(path, observations=obs, actions=actions, weights=np.ones(48))
    identity = CheckpointIdentity('config', 'xml', ('frame',), ('task',), ('a','b','c','d'))
    result = distill(path, tmp_path/'ckpt', identity=identity,
                     privileged_observation_size=8, updates=40, batch_size=32, seed=4)
    assert result['final_action_mse'] < result['initial_action_mse'] * .2
    payload = load_checkpoint(tmp_path/'ckpt', expected=identity)
    assert payload.training_transitions == 0
    assert result['actor_initialization'] == 'fresh_random'
    assert result['environment_interactions'] == 0
    assert payload.observation_normalizer.mean['state'].shape == (6,)
    assert np.all(payload.observation_normalizer.std['state'] >= 1.)
    assert payload.observation_normalizer.mean['privileged_state'].shape == (8,)


def test_ppo_supervision_decay_and_restores_loss_on_failure(tmp_path, monkeypatch):
    import jax
    import jax.numpy as jp
    from types import SimpleNamespace
    from brax.training.types import UInt64
    from brax.training.agents.ppo import losses
    from jit_dvgc.policy_distillation import wrap_trainer
    path = tmp_path / 'data.npz'
    np.savez(path, observations=np.zeros((3, 2)), actions=np.ones((3, 4)), weights=np.ones(3))
    def base(*args, **kwargs):
        return jp.array(2.), {}
    monkeypatch.setattr(losses, 'compute_ppo_loss', base)
    networks = SimpleNamespace(
        policy_network=SimpleNamespace(apply=lambda norm, policy, obs: jp.zeros((len(obs['state']), 4))),
        parametric_action_distribution=SimpleNamespace(mode=lambda x: x))
    def trainer(**kwargs):
        fn = losses.compute_ppo_loss
        result, metrics = fn(SimpleNamespace(policy=None), SimpleNamespace(count=UInt64(0, 60)),
                             None, jax.random.PRNGKey(0), networks)
        assert float(result) == 3.
        assert float(metrics['distillation_coefficient']) == 1.
        result, _ = fn(SimpleNamespace(policy=None), SimpleNamespace(count=UInt64(0, 110)),
                       None, jax.random.PRNGKey(0), networks)
        assert float(result) == 2.
        raise RuntimeError('trainer error')
    wrapped = wrap_trainer(trainer, path, coefficient=2., batch_size=2, decay_transitions=100)
    with pytest.raises(RuntimeError, match='trainer error'):
        wrapped(restore_params=(SimpleNamespace(count=UInt64(0, 10)), None, None))
    assert losses.compute_ppo_loss is base
