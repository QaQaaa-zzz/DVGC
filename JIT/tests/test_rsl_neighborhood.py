"""CPU contracts for causal neighborhood inputs to the RSL explorer."""
import numpy as np
import pytest

from jit_dvgc import rsl_pulse


def spec():
    return dict(seed=23, learning_rate=.001, epochs=2, minibatch_size=8,
                clip=.2, target_kl=.01, entropy_coefficient=.001,
                value_coefficient=.5, max_grad_norm=1., neighborhood={
                    'neighbors':16, 'feature_dim':17, 'summary_dim':64,
                    'base_dim':106, 'stats_dim':8})


def observations(shape=(3, 8)):
    x = np.random.default_rng(7).normal(size=(*shape, 386)).astype('float32')
    rows = x[..., 106:378].reshape(*shape, 16, 17)
    rows[..., -1] = 0
    rows[..., :5, -1] = 1
    return x


def initialized():
    return rsl_pulse.initialize(spec(), np.full(106, 2.), np.full(106, 3.))


def assert_parity(state, obs):
    import torch
    import jax.numpy as jp
    from tensordict import TensorDict
    policy = rsl_pulse.torch_policy(spec(), state)
    flat = rsl_pulse.normalized(state, obs.reshape(-1, 386))
    td = TensorDict({'obs':torch.tensor(flat)}, [len(flat)])
    with torch.no_grad():
        policy.act(td)
        expected = (policy.action_mean.numpy(), policy.action_std.numpy(),
                    policy.evaluate(td).numpy().ravel())
    actual = rsl_pulse.infer(state, jp.asarray(obs))
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(np.asarray(got).reshape(want.shape), want,
                                   rtol=1e-5, atol=1e-6)


def test_only_base_is_normalized_and_frameworks_agree():
    state = initialized()
    assert state['normalizer_mean'].shape == (386,)
    obs = observations()
    norm = rsl_pulse.normalized(state, obs)
    np.testing.assert_allclose(norm[..., :106], (obs[..., :106]-2)/3)
    np.testing.assert_array_equal(norm[..., 106:], obs[..., 106:])
    assert_parity(state, obs)


def test_masked_padding_and_neighbor_order_do_not_change_policy():
    import jax.numpy as jp
    state = initialized()
    obs = observations((4,))
    baseline = rsl_pulse.infer(state, jp.asarray(obs))
    changed = obs.copy()
    rows = changed[:, 106:378].reshape(4, 16, 17)
    rows[:, 5:, :-1] = np.nan
    rows[:] = rows[:, np.random.default_rng(3).permutation(16)].copy()
    for got, want in zip(rsl_pulse.infer(state, jp.asarray(changed)), baseline):
        np.testing.assert_allclose(got, want, atol=1e-6)
    assert_parity(state, changed)
    empty = obs.copy()
    empty[:, 106:378] = 0
    dirty_empty = empty.copy()
    dirty_empty[:, 106:378].reshape(4,16,17)[..., :-1] = np.nan
    for got, want in zip(rsl_pulse.infer(state, jp.asarray(dirty_empty)),
                         rsl_pulse.infer(state, jp.asarray(empty))):
        assert np.isfinite(got).all()
        np.testing.assert_array_equal(got, want)


def test_context_changes_actor_and_critic_after_nonzero_head_weights():
    import torch
    import jax.numpy as jp
    state = initialized()
    policy = rsl_pulse.torch_policy(spec(), state)
    # Fresh policies deliberately have a zero mean head. Exercise learned weights.
    head = [m for m in policy.actor.modules() if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        head.weight.normal_(0, .1)
    state['params'] = rsl_pulse.export(policy)
    obs = observations((4,))
    changed = obs.copy()
    changed[:, 106:378].reshape(4,16,17)[:, :5, :16] += 2
    original = rsl_pulse.infer(state, jp.asarray(obs))
    context = rsl_pulse.infer(state, jp.asarray(changed))
    assert np.max(np.abs(original[0]-context[0])) > 1e-4
    assert np.max(np.abs(original[1]-context[1])) > 1e-4
    assert np.max(np.abs(original[2]-context[2])) > 1e-4


def test_saved_augmented_inputs_update_and_restore_without_runtime_map(tmp_path):
    import jax.numpy as jp
    from flax.serialization import msgpack_serialize
    state = initialized()
    obs = observations()
    mu, sd, value = map(np.asarray, rsl_pulse.infer(state, jp.asarray(obs)))
    raw = mu + sd*np.random.default_rng(9).normal(size=mu.shape).astype('float32')
    lp = (-.5*((raw-mu)/sd)**2-np.log(sd)-.5*np.log(2*np.pi)).sum(-1)
    mask = np.ones((3,8), bool)
    mask[1:,0] = False
    tape = dict(observation=obs, raw_action=raw, log_prob=lp, value=value, mask=mask)
    feedback = dict(eligible=[True]*7+[False], rewards=[1,-1,1,-1,1,-1,1,0],
                    component_sums={'novelty':0., 'quality':1.})
    update_spec = spec()
    del update_spec['neighborhood']
    updated, metrics, learning, logs = rsl_pulse.update_batch(update_spec, state, tape, feedback)
    assert metrics['effective_training_samples'] == 19
    assert metrics['optimizer_updates'] > 0
    assert any(not np.array_equal(state['params']['actor_encoder'][k]['kernel'], v['kernel'])
               for k,v in updated['params']['actor_encoder'].items())
    assert_parity(updated, obs)
    path = tmp_path/'state.msgpack'
    path.write_bytes(msgpack_serialize(updated))
    restored = rsl_pulse.restore(path)
    assert restored['neighborhood'] == state['neighborhood']
    for got,want in zip(rsl_pulse.infer(restored,jp.asarray(obs)),
                        rsl_pulse.infer(updated,jp.asarray(obs))):
        np.testing.assert_array_equal(got,want)
    assert_parity(restored, obs)
    with pytest.raises(ValueError, match='neighborhood'):
        rsl_pulse.torch_policy(dict(spec(), neighborhood={'neighbors':8}), restored)


def test_legacy_checkpoint_rejects_neighborhood_conversion():
    legacy_spec = spec()
    del legacy_spec['neighborhood']
    legacy = rsl_pulse.initialize(legacy_spec, np.zeros(106), np.ones(106))
    with pytest.raises(ValueError, match='neighborhood'):
        rsl_pulse.torch_policy(spec(), legacy)


def test_same_dimension_different_physical_widths_cannot_restore():
    state = initialized()
    changed = spec()
    changed['neighborhood']['medium_halfwidths'] = [.4,.05,.1,.4,.2,.6,2,5,5,10,20,20]
    with pytest.raises(ValueError, match='neighborhood'):
        rsl_pulse.torch_policy(changed, state)


def test_checkpoint_binds_full_observation_identity_and_layout(tmp_path):
    from flax.serialization import msgpack_serialize, to_bytes
    state = initialized()
    assert state['neighborhood']['medium_halfwidths'] == [.2,.05,.1,.4,.2,.6,2,5,5,10,20,20]
    assert state['neighborhood']['far_scale'] == 2.
    assert state['neighborhood_feature_layout']
    # Explicit defaults and omitted defaults mean the same observation.
    explicit = spec()
    explicit['neighborhood'].update(medium_halfwidths=tuple(state['neighborhood']['medium_halfwidths']), far_scale=2)
    rsl_pulse.validate_neighborhood_identity(explicit, state)
    collection_path = tmp_path/'collection.msgpack'
    collection_path.write_bytes(to_bytes(state))
    restored = rsl_pulse.restore(collection_path)
    assert restored['neighborhood'] == state['neighborhood']
    rsl_pulse.validate_neighborhood_identity(explicit, restored)
    changed = dict(state, neighborhood_feature_layout='incompatible_feature_order')
    path = tmp_path/'bad-layout.msgpack'
    path.write_bytes(msgpack_serialize(changed))
    with pytest.raises(ValueError, match='neighborhood'):
        rsl_pulse.restore(path)
    with pytest.raises(ValueError, match='neighborhood'):
        rsl_pulse.validate_neighborhood_identity({}, dict(state, neighborhood={k:v for k,v in state['neighborhood'].items() if k!='medium_halfwidths'}))
    # Batch map changes are allowed; they do not redefine feature semantics.
    rsl_pulse.validate_neighborhood_identity(dict(explicit, neighborhood_map_sha256='new-round'), state)
