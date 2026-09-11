"""CPU architecture, stochastic equivalence, and normalizer continuation checks."""
import jax
import jax.numpy as jp
import numpy as np
import pytest
from brax.training.acme import running_statistics as rs
from brax.training.agents.ppo import networks as ppo_networks
from flax.core import freeze

from jit_dvgc.exploration_network import make_exploration_network_factory, widen_actor_warm_start
from jit_dvgc.ppo import make_network_factory


def fixture():
    sizes = {"state": 76, "privileged_state": 106}
    old = make_network_factory()(sizes, 4, preprocess_observations_fn=rs.normalize)
    new = make_exploration_network_factory()(sizes, 4, preprocess_observations_fn=rs.normalize)
    stats = rs.init_state({key: jp.zeros(size) for key, size in sizes.items()})
    data = {key: jax.random.normal(jax.random.PRNGKey(size), (32, size)) * 2 + 3 for key, size in sizes.items()}
    stats = rs.update(stats, data)
    params = old.policy_network.init(jax.random.PRNGKey(7))
    return old, new, stats, params


@pytest.mark.parametrize("frozen", [False, True])
def test_widening_preserves_logits_and_stochastic_actions(frozen):
    old, new, stats, params = fixture()
    if frozen:
        params = freeze(params)
    normalizer, widened = widen_actor_warm_start(stats, params)
    state = jax.random.normal(jax.random.PRNGKey(12), (5, 76))
    obs = {"state": state, "privileged_state": jp.concatenate((state, jp.ones((5, 30)) * 19), axis=-1)}
    expected = old.policy_network.apply(stats, params, obs)
    actual = new.policy_network.apply(normalizer, widened, obs)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    assert actual.shape == (5, 8)
    for deterministic in (False, True):
        a = ppo_networks.make_inference_fn(old)((stats, params), deterministic=deterministic)(obs, jax.random.PRNGKey(8))[0]
        b = ppo_networks.make_inference_fn(new)((normalizer, widened), deterministic=deterministic)(obs, jax.random.PRNGKey(8))[0]
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)
        assert b.shape == (5, 4)
        assert np.all(np.abs(b) <= 1)
    assert type(params) is type(widened)
    assert params['params']['hidden_0']['kernel'].shape == (76, 256)
    assert widened['params']['hidden_0']['kernel'].shape == (106, 256)
    np.testing.assert_array_equal(widened['params']['hidden_0']['kernel'][76:], 0)
    for layer in ('hidden_1', 'hidden_2', 'hidden_3'):
        for key in ('kernel', 'bias'):
            np.testing.assert_array_equal(params['params'][layer][key], widened['params'][layer][key])
    value = new.value_network.init(jax.random.PRNGKey(99))
    assert value['params']['hidden_0']['kernel'].shape == (106, 256)
    assert np.isfinite(new.value_network.apply(normalizer, value, obs)).all()


def test_normalizer_count_stats_and_safe_next_update():
    _, _, stats, params = fixture()
    result = widen_actor_warm_start(stats, params)
    assert len(result) == 2  # No frozen critic or optimizer in restore artifact.
    normalizer = result[0]
    assert normalizer.count is stats.count
    for field in ('mean', 'std', 'summed_variance'):
        np.testing.assert_array_equal(getattr(normalizer, field)['privileged_state'][:76], getattr(stats, field)['state'])
        np.testing.assert_array_equal(getattr(normalizer, field)['state'], getattr(stats, field)['state'])
    np.testing.assert_array_equal(normalizer.mean['privileged_state'][76:], 0)
    np.testing.assert_array_equal(normalizer.std['privileged_state'][76:], 1)
    updated = rs.update(normalizer, {'state': jp.zeros((1, 76)), 'privileged_state': jp.zeros((1, 106))})
    np.testing.assert_allclose(updated.std['privileged_state'][76:], np.sqrt(32 / 33), rtol=1e-6)


def test_rejects_shape_drift():
    _, _, stats, params = fixture()
    with pytest.raises(ValueError, match='normalizer observation shape'):
        widen_actor_warm_start(stats, params, actor_observation_size=75)
    with pytest.raises(ValueError, match='strictly wider'):
        widen_actor_warm_start(stats, params, privileged_observation_size=76)
