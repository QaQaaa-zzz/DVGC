import jax
import jax.numpy as jp
import numpy as np
import pytest
from jit_dvgc import policy_retention as retention


def test_penalty_preserves_teacher_and_pulls_student_toward_it():
    student = jp.array([[.8, -.5]])
    teacher = jp.array([[.1, -.1]])
    loss = retention.action_mse(student, teacher)
    assert float(loss) > 0
    gs, gt = jax.grad(retention.action_mse, argnums=(0, 1))(student, teacher)
    assert np.allclose(gt, 0)
    assert float(retention.action_mse(student - .1 * gs, teacher)) < float(loss)
    assert float(retention.action_mse(teacher, teacher)) == 0


def test_anchor_weights_balance_phases_and_reject_pending():
    rows = [dict(phase='upstream', witnessed=True, sampling_weight=3.),
            dict(phase='upstream', witnessed=True, sampling_weight=1.),
            dict(phase='downstream', witnessed=True, sampling_weight=2.)]
    assert np.allclose(retention.anchor_weights(rows), [.375, .125, .5])
    rows[0]['witnessed'] = False
    with pytest.raises(ValueError): retention.anchor_weights(rows)


def test_retention_counts_do_not_turn_unknown_into_loss():
    result = retention.paired_counts([1, 1, 1, 0, 0], [1, 0, None, 1, 0])
    assert result == dict(retained=1, lost=1, gained=1, unchanged_negative=1,
                         unknown=1, old_positive=3)


def test_idle_gpu_parser_blocks_any_compute_process_and_bad_output():
    from jit_dvgc.execution_gate import gpu_idle_assessment
    assert gpu_idle_assessment('')['ready']
    assert not gpu_idle_assessment('404281, 8898')['ready']
    assert not gpu_idle_assessment('driver error')['ready']


def test_zero_weight_adapter_preserves_ppo_value_and_gradient(monkeypatch, tmp_path):
    from brax.training.agents.ppo import losses
    from types import SimpleNamespace
    # Stub only simulator/data I/O and base loss; exercise the real adapter and AD.
    monkeypatch.setattr(retention, 'load_anchor', lambda *a: (np.ones((2, 1)), np.array([.5, .5])))
    def base(params, normalizer, data, rng, network, **kw):
        value = jp.sum(params.policy ** 2)
        return value, {'total_loss': value}
    monkeypatch.setattr(losses, 'compute_ppo_loss', base)
    network = SimpleNamespace(policy_network=SimpleNamespace(
        apply=lambda norm, params, obs: obs['state'] * params),
        parametric_action_distribution=SimpleNamespace(mode=jp.tanh))
    params = losses.PPONetworkParams(policy=jp.array([.8]), value=jp.array([0.]))
    def trainer(**kw):
        fn = lambda p: losses.compute_ppo_loss(p, None, None, jax.random.PRNGKey(0), network)[0]
        return jax.value_and_grad(fn)(params)
    for coefficient in (0., 1.):
        contract = dict(coefficient=coefficient, batch_size=2)
        value, grad = retention.wrap_trainer(trainer, contract, 'source', tmp_path)(
            restore_params=(None, jp.array([.1]), None))
        if coefficient == 0:
            assert np.allclose(value, .64)
            assert np.allclose(grad.policy, 1.6)
        else:
            assert float(value) > .64
            assert float(grad.policy[0]) > 1.6
        assert losses.compute_ppo_loss is base
