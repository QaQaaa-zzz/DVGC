"""The randomized successor keeps jump_ori identity while changing only declared domains."""

from __future__ import annotations

import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jit_dvgc.config import load_config
from jit_dvgc.env import TwoPhaseBikeEnv
from jit_dvgc.phase_u_sim2sim_randomization import (
    apply_root_reset_perturbation,
    sample_domain_fields,
)


def _config(jit_root, tmp_path, mutate=None):
    reference = jit_root / "configs/phase_u_speed2_dense_10m_deferred_seed820701_20260928.json"
    payload = json.loads(reference.read_text())
    payload.pop("rerun_reference")
    payload["ppo"]["seed"] = 820702
    block = 24_576
    payload["formal"]["checkpoint_transitions"] = list(range(0, 10_002_432 + block, block))
    payload["sim2sim_reference"] = {
        "resolved_config": str(reference.resolve()),
        "sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
    }
    payload["sim2sim_randomization"] = {
        "wheel_lateral_friction": [0.5, 5.0],
        "wheel_forward_friction": [0.35, 0.65],
        "body_mass_inertia_scale": [0.9, 1.1],
        "actuator_scale": [0.9, 1.1],
        "reset_position_halfwidth_m": [0.03, 0.02, 0.005],
        "reset_orientation_halfwidth_rad": [np.pi / 180, np.pi / 180, 2 * np.pi / 180],
        "reset_linear_velocity_halfwidth_mps": [0.2, 0.1, 0.1],
        "reset_angular_velocity_halfwidth_radps": [0.15, 0.15, 0.15],
    }
    if mutate:
        mutate(payload)
    path = tmp_path / "successor.json"
    path.write_text(json.dumps(payload))
    return path


def test_sim2sim_config_locks_reward_and_saves_every_block(jit_root, tmp_path):
    cfg = load_config(_config(jit_root, tmp_path))
    assert cfg.ppo.seed == 820702
    assert len(cfg.formal.checkpoint_transitions) == 408
    assert cfg.formal.fixed_evaluation_transitions == ()
    assert cfg.sim2sim_randomization.wheel_forward_friction == (0.35, 0.65)


def test_domain_wrapper_model_slot_controls_runtime_model(jit_root, tmp_path):
    env = TwoPhaseBikeEnv(load_config(_config(jit_root, tmp_path)), convert_model=False)
    assert env._mjx_model is env.mjx_model
    marker = object()
    env._mjx_model = marker
    assert env.mjx_model is marker


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p["reward"].update(speed_coeff=9), "reference.*drift"),
        (lambda p: p["sim2sim_randomization"].update(wheel_forward_friction=[-0.1, 0.5]), "friction"),
        (lambda p: p["formal"].update(checkpoint_transitions=[0, 10_002_432]), "every.*block"),
        (lambda p: p["formal"].update(fixed_evaluation_transitions=[24_576]), "fixed evaluations"),
        (lambda p: p["sim2sim_reference"].update(sha256="0" * 64), "hash"),
    ],
)
def test_sim2sim_config_rejects_drift(jit_root, tmp_path, mutate, message):
    with pytest.raises(ValueError, match=message):
        load_config(_config(jit_root, tmp_path, mutate))


def test_initial_perturbation_is_bounded_and_preserves_unit_quaternion(jit_root, tmp_path):
    cfg = load_config(_config(jit_root, tmp_path)).sim2sim_randomization
    qpos = jnp.array([2.5, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.4], dtype=jnp.float32)
    qvel = jnp.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0], dtype=jnp.float32)
    result = [apply_root_reset_perturbation(qpos, qvel, 0, 0, jax.random.PRNGKey(i), cfg) for i in range(32)]
    for new_pos, new_vel in result:
        np.testing.assert_array_less(np.abs(np.asarray(new_pos[:3] - qpos[:3])), np.asarray(cfg.reset_position_halfwidth_m) + 1e-6)
        np.testing.assert_array_less(np.abs(np.asarray(new_vel[:3] - qvel[:3])), np.asarray(cfg.reset_linear_velocity_halfwidth_mps) + 1e-6)
        np.testing.assert_array_less(np.abs(np.asarray(new_vel[3:6])), np.asarray(cfg.reset_angular_velocity_halfwidth_radps) + 1e-6)
        np.testing.assert_allclose(np.linalg.norm(np.asarray(new_pos[3:7])), 1.0, atol=1e-6)
        assert float(new_pos[7]) == pytest.approx(0.4)
        assert float(new_vel[6]) == pytest.approx(3.0)
    assert not np.array_equal(np.asarray(result[0][0]), np.asarray(result[1][0]))


def test_domain_fields_randomize_explicit_wheel_pairs_and_coupled_inertia(jit_root, tmp_path):
    cfg = load_config(_config(jit_root, tmp_path)).sim2sim_randomization
    fields = sample_domain_fields(
        pair_friction=jnp.array([[1., 0.1, 0., 0., 0.]] * 6),
        body_mass=jnp.array([0., 1., 2.]),
        body_inertia=jnp.array([[0., 0., 0.], [1., 2., 3.], [2., 3., 4.]]),
        actuator_gainprm=jnp.array([[10., 0., 0.], [5., 0., 0.]]),
        actuator_biasprm=jnp.array([[0., -10., -0.1], [0., 0., -5.]]),
        wheel_pair_indices=(2, 3, 4, 5),
        rng=jax.random.PRNGKey(4),
        config=cfg,
    )
    fric, mass, inertia, gain, bias = map(np.asarray, fields)
    assert np.all((fric[2:6, 0] >= .5) & (fric[2:6, 0] <= 5.0))
    assert np.all((fric[2:6, 1] >= .35) & (fric[2:6, 1] <= .65))
    np.testing.assert_allclose(fric[:2], np.array([[1., .1, 0., 0., 0.]] * 2), rtol=1e-6)
    np.testing.assert_allclose(
        inertia[1:] / np.array([[1., 2., 3.], [2., 3., 4.]]),
        np.broadcast_to(mass[1:, None] / np.array([[1.], [2.]]), (2, 3)),
        rtol=1e-6,
    )
    np.testing.assert_allclose(bias[0, 1], -gain[0, 0], rtol=1e-6)
    assert mass[0] == 0.0
