from __future__ import annotations

import jax
from jax import numpy as jp
import numpy as np
import pytest

from jit_dvgc.ppo import wrap_for_jit_training
from jit_dvgc.unified_formal import build_unified_formal_environment
from jit_dvgc.unified_natural_evaluation import NaturalStartUnifiedEvalEnv


pytestmark = pytest.mark.gpu


def _finite_tree(tree) -> bool:
    for leaf in jax.tree.leaves(jax.device_get(tree)):
        if not np.isfinite(np.asarray(leaf)).all():
            return False
    return True


def _datawarp_metadata(state) -> str:
    return repr(jax.tree.structure(state.data)).replace(" ", "")


def _assert_source_one_hot(state) -> None:
    natural = np.asarray(jax.device_get(state.metrics["reset/source_natural"]))
    soft = np.asarray(jax.device_get(state.metrics["reset/source_soft_tube"]))
    np.testing.assert_allclose(natural + soft, np.ones_like(natural))
    assert np.isin(natural, [0.0, 1.0]).all()
    assert np.isin(soft, [0.0, 1.0]).all()


def _reset_from_pre_forward_natural(env, rng):
    tube_sample = env.tube_pool.sample(rng)
    natural_sample = env._natural_reset_sample(rng, tube_sample)
    state = env._reset_from_tube_sample(natural_sample)
    return env._with_reset_source(state, soft_tube=False)


def test_unified_natural_and_tube_reset_share_runtime_capacity_and_pytree(jit_root):
    assert jax.default_backend() == "gpu"
    config, _artifact, env = build_unified_formal_environment(
        jit_root / "configs/pi_unified_round1_natural10.json"
    )
    assert config.runtime_naccdmax == 1024
    assert env._reset_data_naccdmax() == 1024

    natural = jax.jit(env._reset_natural_unified)(jax.random.PRNGKey(9_500_011))
    tube = jax.jit(env.reset_tube_index)(np.int32(0), np.int32(0))
    jax.block_until_ready(natural)
    jax.block_until_ready(tube)

    assert jax.tree.structure(natural) == jax.tree.structure(tube)
    natural_metadata = _datawarp_metadata(natural)
    tube_metadata = _datawarp_metadata(tube)
    assert natural_metadata == tube_metadata
    assert "(1024,4096,256,1)" in natural_metadata
    assert bool(np.asarray(natural.info["expert_switching_used"])) is False
    assert bool(np.asarray(tube.info["expert_switching_used"])) is False
    assert float(natural.metrics["reset/source_natural"]) == pytest.approx(1.0)
    assert float(tube.metrics["reset/source_soft_tube"]) == pytest.approx(1.0)
    assert _finite_tree(natural.data.qpos)
    assert _finite_tree(natural.data.qvel)
    assert _finite_tree(tube.data.qpos)
    assert _finite_tree(tube.data.qvel)


def test_pre_forward_natural_reset_matches_existing_natural_semantics(jit_root):
    assert jax.default_backend() == "gpu"
    _config, _artifact, env = build_unified_formal_environment(
        jit_root / "configs/pi_unified_round1_natural10.json"
    )
    key = jax.random.PRNGKey(9_500_017)
    existing = jax.jit(env._reset_natural_unified)(key)
    pre_forward = jax.jit(lambda rng: _reset_from_pre_forward_natural(env, rng))(key)
    jax.block_until_ready(existing)
    jax.block_until_ready(pre_forward)

    assert jax.tree.structure(existing) == jax.tree.structure(pre_forward)
    assert _datawarp_metadata(existing) == _datawarp_metadata(pre_forward)
    for left, right in (
        (existing.data.qpos, pre_forward.data.qpos),
        (existing.data.qvel, pre_forward.data.qvel),
        (existing.data.ctrl, pre_forward.data.ctrl),
        (existing.obs["state"], pre_forward.obs["state"]),
        (existing.obs["privileged_state"], pre_forward.obs["privileged_state"]),
    ):
        np.testing.assert_allclose(
            np.asarray(jax.device_get(left)),
            np.asarray(jax.device_get(right)),
            rtol=0.0,
            atol=0.0,
        )
    assert int(pre_forward.info["start_phase"]) == 0
    assert int(pre_forward.info["source_tick"]) == -1
    assert int(pre_forward.info["parent_group_index"]) == -1
    assert int(pre_forward.info["tube_entry_index"]) == -1
    assert int(pre_forward.info["tube_global_index"]) == -1
    assert float(pre_forward.metrics["reset/source_natural"]) == pytest.approx(1.0)
    assert float(pre_forward.metrics["reset/source_soft_tube"]) == pytest.approx(0.0)
    assert bool(np.asarray(pre_forward.info["expert_switching_used"])) is False


def test_natural_evaluation_reset_can_enter_jitted_unified_step(jit_root):
    assert jax.default_backend() == "gpu"
    _config, _artifact, env = build_unified_formal_environment(
        jit_root / "configs/pi_unified_round1_natural10.json",
        env_factory=NaturalStartUnifiedEvalEnv,
    )
    reset = jax.jit(env.reset_natural)
    step = jax.jit(env.step)
    state = reset(jax.random.PRNGKey(9_400_001))
    assert bool(np.asarray(state.info["reset_from_soft_tube"])) is False
    assert float(state.metrics["reset/source_natural"]) == pytest.approx(1.0)
    assert float(state.metrics["reset/source_soft_tube"]) == pytest.approx(0.0)

    next_state = step(state, jp.zeros(4, dtype=jp.float32))
    jax.block_until_ready(next_state)

    assert jax.tree.structure(state) == jax.tree.structure(next_state)
    assert bool(np.asarray(next_state.info["reset_from_soft_tube"])) is False
    assert bool(np.asarray(next_state.info["expert_switching_used"])) is False
    assert float(next_state.metrics["reset/source_natural"]) == pytest.approx(1.0)
    assert float(next_state.metrics["reset/source_soft_tube"]) == pytest.approx(0.0)
    assert _finite_tree(next_state.data.qpos)
    assert _finite_tree(next_state.data.qvel)
    assert _finite_tree(next_state.metrics)


def test_jitted_mixed_reset_step_and_fixed_panel_contract(jit_root):
    assert jax.default_backend() == "gpu"
    _config, _artifact, env = build_unified_formal_environment(
        jit_root / "configs/pi_unified_round1_natural10.json"
    )
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    action = jp.zeros(4, dtype=jp.float32)
    state = reset(jax.random.PRNGKey(9_500_021))
    next_state = step(state, action)
    jax.block_until_ready(next_state)

    assert jax.tree.structure(state) == jax.tree.structure(next_state)
    _assert_source_one_hot(state)
    _assert_source_one_hot(next_state)
    assert float(next_state.metrics["reset/source_natural"]) == pytest.approx(
        float(state.metrics["reset/source_natural"])
    )
    assert float(next_state.metrics["reset/source_soft_tube"]) == pytest.approx(
        float(state.metrics["reset/source_soft_tube"])
    )
    assert bool(np.asarray(state.info["expert_switching_used"])) is False
    assert bool(np.asarray(next_state.info["expert_switching_used"])) is False
    assert np.isfinite(np.asarray(action)).all()
    assert _finite_tree(next_state.data.qpos)
    assert _finite_tree(next_state.data.qvel)
    assert _finite_tree(next_state.metrics)

    panel_state = jax.jit(env.reset_tube_index)(np.int32(0), np.int32(0))
    jax.block_until_ready(panel_state)
    assert float(panel_state.metrics["reset/source_soft_tube"]) == pytest.approx(1.0)
    assert float(panel_state.metrics["reset/source_natural"]) == pytest.approx(0.0)
    assert bool(np.asarray(panel_state.info["expert_switching_used"])) is False


def test_brax_full_reset_cycle_resamples_without_pytree_drift(jit_root):
    assert jax.default_backend() == "gpu"
    _config, _artifact, env = build_unified_formal_environment(
        jit_root / "configs/pi_unified_round1_natural10.json"
    )
    wrapped = wrap_for_jit_training(env, episode_length=1, action_repeat=1)
    reset = jax.jit(wrapped.reset)
    step = jax.jit(wrapped.step)
    keys = jax.random.split(jax.random.PRNGKey(9_500_031), 128)
    actions = jp.zeros((128, 4), dtype=jp.float32)

    state = reset(keys)
    next_state = step(state, actions)
    second_state = step(next_state, actions)
    jax.block_until_ready(second_state)

    assert jax.tree.structure(state) == jax.tree.structure(next_state)
    assert jax.tree.structure(next_state) == jax.tree.structure(second_state)
    for candidate in (state, next_state, second_state):
        _assert_source_one_hot(candidate)
        assert not np.asarray(
            jax.device_get(candidate.info["expert_switching_used"])
        ).any()
        assert _finite_tree(candidate.data.qpos)
        assert _finite_tree(candidate.data.qvel)
        assert _finite_tree(candidate.metrics)
    before = np.asarray(jax.device_get(state.info["reset_from_soft_tube"]))
    after = np.asarray(jax.device_get(next_state.info["reset_from_soft_tube"]))
    assert np.any(before != after)
    assert np.isfinite(np.asarray(actions)).all()
