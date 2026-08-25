from __future__ import annotations

from functools import lru_cache

import jax
from jax import numpy as jp
import numpy as np
import pytest
from mujoco_playground._src import wrapper

from jit_dvgc.config import load_config
from jit_dvgc.env import TwoPhaseBikeEnv


pytestmark = pytest.mark.gpu


@lru_cache(maxsize=1)
def _environment(config_path: str) -> TwoPhaseBikeEnv:
    assert jax.default_backend() == "gpu"
    return TwoPhaseBikeEnv(load_config(config_path), convert_model=True)


def test_jitted_reset_and_step_preserve_pytree_and_advance_exact_control_time(jit_root):
    env = _environment(str(jit_root / "configs" / "phase_u_smoke.json"))
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    state = reset(jax.random.PRNGKey(10))
    next_state = step(state, jp.zeros(4, dtype=jp.float32))
    jax.block_until_ready(next_state)

    assert jax.tree.structure(state) == jax.tree.structure(next_state)
    assert float(next_state.data.time - state.data.time) == pytest.approx(0.020, abs=1e-6)
    assert state.obs["state"].shape == (76,)
    assert state.obs["privileged_state"].shape == (106,)
    np.testing.assert_array_equal(np.asarray(state.obs["state"]), np.zeros(76))
    assert set(state.metrics) == set(next_state.metrics)
    assert set(state.info) == set(next_state.info)
    assert float(state.info["time_out"]) == 0.0
    assert float(next_state.info["time_out"]) == float(next_state.info["truncated"])


def test_step_preserves_info_fields_added_by_training_wrappers(jit_root):
    env = _environment(str(jit_root / "configs" / "phase_u_smoke.json"))
    state = jax.jit(env.reset)(jax.random.PRNGKey(13))
    state = state.replace(
        info={**state.info, "wrapper_probe": jp.asarray(7, dtype=jp.int32)}
    )
    next_state = jax.jit(env.step)(state, jp.zeros(4, dtype=jp.float32))
    jax.block_until_ready(next_state)

    assert int(next_state.info["wrapper_probe"]) == 7
    assert set(next_state.info) == set(state.info)


def test_real_brax_wrapper_exposes_all_ppo_extra_fields(jit_root):
    env = _environment(str(jit_root / "configs" / "phase_u_smoke.json"))
    wrapped = wrapper.wrap_for_brax_training(env, episode_length=200)
    keys = jax.random.split(jax.random.PRNGKey(14), 4)
    state = jax.jit(wrapped.reset)(keys)
    next_state = jax.jit(wrapped.step)(
        state, jp.zeros((4, 4), dtype=jp.float32)
    )
    jax.block_until_ready(next_state)

    assert {
        "truncation",
        "episode_metrics",
        "episode_done",
        "time_out",
    }.issubset(next_state.info)


def test_fifty_control_ticks_equal_one_second(jit_root):
    env = _environment(str(jit_root / "configs" / "phase_u_smoke.json"))
    state = jax.jit(env.reset)(jax.random.PRNGKey(11))

    @jax.jit
    def rollout(initial):
        def body(current, _):
            return env.step(current, jp.zeros(4, dtype=jp.float32)), None

        return jax.lax.scan(body, initial, xs=None, length=50)[0]

    final = rollout(state)
    jax.block_until_ready(final)
    assert float(final.data.time - state.data.time) == pytest.approx(1.0, abs=2e-5)


def test_forced_natural_reset_never_uses_airborne_rsi(jit_root):
    env = _environment(str(jit_root / "configs" / "phase_u_smoke.json"))
    state = jax.jit(env.reset_natural)(jax.random.PRNGKey(101))
    jax.block_until_ready(state)

    assert float(state.data.qpos[0]) == pytest.approx(1.5)
    assert float(state.data.qpos[2]) == pytest.approx(0.15)
    assert float(state.data.qvel[0]) == pytest.approx(2.0)
    assert float(state.data.qvel[2]) == pytest.approx(0.0)
    assert float(state.metrics["reset/source_airborne_rsi"]) == 0.0
    assert float(state.info["events"].jump_signal) == 0.0
    assert float(state.obs["state"][-1]) == 0.0


def test_forced_airborne_reset_always_uses_rsi_and_jump_signal(jit_root):
    env = _environment(str(jit_root / "configs" / "phase_u_absolute_smoke.json"))
    state = jax.jit(env.reset_airborne_rsi)(jax.random.PRNGKey(103))
    jax.block_until_ready(state)

    assert 2.7 <= float(state.data.qpos[0]) <= 2.9
    assert 1.8 <= float(state.data.qpos[2]) <= 2.2
    assert 1.8 <= float(state.data.qvel[0]) <= 2.2
    assert 0.8 <= float(state.data.qvel[2]) <= 1.2
    assert float(state.metrics["reset/source_airborne_rsi"]) == 1.0
    assert float(state.info["events"].jump_signal) == 1.0
    assert float(state.obs["state"][-1]) == 1.0


def test_forced_airborne_rollout_continues_past_first_apex(jit_root):
    env = _environment(str(jit_root / "configs" / "phase_u_absolute_smoke.json"))
    state = jax.jit(env.reset_airborne_rsi)(jax.random.PRNGKey(104))
    step = jax.jit(env.step)
    saw_apex = False
    for _ in range(20):
        state = step(state, jp.zeros(4, dtype=jp.float32))
        jax.block_until_ready(state)
        if bool(state.info["events"].apex_seen):
            saw_apex = True
            assert not bool(state.info["terminated"])
            assert not bool(state.done)
            break
        if bool(state.done):
            break
    assert saw_apex


def test_mixed_reset_is_reproducible_and_airborne_samples_are_bounded(jit_root):
    env = _environment(str(jit_root / "configs" / "phase_u_smoke.json"))
    keys = jax.random.split(jax.random.PRNGKey(102), 1024)
    reset_many = jax.jit(jax.vmap(env.reset))
    first = reset_many(keys)
    second = reset_many(keys)
    jax.block_until_ready((first, second))

    np.testing.assert_array_equal(np.asarray(first.data.qpos), np.asarray(second.data.qpos))
    sources = np.asarray(first.metrics["reset/source_airborne_rsi"]) > 0.5
    assert 20 <= int(sources.sum()) <= 85
    qpos = np.asarray(first.data.qpos)[sources]
    qvel = np.asarray(first.data.qvel)[sources]
    assert np.all((2.7 <= qpos[:, 0]) & (qpos[:, 0] <= 2.9))
    assert np.all((1.8 <= qpos[:, 2]) & (qpos[:, 2] <= 2.2))
    assert np.all((1.8 <= qvel[:, 0]) & (qvel[:, 0] <= 2.2))
    assert np.all((0.8 <= qvel[:, 2]) & (qvel[:, 2] <= 1.2))
    assert np.all(np.asarray(first.info["events"].jump_signal)[sources])
    assert np.all(np.asarray(first.obs["state"])[sources, -1] == 1.0)


def test_1024_environment_short_rollout_is_finite(jit_root):
    env = _environment(str(jit_root / "configs" / "phase_u_smoke.json"))
    keys = jax.random.split(jax.random.PRNGKey(12), 1024)
    reset_many = jax.jit(jax.vmap(env.reset))
    step_many = jax.jit(jax.vmap(env.step))
    state = reset_many(keys)
    actions = jp.zeros((1024, 4), dtype=jp.float32)
    for _ in range(4):
        state = step_many(state, actions)
    jax.block_until_ready(state)

    assert state.data.qpos.shape == (1024, 12)
    assert state.obs["state"].shape == (1024, 76)
    assert state.obs["privileged_state"].shape == (1024, 106)
    assert bool(jp.isfinite(state.data.qpos).all())
    assert bool(jp.isfinite(state.data.qvel).all())
    assert bool(jp.isfinite(state.obs["state"]).all())
    assert bool(jp.isfinite(state.obs["privileged_state"]).all())
    assert bool(jp.isfinite(state.reward).all())
