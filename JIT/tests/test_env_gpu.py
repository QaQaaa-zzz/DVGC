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
    assert state.obs["state"].shape == (81,)
    assert state.obs["privileged_state"].shape == (114,)
    np.testing.assert_array_equal(np.asarray(state.obs["state"]), np.zeros(81))
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
    assert state.obs["state"].shape == (1024, 81)
    assert bool(jp.isfinite(state.data.qpos).all())
    assert bool(jp.isfinite(state.data.qvel).all())
    assert bool(jp.isfinite(state.obs["state"]).all())
    assert bool(jp.isfinite(state.obs["privileged_state"]).all())
    assert bool(jp.isfinite(state.reward).all())
