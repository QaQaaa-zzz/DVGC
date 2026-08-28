from __future__ import annotations

import jax
from jax import numpy as jp
import numpy as np
import pytest

from jit_dvgc.unified_round1 import build_round1_environment


pytestmark = pytest.mark.gpu


def test_round1_jitted_reset_and_step_share_one_pytree(jit_root):
    assert jax.default_backend() == "gpu"
    _config, _artifact, env = build_round1_environment(
        jit_root / "configs/pi_unified_round1_natural10.json"
    )
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    state = reset(jax.random.PRNGKey(9_500_001))
    next_state = step(state, jp.zeros(4, dtype=jp.float32))
    jax.block_until_ready(next_state)

    assert jax.tree.structure(state) == jax.tree.structure(next_state)
    assert set(state.info) == set(next_state.info)
    assert set(state.metrics) == set(next_state.metrics)
    assert "round1_reset_from_soft_tube" in state.info
    assert "reset/source_natural" in state.metrics
    source_sum = float(state.metrics["reset/source_natural"]) + float(
        state.metrics["reset/source_soft_tube"]
    )
    assert source_sum == pytest.approx(1.0)
    assert float(next_state.metrics["reset/source_natural"]) == pytest.approx(
        float(state.metrics["reset/source_natural"])
    )
    assert float(next_state.metrics["reset/source_soft_tube"]) == pytest.approx(
        float(state.metrics["reset/source_soft_tube"])
    )
    assert np.isfinite(np.asarray(next_state.data.qpos)).all()
    assert np.isfinite(np.asarray(next_state.data.qvel)).all()


def test_round1_fixed_tube_panel_reset_remains_soft_tube(jit_root):
    assert jax.default_backend() == "gpu"
    _config, _artifact, env = build_round1_environment(
        jit_root / "configs/pi_unified_round1_natural10.json"
    )
    state = jax.jit(env.reset_tube_index)(np.int32(0), np.int32(0))
    jax.block_until_ready(state)
    assert float(state.metrics["reset/source_soft_tube"]) == pytest.approx(1.0)
    assert float(state.metrics["reset/source_natural"]) == pytest.approx(0.0)
    assert bool(np.asarray(state.info["expert_switching_used"])) is False
