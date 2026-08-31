from __future__ import annotations

import jax
from jax import numpy as jp
import numpy as np
import pytest

from jit_dvgc.unified_continuation_labels import fresh_unified_continuation_start
from jit_dvgc.unified_envelope_snapshot import capture_unified_envelope_snapshot
from jit_dvgc.unified_formal import build_unified_formal_environment


pytestmark = pytest.mark.gpu


def _finite_tree(tree) -> bool:
    return all(
        np.isfinite(np.asarray(leaf)).all()
        for leaf in jax.tree.leaves(jax.device_get(tree))
    )


@pytest.mark.parametrize("phase_index", [0, 1])
def test_fresh_continuation_preserves_candidate_context_and_resets_budget(
    jit_root, phase_index
):
    assert jax.default_backend() == "gpu"
    config, _artifact, env = build_unified_formal_environment(
        jit_root / "configs/pi_unified_round1_natural10.json"
    )
    reset = jax.jit(env.reset_tube_index)
    step = jax.jit(env.step)
    state = reset(np.int32(phase_index), np.int32(0))
    jax.block_until_ready(state)
    assert int(np.asarray(state.info["active_phase"])) == phase_index

    up_events = state.info["up_events"].replace(episode_step=jp.asarray(7, jp.int32))
    state = state.replace(
        info={
            **state.info,
            "up_events": up_events,
            "episode_step": jp.asarray(7, jp.int32),
            "phase_episode_step": jp.asarray(3, jp.int32),
            "episode_return": jp.asarray(1.25, jp.float32),
        }
    )
    snapshot = capture_unified_envelope_snapshot(
        state,
        env=env,
        parent_trajectory=f"gpu_phase_{phase_index}",
        parent_state_sha256="1" * 64,
        config_sha256=config.config_sha256,
        policy_actor_sha256="2" * 64,
        policy_payload_sha256="3" * 64,
        policy_iteration=0,
    )
    fresh = fresh_unified_continuation_start(snapshot, env)

    np.testing.assert_allclose(
        np.asarray(jax.device_get(fresh.data.qpos)), snapshot.qpos, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(fresh.data.qvel)), snapshot.qvel, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(fresh.info["history"].frames)),
        snapshot.observation_fifo,
        rtol=0.0,
        atol=0.0,
    )
    assert int(np.asarray(fresh.info["active_phase"])) == phase_index
    assert int(np.asarray(fresh.info["start_phase"])) == phase_index
    assert bool(np.asarray(fresh.info["phase_transitioned"])) is False
    assert int(np.asarray(fresh.info["episode_step"])) == 0
    assert int(np.asarray(fresh.info["phase_episode_step"])) == 0
    assert int(np.asarray(fresh.info["up_events"].episode_step)) == 0
    assert float(np.asarray(fresh.info["episode_return"])) == pytest.approx(0.0)
    assert bool(np.asarray(fresh.info["expert_switching_used"])) is False

    next_state = step(fresh, jp.zeros((4,), dtype=jp.float32))
    jax.block_until_ready(next_state)
    assert jax.tree.structure(fresh) == jax.tree.structure(next_state)
    assert bool(np.asarray(next_state.info["expert_switching_used"])) is False
    assert _finite_tree(next_state.data.qpos)
    assert _finite_tree(next_state.data.qvel)
    assert _finite_tree(next_state.metrics)
