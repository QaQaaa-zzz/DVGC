from __future__ import annotations

import jax
from jax import numpy as jp
import numpy as np
import pytest

from jit_dvgc.unified_envelope_snapshot import (
    capture_unified_envelope_snapshot,
    load_unified_envelope_snapshot,
    restore_unified_envelope_snapshot,
    save_unified_envelope_snapshot,
)
from jit_dvgc.unified_formal import build_unified_formal_environment


pytestmark = pytest.mark.gpu


def _finite(tree) -> bool:
    return all(
        np.isfinite(np.asarray(leaf)).all()
        for leaf in jax.tree.leaves(jax.device_get(tree))
    )


@pytest.mark.parametrize("phase_index", [0, 1])
def test_unified_envelope_snapshot_roundtrip_can_enter_jitted_step(
    jit_root, tmp_path, phase_index
):
    assert jax.default_backend() == "gpu"
    config, artifact, env = build_unified_formal_environment(
        jit_root / "configs/pi_unified_round1_natural10.json"
    )
    phase_rows = [row for row in artifact.entries if row["phase"] == ("upstream" if phase_index == 0 else "downstream")]
    assert phase_rows
    parent = phase_rows[0]

    reset = jax.jit(env.reset_tube_index)
    step = jax.jit(env.step)
    state = reset(np.int32(phase_index), np.int32(0))
    state = step(state, jp.zeros(4, dtype=jp.float32))
    jax.block_until_ready(state)
    assert not bool(np.asarray(state.done))

    snapshot = capture_unified_envelope_snapshot(
        state,
        env=env,
        parent_trajectory=str(parent["parent_group_id"]),
        parent_state_sha256=str(parent["state_sha256"]),
        config_sha256=config.config_sha256,
        policy_actor_sha256="a" * 64,
        policy_payload_sha256="b" * 64,
        policy_iteration=0,
    )
    path = tmp_path / f"phase_{phase_index}"
    save_unified_envelope_snapshot(path, snapshot)
    restored = restore_unified_envelope_snapshot(
        load_unified_envelope_snapshot(path), env
    )
    jax.block_until_ready(restored)

    assert jax.tree.structure(restored) == jax.tree.structure(state)
    np.testing.assert_allclose(
        np.asarray(jax.device_get(restored.data.qpos)),
        np.asarray(jax.device_get(state.data.qpos)),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(restored.data.qvel)),
        np.asarray(jax.device_get(state.data.qvel)),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(restored.obs["state"])),
        np.asarray(jax.device_get(state.obs["state"])),
        rtol=0.0,
        atol=1.0e-6,
    )
    assert int(np.asarray(restored.info["active_phase"])) == int(
        np.asarray(state.info["active_phase"])
    )
    assert int(np.asarray(restored.info["episode_step"])) == int(
        np.asarray(state.info["episode_step"])
    )
    assert int(np.asarray(restored.info["phase_episode_step"])) == int(
        np.asarray(state.info["phase_episode_step"])
    )
    assert bool(np.asarray(restored.info["expert_switching_used"])) is False
    assert bool(np.asarray(restored.info["reset_from_soft_tube"])) is True

    next_state = step(restored, jp.zeros(4, dtype=jp.float32))
    jax.block_until_ready(next_state)
    assert jax.tree.structure(next_state) == jax.tree.structure(restored)
    assert bool(np.asarray(next_state.info["expert_switching_used"])) is False
    assert _finite(next_state.data.qpos)
    assert _finite(next_state.data.qvel)
    assert _finite(next_state.metrics)
