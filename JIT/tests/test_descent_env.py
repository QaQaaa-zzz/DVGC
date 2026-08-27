from __future__ import annotations
import jax
import pytest
import jax
from jit_dvgc.config import load_config
from jit_dvgc.env import TwoPhaseBikeEnv

def test_descent_environment_requires_snapshot_pool(jit_root):
    config=load_config(jit_root/"configs/descent_recovery_smoke.json")
    with pytest.raises(ValueError, match="snapshot pool"):
        TwoPhaseBikeEnv(config, convert_model=False)

@pytest.mark.gpu
def test_descent_pool_vmap_reset_step_and_source_contract(jit_root):
    import jax.numpy as jp
    import numpy as np
    from jit_dvgc.handoff_snapshot import compatibility_identity
    from jit_dvgc.snapshot_pool import SnapshotPool
    source_config = load_config(jit_root / "configs/phase_u_continuation_10m.json")
    source_env = TwoPhaseBikeEnv(source_config)
    pool = SnapshotPool.from_closed_bank(jit_root / "runs/handoff_bank/handoff_bank_9977856_jit8", compatibility=compatibility_identity(source_env))
    env = TwoPhaseBikeEnv(load_config(jit_root / "configs/descent_recovery_smoke.json"), snapshot_pool=pool)
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))
    states = reset(jax.random.split(jax.random.key(700), 4))
    assert states.obs["state"].shape == (4, 76)
    assert states.obs["privileged_state"].shape == (4, 106)
    assert np.all(np.isfinite(np.asarray(states.data.qpos)))
    assert np.all(np.asarray(states.info["episode_step"]) == 0)
    assert np.all(np.asarray(states.info["parent_group_index"]) >= 0)
    assert np.all(np.isin(np.asarray(states.info["source_tick"]), [s.tick for s in pool.snapshots]))
    next_states = step(states, jp.zeros((4, 4), dtype=jp.float32))
    assert np.all(np.asarray(next_states.info["episode_step"]) == 1)
    assert np.all(np.isfinite(np.asarray(next_states.data.qpos)))
    source = pool.snapshot(0)
    restored = env.restore_handoff_snapshot(source)
    print("source_restore_max_abs", {
        "qpos": float(np.max(np.abs(np.asarray(restored.data.qpos) - source.qpos))),
        "qvel": float(np.max(np.abs(np.asarray(restored.data.qvel) - source.qvel))),
        "fifo": float(np.max(np.abs(np.asarray(restored.info["history"].frames) - source.observation_fifo))),
        "actor_obs": float(np.max(np.abs(np.asarray(restored.obs["state"]) - source.observation))),
        "last_action": float(np.max(np.abs(np.asarray(restored.info["last_action"]) - source.last_action))),
    })
    np.testing.assert_array_equal(np.asarray(restored.info["history"].frames), source.observation_fifo)
    assert int(restored.info["history"].valid_count) == source.history_valid_count
    np.testing.assert_allclose(np.asarray(restored.obs["state"]), source.observation, rtol=0.0, atol=1e-6)

    # A one-item pool makes the sampled reset index deterministic and checks
    # the Phase D reset contract against the exact source snapshot.
    single_pool = SnapshotPool((source,), (pool.parent_group_ids[0],), pool.compatibility)
    single_env = TwoPhaseBikeEnv(load_config(jit_root / "configs/descent_recovery_smoke.json"), snapshot_pool=single_pool)
    sampled = single_env.reset(jax.random.key(701))
    print("descent_reset_max_abs", {
        "qpos": float(np.max(np.abs(np.asarray(sampled.data.qpos) - source.qpos))),
        "qvel": float(np.max(np.abs(np.asarray(sampled.data.qvel) - source.qvel))),
        "fifo": float(np.max(np.abs(np.asarray(sampled.info["history"].frames) - source.observation_fifo))),
        "actor_obs": float(np.max(np.abs(np.asarray(sampled.obs["state"]) - source.observation))),
        "last_action": float(np.max(np.abs(np.asarray(sampled.info["last_action"]) - source.last_action))),
    })
    np.testing.assert_allclose(np.asarray(sampled.data.qpos), source.qpos, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(sampled.data.qvel), source.qvel, rtol=0.0, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(sampled.info["history"].frames), source.observation_fifo)
    assert int(sampled.info["history"].valid_count) == source.history_valid_count
    np.testing.assert_allclose(np.asarray(sampled.info["last_action"]), source.last_action, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(sampled.obs["state"]), source.observation, rtol=0.0, atol=1e-6)
    assert int(sampled.info["source_tick"]) == source.tick
    assert int(sampled.info["episode_step"]) == 0
