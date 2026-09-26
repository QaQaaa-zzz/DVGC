from __future__ import annotations
import json
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jit_dvgc.handoff_snapshot import capture_snapshot, save_snapshot
from jit_dvgc.snapshot_pool import SnapshotPool
from types import SimpleNamespace
class _State:
    data=SimpleNamespace(qpos=np.arange(12,dtype=np.float32),qvel=np.arange(11,dtype=np.float32),ctrl=np.zeros(4,dtype=np.float32))
    obs={"state":np.zeros(76,dtype=np.float32)}
    info={"history":SimpleNamespace(frames=np.zeros((3,25),dtype=np.float32),valid_count=np.int32(3)),"events":SimpleNamespace(jump_signal=False,jump_zone_seen=False,jump_zone_consumed=False,ascending_seen=False,height_seen=False,apex_seen=True,stuck_anchor_x=0.,stuck_ticks=0,stuck=False,episode_step=0),"last_action":np.zeros(4,dtype=np.float32),"rng":np.array([1,2],dtype=np.uint32),"episode_step":np.int32(0)}

def _snapshot(compat):
    return capture_snapshot(_State(), config_sha256="c"*64, xml_sha256="a"*64, policy_sha256="b"*64, parent_trajectory="traj", compatibility=compat)

def test_empty_pool_rejected_and_sample_is_reproducible(tmp_path):
    compat={"timing":"fixed"};
    with pytest.raises(ValueError, match="must not be empty"):
        SnapshotPool.from_paths([], compatibility=compat)
    for i in range(2): save_snapshot(tmp_path/f"s{i}", _snapshot(compat))
    pool=SnapshotPool.from_paths([tmp_path/"s0",tmp_path/"s1"], compatibility=compat)
    a=pool.sample(jax.random.key(3)); b=pool.sample(jax.random.key(3))
    assert int(a["parent_group_index"]) == int(b["parent_group_index"])
    assert int(a["history_valid_count"]) == 3

def test_failed_bank_and_incompatible_snapshot_rejected(tmp_path):
    compat={"timing":"fixed"}; bad={"timing":"other"}
    p=tmp_path/"bank"; p.mkdir(); (p/"manifest.json").write_text(json.dumps({"status":"failed"})); (p/"index.json").write_text("[]")
    with pytest.raises(ValueError, match="closed"):
        SnapshotPool.from_closed_bank(p, compatibility=compat)
    save_snapshot(tmp_path/"bad", _snapshot(bad))
    with pytest.raises(ValueError, match="compatibility"):
        SnapshotPool.from_paths([tmp_path/"bad"], compatibility=compat)

def test_sample_is_jax_vmap_compatible(tmp_path):
    compat={"timing":"fixed"}; save_snapshot(tmp_path/"s", _snapshot(compat))
    pool=SnapshotPool.from_paths([tmp_path/"s"], compatibility=compat)
    result=jax.jit(jax.vmap(pool.sample))(jax.random.split(jax.random.key(7), 2))
    assert result["qpos"].shape == (2, 12)

def test_sampled_item_restores_and_jitted_step_matches_source(jit_root):
    import json
    from jit_dvgc.config import load_config
    from jit_dvgc.env import TwoPhaseBikeEnv
    from jit_dvgc.handoff_snapshot import compatibility_identity
    env = TwoPhaseBikeEnv(load_config(jit_root / "configs/phase_u_continuation_10m.json"))
    pool = SnapshotPool.from_closed_bank(jit_root / "runs/handoff_bank/handoff_bank_9977856_jit8", compatibility=compatibility_identity(env))
    index = 0
    source = pool.snapshot(index)
    sampled = pool.sample(jax.random.key(99)) if False else pool.sample_index(index)
    restored = env.restore_handoff_snapshot(pool.materialize(sampled))
    assert np.array_equal(np.asarray(restored.info["history"].frames), source.observation_fifo)
    assert int(restored.info["history"].valid_count) == source.history_valid_count
    assert np.array_equal(np.asarray(restored.info["last_action"]), source.last_action)
    assert int(restored.info["episode_step"]) == source.tick
    direct = env.restore_handoff_snapshot(source)
    step = jax.jit(env.step)
    action = jnp.zeros(4, dtype=jnp.float32)
    a, b = step(direct, action), step(restored, action)
    np.testing.assert_allclose(np.asarray(a.data.qpos), np.asarray(b.data.qpos), atol=1e-4, rtol=0)
    np.testing.assert_allclose(np.asarray(a.data.qvel), np.asarray(b.data.qvel), atol=1e-4, rtol=0)
    np.testing.assert_allclose(np.asarray(a.obs["state"]), np.asarray(b.obs["state"]), atol=1e-4, rtol=0)
