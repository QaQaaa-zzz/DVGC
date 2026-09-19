from __future__ import annotations

import numpy as np
import jax
from jax import numpy as jp
import json
import pytest

from jit_dvgc.handoff_snapshot import (
    HandoffSnapshot,
    capture_snapshot,
    load_snapshot,
    restore_snapshot,
    save_snapshot,
)


class _Data:
    qpos = np.arange(12, dtype=np.float32)
    qvel = np.arange(11, dtype=np.float32) + 20
    ctrl = np.arange(4, dtype=np.float32) + 40


class _State:
    data = _Data()
    obs = {"state": np.arange(76, dtype=np.float32)}
    info = {
        "history": type("H", (), {"frames": np.arange(75, dtype=np.float32).reshape(3, 25), "valid_count": np.int32(3)})(),
        "events": type("E", (), {"jump_signal": True, "jump_zone_seen": True, "jump_zone_consumed": False, "ascending_seen": True, "height_seen": True, "apex_seen": False, "stuck_anchor_x": 1.0, "stuck_ticks": 2, "stuck": False, "episode_step": 17})(),
        "last_action": np.ones(4, dtype=np.float32),
        "rng": np.asarray([7, 9], dtype=np.uint32),
        "episode_step": np.int32(17),
    }


def test_snapshot_save_load_preserves_state_and_provenance(tmp_path):
    snapshot = capture_snapshot(_State(), config_sha256="c" * 64, xml_sha256="a" * 64,
                                policy_sha256="b" * 64, parent_trajectory="traj-7",
                                parent_tick=11, policy_identity="pi-up")
    assert isinstance(snapshot, HandoffSnapshot)
    assert snapshot.tick == 17
    assert snapshot.history_valid_count == 3
    np.testing.assert_array_equal(snapshot.rng, _State.info["rng"])
    np.testing.assert_array_equal(snapshot.qpos, _Data.qpos)
    np.testing.assert_array_equal(snapshot.observation_fifo, _State.info["history"].frames)
    save_snapshot(tmp_path / "snap", snapshot)
    restored = load_snapshot(tmp_path / "snap")
    assert restored == snapshot


def test_snapshot_rejects_payload_and_sidecar_tampering(tmp_path):
    snapshot = capture_snapshot(_State(), config_sha256="c" * 64, xml_sha256="a" * 64,
                                policy_sha256="b" * 64, parent_trajectory="traj-7",
                                parent_tick=11, policy_identity="pi-up")
    save_snapshot(tmp_path / "snap", snapshot)
    payload = tmp_path / "snap" / "snapshot.pkl"
    payload.write_bytes(payload.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="payload_sha256"):
        load_snapshot(tmp_path / "snap")

    save_snapshot(tmp_path / "snap2", snapshot)
    sidecar = tmp_path / "snap2" / "identity.json"
    identity = json.loads(sidecar.read_text())
    identity["policy_identity"] = "wrong"
    sidecar.write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="policy_identity"):
        load_snapshot(tmp_path / "snap2")


def test_restore_snapshot_round_trip_is_explicitly_next_step_ready():
    snapshot = capture_snapshot(_State(), config_sha256="c" * 64, xml_sha256="a" * 64,
                                policy_sha256="b" * 64, parent_trajectory="traj-7")
    # The pure restore path is useful to callers that already own a state template.
    target = restore_snapshot(snapshot)
    np.testing.assert_array_equal(target["qpos"], snapshot.qpos)
    np.testing.assert_array_equal(target["qvel"], snapshot.qvel)
    np.testing.assert_array_equal(target["ctrl"], snapshot.ctrl)
    np.testing.assert_array_equal(target["last_action"], snapshot.last_action)
    assert target["tick"] == snapshot.tick


def test_online_restore_has_same_next_step_dynamics(jit_root):
    from jit_dvgc.config import load_config
    from jit_dvgc.env import TwoPhaseBikeEnv

    env = TwoPhaseBikeEnv(load_config(jit_root / "configs" / "phase_u_smoke.json"))
    state = env.reset_natural(jax.random.key(19))
    action = jp.asarray([0.1, -0.2, 0.05, 0.0], dtype=jp.float32)
    state = env.step(state, action)
    snapshot = env.capture_handoff_snapshot(
        state, policy_sha256="b" * 64, parent_trajectory="online-test"
    )
    restored = env.restore_handoff_snapshot(snapshot)
    expected = env.step(state, action)
    actual = env.step(restored, action)
    np.testing.assert_allclose(expected.data.qpos, actual.data.qpos, rtol=0, atol=1e-6)
    np.testing.assert_allclose(expected.data.qvel, actual.data.qvel, rtol=0, atol=1e-4)
    np.testing.assert_allclose(expected.obs["state"], actual.obs["state"], rtol=0, atol=1e-4)


def test_restore_uses_runtime_contract_not_phase_reward_or_reset_hash(jit_root):
    from jit_dvgc.config import load_config
    from jit_dvgc.env import TwoPhaseBikeEnv
    env = TwoPhaseBikeEnv(load_config(jit_root / "configs" / "phase_u_smoke.json"))
    state = env.reset_natural(jax.random.key(21))
    snapshot = env.capture_handoff_snapshot(state, policy_sha256="b" * 64, parent_trajectory="contract")
    # Source config provenance may differ at a phase handoff; runtime contract does not.
    snapshot.config_sha256 = "d" * 64
    assert env.restore_handoff_snapshot(snapshot) is not None


@pytest.mark.parametrize("field", ["xml_sha256", "actor_frame_fields", "action_order", "sim_dt"])
def test_restore_rejects_incompatible_runtime_contract(jit_root, field):
    from jit_dvgc.config import load_config
    from jit_dvgc.env import TwoPhaseBikeEnv
    env = TwoPhaseBikeEnv(load_config(jit_root / "configs" / "phase_u_smoke.json"))
    state = env.reset_natural(jax.random.key(22))
    snapshot = env.capture_handoff_snapshot(state, policy_sha256="b" * 64, parent_trajectory="contract")
    bad = dict(snapshot.compatibility_identity)
    bad[field] = "incompatible" if field != "sim_dt" else 0.123
    snapshot.compatibility_identity = bad
    with pytest.raises(ValueError, match="compatibility"):
        env.restore_handoff_snapshot(snapshot)
