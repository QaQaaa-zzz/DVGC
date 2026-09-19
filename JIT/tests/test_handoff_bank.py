from __future__ import annotations
import json
import numpy as np
from types import SimpleNamespace
from jit_dvgc.handoff_bank import BankCollector, collect_rollout, pytree_sha256, select_offsets
from jit_dvgc.handoff_snapshot import capture_snapshot
class _State:
    data = SimpleNamespace(qpos=np.arange(12, dtype=np.float32), qvel=np.arange(11, dtype=np.float32), ctrl=np.zeros(4, dtype=np.float32))
    obs = {"state": np.zeros(76, dtype=np.float32)}
    info = {"history": SimpleNamespace(frames=np.zeros((3, 25), dtype=np.float32), valid_count=np.int32(3)),
            "events": SimpleNamespace(jump_signal=False, jump_zone_seen=False, jump_zone_consumed=False, ascending_seen=False, height_seen=False, apex_seen=True, stuck_anchor_x=0., stuck_ticks=0, stuck=False, episode_step=0),
            "last_action": np.zeros(4, dtype=np.float32), "rng": np.array([1, 2], dtype=np.uint32), "episode_step": np.int32(0)}

def test_select_offsets_is_bounded_and_semantic():
    selected = select_offsets(5, max_tick=8)
    assert selected == {0: "pre_apex", 4: "nearest_pre_apex", 5: "nearest_apex", 6: "post_apex"}

def test_bank_deduplicates_qpos_qvel_and_closes_manifest(tmp_path):
    collector = BankCollector(tmp_path / "bank", "smoke", "checkpoint", "c" * 64, "a" * 64, "b" * 64, 20)
    states = [_State() for _ in range(3)]
    def capture(state, parent, tick):
        return capture_snapshot(state, config_sha256="c" * 64, xml_sha256="a" * 64, policy_sha256="b" * 64, parent_trajectory=parent, parent_tick=tick)
    assert collect_rollout(collector, states, seed=4, capture=capture, parent_trajectory="traj", apex_tick=1) == 1
    manifest = collector.close()
    assert manifest["status"] == "closed"
    assert manifest["snapshot_count"] == 1
    assert json.loads((tmp_path / "bank" / "index.json").read_text())[0]["role"] == "nearest_pre_apex"

def test_failed_bank_is_closed_with_failure_reason(tmp_path):
    collector = BankCollector(tmp_path / "bank", "smoke", "checkpoint", "c" * 64, "a" * 64, "b" * 64, 1)
    manifest = collector.close(status="failed", failure="bounded rollout failed")
    assert manifest["status"] == "failed"
    assert manifest["failure"] == "bounded rollout failed"

def test_manifest_separates_global_transition_and_per_trajectory_budgets(tmp_path):
    collector = BankCollector(tmp_path / "bank", "smoke", "checkpoint", "c" * 64, "a" * 64, "b" * 64, 20000)
    collector.max_ticks = 1250
    manifest = collector.close()
    accounting = manifest["interaction_accounting"]
    assert accounting["max_transitions"] == 20000
    assert accounting["max_ticks"] == 1250

def test_pytree_policy_hash_is_complete_and_structure_sensitive():
    import jax.numpy as jp
    tree = {"a": jp.asarray([1., 2.]), "b": (jp.asarray(3, dtype=jp.int32),)}
    digest = pytree_sha256(tree)
    assert digest == pytree_sha256({"a": jp.asarray([1., 2.]), "b": (jp.asarray(3, dtype=jp.int32),)})
    assert digest != pytree_sha256({"a": jp.asarray([1., 2.001]), "b": (jp.asarray(3, dtype=jp.int32),)})
    assert digest != pytree_sha256({"a": jp.asarray([[1., 2.]]), "b": (jp.asarray(3, dtype=jp.int32),)})
    assert digest != pytree_sha256({"a": jp.asarray([1., 2.]), "b": (jp.asarray(3., dtype=jp.float32),)})
    assert digest != pytree_sha256({"a": jp.asarray([1., 2.]), "b": jp.asarray(3, dtype=jp.int32)})


def test_checkpoint_policy_first_action_matches_fixed_eval_npz(jit_root):
    import jax
    from jit_dvgc.checkpoint import CheckpointIdentity, load_checkpoint
    from jit_dvgc.config import load_config
    from jit_dvgc.constants import ACTION_ORDER, ACTOR_FRAME_FIELDS, ACTOR_TASK_FIELDS
    from jit_dvgc.env import TwoPhaseBikeEnv
    from jit_dvgc.ppo import make_checkpoint_policy

    config = load_config(jit_root / "configs" / "phase_u_continuation_10m.json")
    env = TwoPhaseBikeEnv(config)
    checkpoint = jit_root / "runs/phase_u/phase_u_v4_pitch15penalty_9977856_seed820901_20260826/checkpoints/transition_9977856"
    payload = load_checkpoint(checkpoint, expected=CheckpointIdentity(config.config_sha256, env._bundle.xml_sha256, ACTOR_FRAME_FIELDS, ACTOR_TASK_FIELDS, ACTION_ORDER))
    policy = make_checkpoint_policy(env, payload)
    state = env.reset_natural(jax.random.PRNGKey(1000001))
    action, _ = policy(state.obs, jax.random.PRNGKey(1000001))
    reference = np.load(jit_root / "runs/phase_u/phase_u_v4_pitch15penalty_9977856_seed820901_20260826/evaluations/transition_9977856/seed_1000001.npz")["action"][1]
    np.testing.assert_allclose(np.asarray(action), reference, rtol=0, atol=1e-5)
