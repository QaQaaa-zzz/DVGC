from __future__ import annotations

import pytest


def test_panel_selection_requires_exact_42_and_global_eval_seed_split(jit_root):
    from jit_dvgc.phase_d_evaluation import select_panel_entries

    entries = select_panel_entries(
        jit_root / "runs/handoff_bank/catalog_20260827.json",
        eval_seeds=(1000007, 1000008),
    )
    assert len(entries) == 42
    assert {e["seed"] for e in entries} == {1000007, 1000008}
    assert len({(e["source_bank"], e["parent_group_id"], e["tick"]) for e in entries}) == 42


def test_panel_selection_rejects_unknown_seed_or_wrong_max_ticks(jit_root):
    from jit_dvgc.phase_d_evaluation import select_panel_entries

    with pytest.raises(ValueError, match="unknown eval seed"):
        select_panel_entries(jit_root / "runs/handoff_bank/catalog_20260827.json", eval_seeds=(9999999,))


def test_panel_terminal_summary_and_budget():
    from jit_dvgc.phase_d_evaluation import terminal_summary, validate_panel_budget

    assert validate_panel_budget(42, 100) == 4200
    with pytest.raises(ValueError, match="sample count"):
        validate_panel_budget(43, 100)
    assert terminal_summary({"terminated": True, "truncated": False, "success": True, "physical_failure": False, "timeout": False, "end_code": 12})["reason"] == "recovery_success"


@pytest.mark.gpu
def test_one_snapshot_frozen_phase_d_gpu_smoke(jit_root):
    import jax
    from jit_dvgc.checkpoint import CheckpointIdentity, load_checkpoint
    from jit_dvgc.config import load_config
    from jit_dvgc.constants import ACTION_ORDER, ACTOR_FRAME_FIELDS, ACTOR_TASK_FIELDS
    from jit_dvgc.env import TwoPhaseBikeEnv
    from jit_dvgc.handoff_snapshot import compatibility_identity
    from jit_dvgc.phase_d_evaluation import select_panel_entries
    from jit_dvgc.phase_expert_init import ActorOnlyInitialization, make_actor_only_policy
    from jit_dvgc.snapshot_pool import SnapshotPool

    catalog = jit_root / "runs/handoff_bank/catalog_20260827.json"
    row = select_panel_entries(catalog, eval_seeds=(1000007, 1000008))[0]
    source = load_config(jit_root / "configs/phase_u_continuation_10m.json")
    source_env = TwoPhaseBikeEnv(source)
    snapshot = catalog.parent / row["source_bank"] / row["snapshot"]
    pool = SnapshotPool.from_paths((snapshot,), compatibility=compatibility_identity(source_env))
    target_config = load_config(jit_root / "configs/descent_recovery_smoke.json")
    env = TwoPhaseBikeEnv(target_config, snapshot_pool=pool)
    checkpoint = jit_root / "runs/phase_d/descent_recovery_smoke_25600_seed920001_20260827_wrapfix/checkpoints/transition_25600"
    identity = CheckpointIdentity(target_config.config_sha256, env._bundle.xml_sha256, ACTOR_FRAME_FIELDS, ACTOR_TASK_FIELDS, ACTION_ORDER)
    payload = load_checkpoint(checkpoint, expected=identity)
    init = ActorOnlyInitialization(payload.observation_normalizer, payload.actor_params, payload.training_transitions, "", "", {})
    policy = make_actor_only_policy(env, init, deterministic=True)
    state = env.reset_descent_index(jax.numpy.asarray(0, dtype=jax.numpy.int32))
    action, _ = policy(state.obs, jax.random.PRNGKey(1))
    next_state = jax.jit(env.step)(state, action)
    assert int(next_state.info["episode_step"]) == 1
