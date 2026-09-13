from __future__ import annotations

from dataclasses import replace

import jax
import numpy as np
import pytest


CHECKPOINT_RELATIVE = (
    "runs/phase_u/phase_u_v4_pitch15penalty_9977856_seed820901_20260826"
    "/checkpoints/transition_9977856"
)


def _configs(jit_root):
    from jit_dvgc.config import load_config

    source = load_config(jit_root / "configs/phase_u_continuation_10m.json")
    target = load_config(jit_root / "configs/descent_recovery_smoke.json")
    return source, target


@pytest.mark.gpu
def test_actor_only_initialization_matches_parent_and_has_fresh_value_optimizer(jit_root):
    from jit_dvgc.env import TwoPhaseBikeEnv
    from jit_dvgc.checkpoint import CheckpointIdentity, load_checkpoint
    from jit_dvgc.constants import ACTION_ORDER, ACTOR_FRAME_FIELDS, ACTOR_TASK_FIELDS
    from jit_dvgc.ppo import make_checkpoint_policy
    from jit_dvgc.phase_expert_init import (
        build_actor_only_initialization,
        make_actor_only_policy,
        trainer_kwargs_for_actor_only,
    )

    source_config, target_config = _configs(jit_root)
    source_env = TwoPhaseBikeEnv(source_config)
    target_env = TwoPhaseBikeEnv(target_config, snapshot_pool=_one_item_pool(jit_root, source_env))
    result = build_actor_only_initialization(
        jit_root / CHECKPOINT_RELATIVE, source_config=source_config, target_env=target_env
    )
    assert result.provenance == {
        "actor_initialized": True,
        "critic_fresh": True,
        "optimizer_fresh": True,
    }
    assert not hasattr(result, "critic_params")
    assert result.parent_transition == 9_977_856
    assert result.payload_sha256
    assert result.actor_sha256
    assert result.restore_params[0] is result.observation_normalizer
    assert result.restore_params[1] is result.actor_params
    kwargs = trainer_kwargs_for_actor_only(result)
    assert kwargs["restore_value_fn"] is False
    assert "critic_params" not in kwargs

    source_state = source_env.reset_natural(jax.random.PRNGKey(1000001))
    parent_policy = make_actor_only_policy(source_env, result)
    action, _ = parent_policy(source_state.obs, jax.random.PRNGKey(1000001))
    payload = load_checkpoint(
        jit_root / CHECKPOINT_RELATIVE,
        expected=CheckpointIdentity(
            source_config.config_sha256,
            source_env._bundle.xml_sha256,
            ACTOR_FRAME_FIELDS,
            ACTOR_TASK_FIELDS,
            ACTION_ORDER,
        ),
    )
    reference_action, _ = make_checkpoint_policy(source_env, payload)(
        source_state.obs, jax.random.PRNGKey(1000001)
    )
    assert np.all(np.isfinite(np.asarray(action)))
    np.testing.assert_allclose(np.asarray(action), np.asarray(reference_action), rtol=0.0, atol=1e-6)


@pytest.mark.gpu
def test_actor_only_initialization_rejects_wrong_source_config_or_target_contract(jit_root):
    from jit_dvgc.env import TwoPhaseBikeEnv
    from jit_dvgc.phase_expert_init import build_actor_only_initialization

    source_config, target_config = _configs(jit_root)
    source_env = TwoPhaseBikeEnv(source_config)
    target_env = TwoPhaseBikeEnv(target_config, snapshot_pool=_one_item_pool(jit_root, source_env))
    checkpoint = jit_root / CHECKPOINT_RELATIVE
    with pytest.raises(ValueError, match="config"):
        build_actor_only_initialization(
            checkpoint,
            source_config=replace(source_config, config_sha256="0" * 64),
            target_env=target_env,
        )
    bad_model = dict(source_config.model)
    bad_model["xml_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="XML|xml"):
        build_actor_only_initialization(
            checkpoint,
            source_config=replace(source_config, model=bad_model),
            target_env=target_env,
        )
    bad_action = replace(target_config.action, base_rear_speed=target_config.action.base_rear_speed + 1)
    bad_target = TwoPhaseBikeEnv(replace(target_config, action=bad_action), snapshot_pool=_one_item_pool(jit_root, source_env))
    with pytest.raises(ValueError, match="compatibility"):
        build_actor_only_initialization(
            checkpoint, source_config=source_config, target_env=bad_target
        )


def _one_item_pool(jit_root, source_env):
    from jit_dvgc.handoff_snapshot import compatibility_identity
    from jit_dvgc.snapshot_pool import SnapshotPool

    return SnapshotPool.from_closed_bank(
        jit_root / "runs/handoff_bank/handoff_bank_9977856_jit8",
        compatibility=compatibility_identity(source_env),
    )
