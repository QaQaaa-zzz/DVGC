from __future__ import annotations

import pytest


@pytest.mark.gpu
def test_phase_d_gpu_compile_update_uses_actor_only_restore(jit_root):
    """Small compile/update gate; deliberately below the real smoke budget."""
    import jax
    from jit_dvgc.config import load_config
    from jit_dvgc.env import TwoPhaseBikeEnv
    from jit_dvgc.handoff_snapshot import compatibility_identity
    from jit_dvgc.phase_expert_init import build_actor_only_initialization
    from jit_dvgc.phase_d_smoke import build_phase_d_trainer_kwargs
    from jit_dvgc.ppo import make_network_factory, wrap_for_jit_training
    from jit_dvgc.snapshot_pool import SnapshotPool
    from brax.training.agents.ppo import train as ppo_train

    source_config = load_config(jit_root / "configs/phase_u_continuation_10m.json")
    target_config = load_config(jit_root / "configs/descent_recovery_smoke.json")
    source_env = TwoPhaseBikeEnv(source_config)
    pool = SnapshotPool.from_closed_bank(
        jit_root / "runs/handoff_bank/handoff_bank_9977856_jit8",
        compatibility=compatibility_identity(source_env),
    )
    env = TwoPhaseBikeEnv(target_config, snapshot_pool=pool)
    initialization = build_actor_only_initialization(
        jit_root / "runs/phase_u/phase_u_v4_pitch15penalty_9977856_seed820901_20260826/checkpoints/transition_9977856",
        source_config=source_config,
        target_env=env,
    )
    kwargs = build_phase_d_trainer_kwargs(
        initialization,
        num_timesteps=64,
        environment=env,
        max_devices_per_host=1,
        wrap_env=True,
        wrap_env_fn=wrap_for_jit_training,
        num_envs=4,
        episode_length=25,
        action_repeat=1,
        learning_rate=target_config.ppo.learning_rate,
        entropy_cost=target_config.ppo.entropy_cost,
        discounting=target_config.ppo.discounting,
        unroll_length=4,
        batch_size=4,
        num_minibatches=1,
        num_updates_per_batch=1,
        normalize_observations=True,
        reward_scaling=target_config.ppo.reward_scaling,
        clipping_epsilon=target_config.ppo.clipping_epsilon,
        gae_lambda=target_config.ppo.gae_lambda,
        max_grad_norm=target_config.ppo.max_grad_norm,
        bootstrap_on_timeout=True,
        network_factory=make_network_factory(),
        seed=target_config.ppo.seed,
        num_evals=0,
        num_eval_envs=4,
        deterministic_eval=True,
        run_evals=False,
    )
    _, params, _ = ppo_train.train(**kwargs)
    assert len(params) == 3


def test_phase_d_parent_group_split_is_disjoint():
    from jit_dvgc.phase_d_smoke import validate_parent_group_split

    assert validate_parent_group_split(("a", "b"), ("c",)) is True
    with pytest.raises(ValueError, match="parent_group"):
        validate_parent_group_split(("a", "b"), ("b", "c"))


def test_phase_d_trainer_kwargs_are_actor_only():
    from jit_dvgc.phase_d_smoke import build_phase_d_trainer_kwargs

    class Init:
        restore_params = ("normalizer", "actor")

    kwargs = build_phase_d_trainer_kwargs(Init(), num_timesteps=64)
    assert kwargs["restore_params"] == ("normalizer", "actor")
    assert kwargs["restore_value_fn"] is False
    assert kwargs["num_timesteps"] == 64


def test_phase_d_smoke_rejects_formal_and_requires_input():
    from jit_dvgc.phase_d_smoke import validate_phase_d_smoke_args

    with pytest.raises(ValueError, match="formal"):
        validate_phase_d_smoke_args(formal=True, snapshot_bank=None, snapshot_catalog=None, actor_init_checkpoint=None, actor_init_config=None)
    with pytest.raises(ValueError, match="snapshot"):
        validate_phase_d_smoke_args(formal=False, snapshot_bank=None, snapshot_catalog=None, actor_init_checkpoint="x", actor_init_config="y")
