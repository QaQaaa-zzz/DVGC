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
        config=target_config,
        environment=env,
        num_envs=4,
        unroll_length=4,
        batch_size=4,
        num_minibatches=1,
        num_eval_envs=4,
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


def test_phase_d_trainer_kwargs_always_use_jit_wrapper():
    from jit_dvgc.phase_d_smoke import build_phase_d_trainer_kwargs
    from jit_dvgc.ppo import wrap_for_jit_training

    class Init:
        restore_params = ("normalizer", "actor")

    kwargs = build_phase_d_trainer_kwargs(Init(), num_timesteps=64)
    assert kwargs["wrap_env"] is True
    assert kwargs["wrap_env_fn"] is wrap_for_jit_training
    assert kwargs["restore_value_fn"] is False


def test_phase_d_smoke_rejects_formal_and_requires_input():
    from jit_dvgc.phase_d_smoke import validate_phase_d_smoke_args

    with pytest.raises(ValueError, match="formal"):
        validate_phase_d_smoke_args(formal=True, snapshot_bank=None, snapshot_catalog=None, actor_init_checkpoint=None, actor_init_config=None, eval_seeds=())
    with pytest.raises(ValueError, match="snapshot"):
        validate_phase_d_smoke_args(formal=False, snapshot_bank=None, snapshot_catalog=None, actor_init_checkpoint="x", actor_init_config="y", eval_seeds=(1,))


def test_catalog_split_is_seed_global_and_rejects_unknown_or_empty(jit_root):
    from jit_dvgc.phase_d_smoke import split_catalog_entries

    catalog = jit_root / "runs/handoff_bank/catalog_20260827.json"
    train, evaluation, metadata = split_catalog_entries(catalog, eval_seeds=(1000007, 1000008))
    assert train and evaluation
    assert {row["seed"] for row in train} == set(range(1000001, 1000007))
    assert {row["seed"] for row in evaluation} == {1000007, 1000008}
    assert {row["parent_group_id"] for row in train}.isdisjoint(
        {row["parent_group_id"] for row in evaluation}
    )
    assert metadata["entry_count"] == 168
    with pytest.raises(ValueError, match="unknown eval seed"):
        split_catalog_entries(catalog, eval_seeds=(9999999,))
    with pytest.raises(ValueError, match="train pool is empty"):
        split_catalog_entries(catalog, eval_seeds=tuple(sorted({row["seed"] for row in __import__('json').loads(catalog.read_text())["entries"]})))


def test_catalog_hash_changes_when_catalog_is_tampered(tmp_path, jit_root):
    from jit_dvgc.phase_d_smoke import catalog_sha256

    source = jit_root / "runs/handoff_bank/catalog_20260827.json"
    copy = tmp_path / "catalog.json"
    copy.write_bytes(source.read_bytes())
    original = catalog_sha256(copy)
    copy.write_bytes(copy.read_bytes() + b"\n")
    assert catalog_sha256(copy) != original
