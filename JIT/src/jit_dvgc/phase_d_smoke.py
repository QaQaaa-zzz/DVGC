"""Small, auditable Phase D engineering-smoke training link.

This module intentionally owns orchestration only.  Physics, observation and
reward semantics remain in the existing environment modules.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

import jax
import numpy as np
from brax.training.agents.ppo import train as ppo_train

from .checkpoint import CheckpointIdentity, CheckpointPayload, load_checkpoint, save_checkpoint
from .config import load_config
from .constants import ACTION_ORDER, ACTOR_FRAME_FIELDS, ACTOR_TASK_FIELDS
from .env import TwoPhaseBikeEnv
from .handoff_snapshot import compatibility_identity
from .phase_expert_init import build_actor_only_initialization, make_actor_only_policy
from .ppo import make_network_factory
from .provenance import InteractionAccounting, RunDeclaration, close_run, mark_run_running, predeclare_run
from .snapshot_pool import SnapshotPool


def catalog_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_bank_hashes(catalog_path: Path, entries) -> dict[str, dict[str, str]]:
    result = {}
    for bank in sorted({row["source_bank"] for row in entries}):
        bank_path = Path(catalog_path).parent / bank
        manifest_hash = catalog_sha256(bank_path / "manifest.json")
        identities = []
        for row in entries:
            if row["source_bank"] == bank:
                identities.append(catalog_sha256(bank_path / row["snapshot"] / "identity.json"))
        result[bank] = {
            "manifest_sha256": manifest_hash,
            "snapshot_identity_sha256": hashlib.sha256("".join(sorted(identities)).encode()).hexdigest(),
        }
    return result


def split_catalog_entries(path: Path, eval_seeds) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Split globally by seed, so a parent group cannot cross train/eval."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = list(payload.get("entries", []))
    if not entries:
        raise ValueError("snapshot catalog is empty")
    eval_set = {int(seed) for seed in eval_seeds}
    known = {int(row["seed"]) for row in entries}
    unknown = eval_set - known
    if unknown:
        raise ValueError(f"unknown eval seed(s): {sorted(unknown)}")
    train = [row for row in entries if int(row["seed"]) not in eval_set]
    evaluation = [row for row in entries if int(row["seed"]) in eval_set]
    if not train:
        raise ValueError("train pool is empty")
    if not evaluation:
        raise ValueError("eval pool is empty")
    train_groups = {row["parent_group_id"] for row in train}
    eval_groups = {row["parent_group_id"] for row in evaluation}
    validate_parent_group_split(train_groups, eval_groups)
    return train, evaluation, {
        "catalog_sha256": catalog_sha256(path),
        "entry_count": len(entries),
        "train_seeds": sorted({int(row["seed"]) for row in train}),
        "eval_seeds": sorted(eval_set),
        "train_parent_groups": sorted(train_groups),
        "eval_parent_groups": sorted(eval_groups),
        "source_bank_hashes": _source_bank_hashes(path, entries),
        "diagnostic_source": {key: evaluation[0].get(key) for key in ("seed", "parent_group_id", "role", "tick", "source_bank")},
    }


def validate_parent_group_split(train_groups, eval_groups) -> bool:
    overlap = set(train_groups).intersection(eval_groups)
    if overlap:
        raise ValueError(f"parent_group split overlap: {sorted(overlap)}")
    return True


def validate_phase_d_smoke_args(*, formal: bool, snapshot_bank, snapshot_catalog,
                                actor_init_checkpoint, actor_init_config, eval_seeds=()) -> None:
    if formal:
        raise ValueError("Phase D formal training is not implemented")
    if snapshot_bank is None and snapshot_catalog is None:
        raise ValueError("Phase D smoke requires a snapshot bank or catalog")
    if not tuple(eval_seeds):
        raise ValueError("Phase D smoke requires --eval-seeds")
    if actor_init_checkpoint is None or actor_init_config is None:
        raise ValueError("Phase D smoke requires actor initialization checkpoint and config")


def build_phase_d_trainer_kwargs(initialization: Any, *, num_timesteps: int, **kwargs) -> dict[str, Any]:
    result = dict(kwargs)
    result.update({
        "num_timesteps": int(num_timesteps),
        "restore_params": initialization.restore_params,
        "restore_value_fn": False,
    })
    return result


def pool_from_input(path: Path, *, compatibility: Mapping[str, Any]) -> SnapshotPool:
    path = Path(path)
    if path.name == "catalog.json" or path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = payload.get("entries", payload if isinstance(payload, list) else [])
        if not entries:
            raise ValueError("snapshot catalog is empty")
        paths = [path.parent / row["source_bank"] / row["snapshot"] for row in entries]
        return SnapshotPool.from_paths(paths, compatibility=compatibility)
    return SnapshotPool.from_closed_bank(path, compatibility=compatibility)


def split_input_pools(path: Path, *, eval_seeds, compatibility: Mapping[str, Any]):
    """Load two pools with a seed-global train/eval split."""
    path = Path(path)
    if path.suffix == ".json":
        train, evaluation, metadata = split_catalog_entries(path, eval_seeds)
        train_paths = [path.parent / row["source_bank"] / row["snapshot"] for row in train]
        eval_paths = [path.parent / row["source_bank"] / row["snapshot"] for row in evaluation]
    else:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "closed":
            raise ValueError("snapshot bank must be closed")
        rows = json.loads((path / "index.json").read_text(encoding="utf-8"))
        known = {int(row["seed"]) for row in rows}
        eval_set = {int(seed) for seed in eval_seeds}
        if eval_set - known:
            raise ValueError(f"unknown eval seed(s): {sorted(eval_set - known)}")
        train = [row for row in rows if int(row["seed"]) not in eval_set]
        evaluation = [row for row in rows if int(row["seed"]) in eval_set]
        if not train:
            raise ValueError("train pool is empty")
        if not evaluation:
            raise ValueError("eval pool is empty")
        validate_parent_group_split(
            {row.get("parent_group_id", f"{row.get('parent_trajectory')}__{row['seed']}") for row in train},
            {row.get("parent_group_id", f"{row.get('parent_trajectory')}__{row['seed']}") for row in evaluation},
        )
        train_paths = [path / row["snapshot"] for row in train]
        eval_paths = [path / row["snapshot"] for row in evaluation]
        metadata = {
            "catalog_sha256": None,
            "entry_count": len(rows),
            "train_seeds": sorted({int(row["seed"]) for row in train}),
            "eval_seeds": sorted(eval_set),
            "train_parent_groups": sorted({row.get("parent_group_id", row.get("parent_trajectory")) for row in train}),
            "eval_parent_groups": sorted({row.get("parent_group_id", row.get("parent_trajectory")) for row in evaluation}),
            "source_bank_hashes": {
                path.name: {"manifest_sha256": catalog_sha256(path / "manifest.json"),
                            "snapshot_identity_sha256": hashlib.sha256("".join(sorted(
                                catalog_sha256(path / row["snapshot"] / "identity.json") for row in rows
                            )).encode()).hexdigest()}
            },
            "diagnostic_source": {key: evaluation[0].get(key) for key in ("seed", "parent_group_id", "role", "tick")},
        }
    return (SnapshotPool.from_paths(train_paths, compatibility=compatibility),
            SnapshotPool.from_paths(eval_paths, compatibility=compatibility), metadata)


def _target_identity(config, env):
    return CheckpointIdentity(config.config_sha256, env._bundle.xml_sha256,
                              ACTOR_FRAME_FIELDS, ACTOR_TASK_FIELDS, ACTION_ORDER)


def run_phase_d_smoke(
    config_path: Path,
    run_id: str,
    *,
    snapshot_bank: Path | None = None,
    snapshot_catalog: Path | None = None,
    actor_init_checkpoint: Path,
    actor_init_config: Path,
    eval_seeds: tuple[int, ...] = (),
    run_root: Path | None = None,
    trainer: Callable[..., Any] = ppo_train.train,
    env_factory: Callable[..., Any] = TwoPhaseBikeEnv,
) -> dict[str, Any]:
    validate_phase_d_smoke_args(formal=False, snapshot_bank=snapshot_bank,
                                snapshot_catalog=snapshot_catalog,
                                actor_init_checkpoint=actor_init_checkpoint,
                                actor_init_config=actor_init_config,
                                eval_seeds=eval_seeds)
    config = load_config(Path(config_path))
    if config.phase != "descent_recovery" or config.formal is not None:
        raise ValueError("Phase D smoke requires a non-formal descent_recovery config")
    source_config = load_config(Path(actor_init_config))
    # Build a source-compatible pool before constructing the target environment.
    source_env = TwoPhaseBikeEnv(source_config)
    input_path = snapshot_catalog or snapshot_bank
    train_pool, eval_pool, split_metadata = split_input_pools(
        input_path, eval_seeds=eval_seeds, compatibility=compatibility_identity(source_env)
    )
    env = env_factory(config, snapshot_pool=train_pool)
    eval_env = env_factory(config, snapshot_pool=eval_pool)
    initialization = build_actor_only_initialization(actor_init_checkpoint,
                                                     source_config=source_config,
                                                     target_env=env)
    root = Path(run_root) if run_root is not None else Path(os.environ.get("JIT_RUN_ROOT", "JIT/runs/phase_d"))
    run_dir = root / run_id
    declaration = RunDeclaration(
        run_id=run_id, purpose="descent_recovery_engineering_smoke", output_dir=run_dir,
        config_sha256=config.config_sha256, xml_sha256=env._bundle.xml_sha256,
        reference_sha256=str(config.model["reference_sha256"]),
        training_transition_ceiling=config.ppo.requested_transitions,
        stopping_conditions=("stop_after_one_declared_ppo_block", "stop_on_terminal_diagnostic",
                             "stop_on_nonfinite_metric", "stop_on_checkpoint_restore_failure"),
        resume_command=("train_phase_expert.py --phase descent_recovery --smoke "
                        f"--config {config_path} --run-id {run_id} "
                        f"--eval-seeds {' '.join(str(seed) for seed in eval_seeds)}"),
    )
    predeclare_run(declaration, resolved_config=config.raw)
    (run_dir / "phase_d_provenance.json").write_text(json.dumps({
        "snapshot_input": str(input_path), "actor_init_checkpoint": str(Path(actor_init_checkpoint).resolve()),
        "actor_init_config": str(Path(actor_init_config).resolve()),
        "parent_transition": initialization.parent_transition,
        "parent_payload_sha256": initialization.payload_sha256,
        "parent_actor_sha256": initialization.actor_sha256,
        **dict(initialization.provenance), "train_eval_split": "parent_group_id only",
        **split_metadata,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    mark_run_running(run_dir, process_id=os.getpid(), metadata={"phase": "descent_recovery"})
    try:
        kwargs = build_phase_d_trainer_kwargs(
            initialization,
            num_timesteps=config.ppo.requested_transitions,
            environment=env,
            max_devices_per_host=1, wrap_env=True, num_envs=config.ppo.num_parallel_envs,
            episode_length=config.ppo.episode_horizon, action_repeat=1,
            learning_rate=config.ppo.learning_rate, entropy_cost=config.ppo.entropy_cost,
            discounting=config.ppo.discounting, unroll_length=config.ppo.unroll_length,
            batch_size=config.ppo.batch_size, num_minibatches=config.ppo.num_minibatches,
            num_updates_per_batch=config.ppo.num_updates_per_batch, normalize_observations=True,
            reward_scaling=config.ppo.reward_scaling, clipping_epsilon=config.ppo.clipping_epsilon,
            gae_lambda=config.ppo.gae_lambda, max_grad_norm=config.ppo.max_grad_norm,
            bootstrap_on_timeout=True, network_factory=make_network_factory(), seed=config.ppo.seed,
            num_evals=0, num_eval_envs=config.ppo.num_eval_envs, deterministic_eval=True,
            run_evals=False,
        )
        make_policy, params, metrics = trainer(**kwargs)
        normalizer, actor, critic = params
        identity = _target_identity(config, env)
        checkpoint = run_dir / f"checkpoints/transition_{config.ppo.requested_transitions}"
        save_checkpoint(checkpoint, CheckpointPayload(identity, config.ppo.requested_transitions,
                                                      normalizer, actor, critic))
        restored = load_checkpoint(checkpoint, expected=identity)
        # Diagnostic starts from one fixed source item and stops on the first true terminal.
        state = eval_env.restore_handoff_snapshot(eval_pool.snapshot(0))
        policy = make_actor_only_policy(eval_env, initialization, deterministic=True)
        transitions = 0
        while transitions < config.ppo.episode_horizon and not bool(np.asarray(state.done)):
            action, _ = policy(state.obs, jax.random.PRNGKey(1000000 + transitions))
            state = eval_env.step(state, action)
            transitions += 1
            if bool(np.asarray(state.info["terminated"])) or bool(np.asarray(state.info["truncated"])):
                break
        report = {"status": "completed", "training_transitions": config.ppo.requested_transitions,
                  "diagnostic_transitions": transitions, "checkpoint": str(checkpoint),
                  "restored": restored.training_transitions == config.ppo.requested_transitions,
                  "diagnostic_source": split_metadata["diagnostic_source"]}
        (run_dir / "smoke_report.json").write_text(json.dumps(report, indent=2) + "\n")
        close_run(run_dir, status="completed", accounting=InteractionAccounting(config.ppo.requested_transitions, 0, 0, transitions), reason="completed")
        return report
    except Exception as exc:
        close_run(run_dir, status="engineering_error", accounting=InteractionAccounting(0, 0, 0, 0), reason=str(exc))
        raise
