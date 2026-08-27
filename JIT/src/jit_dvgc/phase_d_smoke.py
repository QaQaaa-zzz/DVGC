"""Small, auditable Phase D engineering-smoke training link.

This module intentionally owns orchestration only.  Physics, observation and
reward semantics remain in the existing environment modules.
"""
from __future__ import annotations

from dataclasses import asdict
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


def validate_parent_group_split(train_groups, eval_groups) -> bool:
    overlap = set(train_groups).intersection(eval_groups)
    if overlap:
        raise ValueError(f"parent_group split overlap: {sorted(overlap)}")
    return True


def validate_phase_d_smoke_args(*, formal: bool, snapshot_bank, snapshot_catalog,
                                actor_init_checkpoint, actor_init_config) -> None:
    if formal:
        raise ValueError("Phase D formal training is not implemented")
    if snapshot_bank is None and snapshot_catalog is None:
        raise ValueError("Phase D smoke requires a snapshot bank or catalog")
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
    run_root: Path | None = None,
    trainer: Callable[..., Any] = ppo_train.train,
    env_factory: Callable[..., Any] = TwoPhaseBikeEnv,
) -> dict[str, Any]:
    validate_phase_d_smoke_args(formal=False, snapshot_bank=snapshot_bank,
                                snapshot_catalog=snapshot_catalog,
                                actor_init_checkpoint=actor_init_checkpoint,
                                actor_init_config=actor_init_config)
    config = load_config(Path(config_path))
    if config.phase != "descent_recovery" or config.formal is not None:
        raise ValueError("Phase D smoke requires a non-formal descent_recovery config")
    source_config = load_config(Path(actor_init_config))
    # Build a source-compatible pool before constructing the target environment.
    source_env = TwoPhaseBikeEnv(source_config)
    input_path = snapshot_catalog or snapshot_bank
    pool = pool_from_input(input_path, compatibility=compatibility_identity(source_env))
    env = env_factory(config, snapshot_pool=pool)
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
                        f"--config {config_path} --run-id {run_id}"),
    )
    predeclare_run(declaration, resolved_config=config.raw)
    (run_dir / "phase_d_provenance.json").write_text(json.dumps({
        "snapshot_input": str(input_path), "actor_init_checkpoint": str(Path(actor_init_checkpoint).resolve()),
        "actor_init_config": str(Path(actor_init_config).resolve()),
        "parent_transition": initialization.parent_transition,
        "parent_payload_sha256": initialization.payload_sha256,
        "parent_actor_sha256": initialization.actor_sha256,
        **dict(initialization.provenance), "train_eval_split": "parent_group_id only",
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
        state = env.restore_handoff_snapshot(pool.snapshot(0))
        policy = make_actor_only_policy(env, initialization, deterministic=True)
        transitions = 0
        while transitions < config.ppo.episode_horizon and not bool(np.asarray(state.done)):
            action, _ = policy(state.obs, jax.random.PRNGKey(1000000 + transitions))
            state = env.step(state, action)
            transitions += 1
            if bool(np.asarray(state.info["terminated"])) or bool(np.asarray(state.info["truncated"])):
                break
        report = {"status": "completed", "training_transitions": config.ppo.requested_transitions,
                  "diagnostic_transitions": transitions, "checkpoint": str(checkpoint),
                  "restored": restored.training_transitions == config.ppo.requested_transitions}
        (run_dir / "smoke_report.json").write_text(json.dumps(report, indent=2) + "\n")
        close_run(run_dir, status="completed", accounting=InteractionAccounting(config.ppo.requested_transitions, 0, 0, transitions), reason="completed")
        return report
    except Exception as exc:
        close_run(run_dir, status="engineering_error", accounting=InteractionAccounting(0, 0, 0, 0), reason=str(exc))
        raise
