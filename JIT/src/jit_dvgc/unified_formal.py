"""Formal 10M+ PPO orchestration for the single unified Tube-RSI Actor."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from brax.training.agents.ppo import train as ppo_train
import jax
import numpy as np

from .checkpoint import CheckpointIdentity, CheckpointPayload, save_checkpoint
from .formal_training import FormalRunController, PanelResult, _flatten_finite_metrics
from .ppo import make_network_factory, wrap_for_jit_training
from .provenance import (
    InteractionAccounting,
    RunDeclaration,
    close_run,
    mark_run_running,
    predeclare_run,
)
from .unified_diagnostic import (
    _load_runtime,
    _tube_points,
    plot_xz_visitation,
    rollout_fixed_tube_panel,
)
from .unified_env import UnifiedTubeRSIEnv
from .unified_training import (
    UnifiedPPOConfig,
    canonical_sha256,
    checkpoint_identity,
    read_json,
)


FORMAL_SCHEMA = "jit_pi_unified_formal_v1"
FORMAL_TARGET = 10_009_600
FORMAL_CHECKPOINTS = (0, 1_024_000, 2_508_800, 5_017_600, 7_500_800, FORMAL_TARGET)
FORMAL_TRAIN_PANELS = FORMAL_CHECKPOINTS[1:]


@dataclass(frozen=True)
class UnifiedFormalSchedule:
    checkpoint_transitions: tuple[int, ...]
    train_panel_transitions: tuple[int, ...]
    samples_per_phase: int
    resume_semantics: str

    @property
    def fixed_evaluation_transitions(self) -> tuple[int, ...]:
        """Compatibility name used by the shared ordered callback controller."""
        return self.train_panel_transitions


@dataclass(frozen=True)
class UnifiedFormalConfig:
    schema: str
    raw: Mapping[str, Any]
    config_sha256: str
    up_config_path: str
    up_config_sha256: str
    down_config_path: str
    down_config_sha256: str
    soft_tube_path: str
    soft_tube_manifest_sha256: str
    tube_rsi_smoke_report: str
    tube_rsi_smoke_report_sha256: str
    runtime_naccdmax: int
    ppo: UnifiedPPOConfig
    formal: UnifiedFormalSchedule


def load_unified_formal_config(path: Path) -> UnifiedFormalConfig:
    payload = read_json(Path(path))
    if payload.get("schema") != FORMAL_SCHEMA:
        raise ValueError("unsupported unified formal schema")
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("unified formal inputs are missing")
    try:
        ppo = UnifiedPPOConfig(**payload["ppo"])
        formal_raw = payload["formal"]
        formal = UnifiedFormalSchedule(
            checkpoint_transitions=tuple(formal_raw["checkpoint_transitions"]),
            train_panel_transitions=tuple(formal_raw["train_panel_transitions"]),
            samples_per_phase=int(formal_raw["samples_per_phase"]),
            resume_semantics=str(formal_raw["resume_semantics"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid unified formal contract: {exc}") from exc
    if payload.get("runtime") != {"naccdmax": 1024}:
        raise ValueError("unified formal runtime must declare naccdmax=1024")
    if ppo.block_transitions != 25_600 or ppo.requested_transitions != FORMAL_TARGET:
        raise ValueError("unified formal PPO must contain exactly 391 aligned blocks")
    if ppo.num_parallel_envs != 1024 or ppo.seed != 821101:
        raise ValueError("unified formal parallelism or seed drift")
    if ppo.batch_size * ppo.num_minibatches % ppo.num_parallel_envs:
        raise ValueError("unified formal sequence count is not environment-aligned")
    for name in (
        "learning_rate",
        "entropy_cost",
        "reward_scaling",
        "discounting",
        "gae_lambda",
        "clipping_epsilon",
        "max_grad_norm",
    ):
        value = float(getattr(ppo, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"unified formal PPO {name} must be finite and positive")
    if formal.checkpoint_transitions != FORMAL_CHECKPOINTS:
        raise ValueError("unified formal checkpoint schedule drift")
    if formal.train_panel_transitions != FORMAL_TRAIN_PANELS:
        raise ValueError("unified formal TRAIN-panel schedule drift")
    if formal.samples_per_phase != 8 or formal.resume_semantics != "fresh_only":
        raise ValueError("unified formal panel or resume contract drift")
    if payload.get("initialization") != {
        "actor": "fresh",
        "critic": "fresh",
        "optimizer": "fresh",
        "restore_checkpoint": None,
    }:
        raise ValueError("unified formal training must start wholly fresh")
    boundary = payload.get("claim_boundary", {})
    if boundary.get("formal_method_stage_training") is not True:
        raise ValueError("unified formal method-stage boundary is missing")
    if (
        boundary.get("test_data_used") is not False
        or boundary.get("validation_data_used") is not False
    ):
        raise ValueError(
            "unified formal training must exclude validation and TEST data"
        )
    required = (
        "up_config_path",
        "up_config_sha256",
        "down_config_path",
        "down_config_sha256",
        "soft_tube_path",
        "soft_tube_manifest_sha256",
        "tube_rsi_smoke_report",
        "tube_rsi_smoke_report_sha256",
    )
    if any(
        not isinstance(inputs.get(name), str) or not inputs[name] for name in required
    ):
        raise ValueError("unified formal input identity is incomplete")
    return UnifiedFormalConfig(
        schema=FORMAL_SCHEMA,
        raw=payload,
        config_sha256=canonical_sha256(payload),
        runtime_naccdmax=1024,
        ppo=ppo,
        formal=formal,
        **{name: inputs[name] for name in required},
    )


class UnifiedFormalController(FormalRunController):
    """Names shared callback evidence as TRAIN diagnostics, not evaluation."""

    def __init__(
        self,
        *,
        config: UnifiedFormalConfig,
        run_dir: Path,
        identity: CheckpointIdentity,
        evaluate_train_panel: Callable[[int, Any, Any], PanelResult],
        checkpoint_saver: Callable[[Path, CheckpointPayload], None] = save_checkpoint,
    ) -> None:
        super().__init__(
            config=config,
            run_dir=run_dir,
            identity=identity,
            starting_training_transition=0,
            evaluate_panel=evaluate_train_panel,
            checkpoint_saver=checkpoint_saver,
        )
        self._last_episode_metric_step = 0

    @property
    def train_panel_interactions(self) -> int:
        return self.fixed_evaluation_transitions

    @property
    def train_panel_transitions(self) -> tuple[int, ...]:
        return self.evaluated_transitions

    def on_progress(self, step: int, metrics: Mapping[str, Any]) -> None:
        if metrics and all(str(key).startswith("episode/") for key in metrics):
            relative = int(step)
            if relative <= self._last_episode_metric_step:
                raise ValueError("episode metric callbacks must be strictly increasing")
            if relative % self.config.ppo.block_transitions:
                raise ValueError("episode metric callback must be block-aligned")
            if relative > self.config.ppo.requested_transitions:
                raise ValueError("episode metric callback exceeded the formal target")
            row = {
                "training_transitions": relative,
                "metrics": _flatten_finite_metrics(metrics),
            }
            with (self.run_dir / "episode_metrics.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            self._last_episode_metric_step = relative
            return
        # Brax's ordered post-epoch callback follows policy_params_fn and is
        # the checkpointable block metric.
        super().on_progress(step, metrics)


def build_unified_formal_trainer_kwargs(
    config: UnifiedFormalConfig,
    environment: Any,
    callbacks: Any,
) -> dict[str, Any]:
    ppo = config.ppo
    return {
        "environment": environment,
        "num_timesteps": ppo.requested_transitions,
        "max_devices_per_host": 1,
        "wrap_env": True,
        "wrap_env_fn": wrap_for_jit_training,
        "num_envs": ppo.num_parallel_envs,
        "episode_length": ppo.episode_horizon,
        "action_repeat": 1,
        "learning_rate": ppo.learning_rate,
        "entropy_cost": ppo.entropy_cost,
        "discounting": ppo.discounting,
        "unroll_length": ppo.unroll_length,
        "batch_size": ppo.batch_size,
        "num_minibatches": ppo.num_minibatches,
        "num_updates_per_batch": ppo.num_updates_per_batch,
        "normalize_observations": True,
        "reward_scaling": ppo.reward_scaling,
        "clipping_epsilon": ppo.clipping_epsilon,
        "gae_lambda": ppo.gae_lambda,
        "max_grad_norm": ppo.max_grad_norm,
        "bootstrap_on_timeout": True,
        "network_factory": make_network_factory(),
        "seed": ppo.seed,
        "num_evals": ppo.requested_transitions // ppo.block_transitions + 1,
        "num_eval_envs": 8,
        "deterministic_eval": True,
        "log_training_metrics": True,
        "training_metrics_steps": ppo.block_transitions,
        "progress_fn": callbacks.on_progress,
        "policy_params_fn": callbacks.on_policy_params,
        "restore_params": None,
        "restore_value_fn": False,
        "run_evals": False,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _evaluate_train_panel(
    env: Any,
    artifact: Any,
    run_dir: Path,
    config: UnifiedFormalConfig,
    step: int,
    make_policy: Any,
    params: Any,
) -> PanelResult:
    panel_dir = run_dir / "train_panels" / f"transition_{step}"
    panel_dir.mkdir(parents=True, exist_ok=False)
    deterministic_policy = make_policy(params, deterministic=True)
    report, trajectories = rollout_fixed_tube_panel(
        env,
        deterministic_policy,
        samples_per_phase=config.formal.samples_per_phase,
        horizon=config.ppo.episode_horizon,
    )
    report = {**report, "training_checkpoint_transition": step}
    _write_json(panel_dir / "report.json", report)
    _write_json(panel_dir / "trajectories.json", {"trajectories": trajectories})
    plot_xz_visitation(
        _tube_points(artifact), trajectories, panel_dir / "xz_visitation.png"
    )
    return PanelResult(step, int(report["environment_interactions"]), report)


def _verify_restored_policy(env: Any, make_policy: Any, params: Any) -> None:
    state = env.reset_tube_index(np.int32(0), np.int32(0))
    policy = make_policy(params, deterministic=True)
    action, _ = policy(state.obs, jax.random.PRNGKey(821101))
    array = np.asarray(jax.device_get(action))
    if array.shape != (4,) or not np.isfinite(array).all():
        raise ValueError("restored unified Actor inference is invalid")


def run_unified_formal(
    config_path: Path,
    run_id: str,
    *,
    run_root: Path | None = None,
    trainer: Callable[..., Any] = ppo_train.train,
    env_factory: Callable[..., Any] = UnifiedTubeRSIEnv,
    backend_name: Callable[[], str] = jax.default_backend,
) -> dict[str, Any]:
    config = load_unified_formal_config(config_path)
    up_config, down_config, artifact, _ = _load_runtime(config)
    if backend_name() != "gpu":
        raise RuntimeError("formal unified PPO requires the visible JAX GPU backend")
    env = env_factory(
        up_config, down_config, artifact, runtime_naccdmax=config.runtime_naccdmax
    )
    if env._bundle.xml_sha256 != up_config.model["xml_sha256"]:
        raise ValueError("unified formal runtime XML identity mismatch")
    root = (
        Path(run_root)
        if run_root is not None
        else Path(os.environ.get("JIT_RUN_ROOT", "JIT/runs/pi_unified"))
    )
    run_dir = root / run_id
    declaration = RunDeclaration(
        run_id=run_id,
        purpose="formal_pi_unified_tube_rsi_ppo",
        output_dir=run_dir,
        config_sha256=config.config_sha256,
        xml_sha256=env._bundle.xml_sha256,
        reference_sha256=str(up_config.model["reference_sha256"]),
        training_transition_ceiling=config.ppo.requested_transitions,
        stopping_conditions=(
            f"stop_at_exact_transition_{config.ppo.requested_transitions}",
            "stop_on_nonfinite_metric",
            "stop_on_cuda_or_oom_error",
            "stop_on_checkpoint_or_train_panel_persistence_failure",
        ),
        resume_command=(
            "PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python "
            "JIT/cli/train_unified.py "
            f"--config {Path(config_path)} --run-id {run_id}"
        ),
        segment_seed=config.ppo.seed,
    )
    predeclare_run(declaration, resolved_config=config.raw)
    _write_json(
        run_dir / "backend.json",
        {"jax_backend": backend_name(), "devices": [str(x) for x in jax.devices()]},
    )
    _write_json(
        run_dir / "formal_provenance.json",
        {
            "schema": "jit_pi_unified_formal_provenance_v1",
            "initialization": config.raw["initialization"],
            "policy_count": 1,
            "expert_switching_used": False,
            "soft_tube_manifest_sha256": config.soft_tube_manifest_sha256,
            "tube_rsi_smoke_report_sha256": config.tube_rsi_smoke_report_sha256,
            "test_data_used": False,
            "validation_data_used": False,
        },
    )
    mark_run_running(
        run_dir,
        process_id=os.getpid(),
        metadata={
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "training_seed": config.ppo.seed,
            "target_training_transition": config.ppo.requested_transitions,
            "resume_semantics": "fresh",
        },
    )

    identity = checkpoint_identity(config, env)

    def evaluate(step: int, make_policy: Any, params: Any) -> PanelResult:
        return _evaluate_train_panel(
            env, artifact, run_dir, config, step, make_policy, params
        )

    controller = UnifiedFormalController(
        config=config,
        run_dir=run_dir,
        identity=identity,
        evaluate_train_panel=evaluate,
    )
    try:
        make_policy, _params, final_metrics = trainer(
            **build_unified_formal_trainer_kwargs(config, env, controller)
        )
        if (
            controller.completed_training_transitions
            != config.ppo.requested_transitions
        ):
            raise ValueError("unified formal trainer returned before the exact target")
        restored = controller.restore_final_checkpoint()
        restored_params = (
            restored.observation_normalizer,
            restored.actor_params,
            restored.critic_params,
        )
        _verify_restored_policy(env, make_policy, restored_params)
        metrics = {
            str(key): float(np.asarray(jax.device_get(value)))
            for key, value in final_metrics.items()
            if np.asarray(jax.device_get(value)).ndim == 0
        }
        if not metrics:
            metrics = dict(controller.final_metrics)
        if not metrics or not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError("unified formal PPO produced no finite final metrics")
        report = {
            "schema": "jit_pi_unified_formal_report_v1",
            "status": "completed",
            "requested_training_transitions": config.ppo.requested_transitions,
            "completed_training_transitions": controller.completed_training_transitions,
            "checkpoint_transitions": list(controller.checkpoint_transitions),
            "train_panel_transitions": list(controller.train_panel_transitions),
            "train_panel_interactions": controller.train_panel_interactions,
            "brax_evaluation_transitions": 0,
            "test_data_used": False,
            "validation_data_used": False,
            "expert_switching_used": False,
            "checkpoint_restored": True,
            "final_metrics": metrics,
        }
        _write_json(run_dir / "formal_report.json", report)
        close_run(
            run_dir,
            status="completed",
            accounting=InteractionAccounting(
                config.ppo.requested_transitions,
                0,
                0,
                controller.train_panel_interactions,
            ),
            reason="formal unified Tube-RSI target and TRAIN-only milestone panels completed; independent final evaluation remains separate",
        )
        return {"run_dir": str(run_dir.resolve()), "formal_report": report}
    except Exception as exc:
        close_run(
            run_dir,
            status="engineering_error",
            accounting=InteractionAccounting(
                controller.segment_training_transitions,
                0,
                0,
                controller.train_panel_interactions,
            ),
            reason=f"{type(exc).__name__}: {exc}",
        )
        raise
