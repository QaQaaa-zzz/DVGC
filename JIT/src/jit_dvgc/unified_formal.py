"""Formal 10M+ PPO orchestration for the single unified Tube-RSI Actor."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

from brax.training.agents.ppo import train as ppo_train
import jax
import numpy as np

from .checkpoint import (
    CheckpointIdentity,
    CheckpointPayload,
    load_checkpoint,
    save_checkpoint,
)
from .constants import ACTION_ORDER, ACTOR_FRAME_FIELDS, ACTOR_TASK_FIELDS
from .formal_training import FormalRunController, PanelResult, _flatten_finite_metrics
from .handoff_bank import pytree_sha256
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
LEGACY_ROUND1_RESET_MIXTURE_SCHEMA = "jit_pi_unified_round1_reset_mix_v1"
RESET_MIXTURE_SCHEMA = "jit_pi_unified_reset_mix_v1"
ROUND1_NATURAL_RESET_PROBABILITY = 0.10
ROUND1_SOFT_TUBE_PROBABILITY = 0.90
ROUND0_FAILURE_EVIDENCE = "PI_UP_APEX_UNIFIED_PRE_JUMP_FAIL"


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
class UnifiedResetMixture:
    selection: str
    natural_reset_probability: float
    soft_tube_probability: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "selection": self.selection,
            "natural_reset_probability": self.natural_reset_probability,
            "soft_tube_probability": self.soft_tube_probability,
        }


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
    reset_mixture: UnifiedResetMixture
    ppo: UnifiedPPOConfig
    formal: UnifiedFormalSchedule


def _load_reset_mixture(payload: Mapping[str, Any]) -> UnifiedResetMixture:
    raw = payload.get("reset_mixture")
    if raw is None:
        return UnifiedResetMixture(
            selection="soft_tube_only",
            natural_reset_probability=0.0,
            soft_tube_probability=1.0,
        )
    if not isinstance(raw, Mapping):
        raise ValueError("unified reset mixture must be an object")

    schema = raw.get("schema")
    if schema == LEGACY_ROUND1_RESET_MIXTURE_SCHEMA:
        expected = {
            "schema": LEGACY_ROUND1_RESET_MIXTURE_SCHEMA,
            "selection": "bernoulli_per_episode",
            "natural_reset_probability": ROUND1_NATURAL_RESET_PROBABILITY,
            "soft_tube_probability": ROUND1_SOFT_TUBE_PROBABILITY,
            "natural_reset_semantics": "existing_phase_u_natural_reset",
            "soft_tube_semantics": "existing_phase_balanced_value_weighted_tube_rsi",
            "single_variable": "reset_distribution_only",
        }
        if raw != expected:
            raise ValueError("unified reset mixture contract drift")
        boundary = payload.get("claim_boundary", {})
        if boundary.get("round1_single_variable_iteration") is not True:
            raise ValueError(
                "legacy Round-1 mixed reset requires the Round-1 single-variable boundary"
            )
        if boundary.get("round0_failure_evidence") != ROUND0_FAILURE_EVIDENCE:
            raise ValueError(
                "legacy Round-1 mixed reset is not bound to the locked Round-0 diagnosis"
            )
        return UnifiedResetMixture(
            selection="bernoulli_per_episode",
            natural_reset_probability=ROUND1_NATURAL_RESET_PROBABILITY,
            soft_tube_probability=ROUND1_SOFT_TUBE_PROBABILITY,
        )

    if schema != RESET_MIXTURE_SCHEMA:
        raise ValueError("unsupported unified reset mixture schema")
    expected_keys = {
        "schema",
        "selection",
        "natural_reset_probability",
        "soft_tube_probability",
        "natural_reset_semantics",
        "soft_tube_semantics",
        "single_variable",
    }
    if set(raw) != expected_keys:
        raise ValueError("unified reset mixture fields drift")
    expected_fixed = {
        "selection": "bernoulli_per_episode",
        "natural_reset_semantics": "existing_phase_u_natural_reset",
        "soft_tube_semantics": "existing_phase_balanced_value_weighted_tube_rsi",
        "single_variable": "reset_distribution_only",
    }
    if any(raw.get(key) != value for key, value in expected_fixed.items()):
        raise ValueError("unified reset mixture semantics drift")
    natural = float(raw["natural_reset_probability"])
    soft = float(raw["soft_tube_probability"])
    if not math.isfinite(natural) or not math.isfinite(soft):
        raise ValueError("unified reset probabilities must be finite")
    if not (0.0 < natural < 1.0) or not (0.0 < soft < 1.0):
        raise ValueError("unified mixed reset probabilities must be strictly between zero and one")
    if not math.isclose(natural + soft, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("unified reset probabilities must sum to one")
    return UnifiedResetMixture(
        selection="bernoulli_per_episode",
        natural_reset_probability=natural,
        soft_tube_probability=soft,
    )


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
        reset_mixture = _load_reset_mixture(payload)
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
        reset_mixture=reset_mixture,
        ppo=ppo,
        formal=formal,
        **{name: inputs[name] for name in required},
    )


_LOAD_FRESH_UNIFIED_FORMAL_CONFIG = load_unified_formal_config


def actor_only_warm_start_initialization(source_frozen_policy: str | Path) -> dict[str, Any]:
    source = str(source_frozen_policy)
    if not source:
        raise ValueError("actor-only warm-start frozen policy path is missing")
    return {
        "actor": "warm_start_frozen_unified",
        "critic": "fresh",
        "optimizer": "fresh",
        "source_frozen_policy": source,
    }


def load_unified_actor_warm_start_config(path: Path) -> UnifiedFormalConfig:
    """Validate a formal config whose only restored training state is an Actor.

    ``warm_start_pi_0`` remains accepted for the two historical comparison runs.
    New iterations use ``warm_start_frozen_unified`` so this path is not tied to
    a particular policy generation.
    """
    path = Path(path)
    payload = read_json(path)
    initialization = payload.get("initialization")
    if not isinstance(initialization, Mapping):
        raise ValueError("actor-only warm-start initialization is missing")
    if set(initialization) != {
        "actor",
        "critic",
        "optimizer",
        "source_frozen_policy",
    }:
        raise ValueError("actor-only warm-start initialization fields drift")
    if initialization.get("actor") not in {
        "warm_start_frozen_unified",
        "warm_start_pi_0",
    }:
        raise ValueError("unsupported actor-only warm-start source mode")
    if (
        initialization.get("critic") != "fresh"
        or initialization.get("optimizer") != "fresh"
    ):
        raise ValueError("actor-only warm-start initialization must keep critic and optimizer fresh")
    if not isinstance(initialization.get("source_frozen_policy"), str) or not str(
        initialization["source_frozen_policy"]
    ):
        raise ValueError("actor-only warm-start frozen policy path is missing")

    sanitized = dict(payload)
    sanitized["initialization"] = {
        "actor": "fresh",
        "critic": "fresh",
        "optimizer": "fresh",
        "restore_checkpoint": None,
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", encoding="utf-8", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(sanitized, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    try:
        parsed = _LOAD_FRESH_UNIFIED_FORMAL_CONFIG(temporary)
    finally:
        temporary.unlink(missing_ok=True)
    return replace(
        parsed,
        raw=payload,
        config_sha256=canonical_sha256(payload),
    )


def load_frozen_actor_restore_params(config_path: Path):
    """Load and verify the immediately preceding frozen unified Actor."""
    config = load_unified_actor_warm_start_config(config_path)
    frozen_path = Path(config.raw["initialization"]["source_frozen_policy"])
    frozen = read_json(frozen_path)
    if (
        frozen.get("schema") != "jit_frozen_unified_policy_v1"
        or frozen.get("status") != "frozen"
    ):
        raise ValueError("warm-start source is not a frozen unified policy")
    policy = frozen.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError("frozen warm-start policy record is missing")

    boundary = config.raw.get("claim_boundary", {})
    target_iteration = int(boundary.get("iteration", -1))
    source_iteration = int(policy.get("iteration", -1))
    if target_iteration < 1 or source_iteration != target_iteration - 1:
        raise ValueError("warm-start source must be the immediately preceding iteration")
    if policy.get("name") != f"pi_{source_iteration}":
        raise ValueError("warm-start source policy name/iteration drift")
    declared_source_name = boundary.get("source_policy_name")
    if declared_source_name is not None and declared_source_name != policy.get("name"):
        raise ValueError("warm-start source differs from the declared source policy")

    source_config_path = Path(str(policy["formal_config"]))
    source_raw = read_json(source_config_path)
    if source_raw.get("initialization", {}).get("actor") in {
        "warm_start_frozen_unified",
        "warm_start_pi_0",
    }:
        source_config = load_unified_actor_warm_start_config(source_config_path)
    else:
        source_config = _LOAD_FRESH_UNIFIED_FORMAL_CONFIG(source_config_path)
    expected = CheckpointIdentity(
        config_sha256=source_config.config_sha256,
        xml_sha256=str(policy["xml_sha256"]),
        actor_frame_fields=ACTOR_FRAME_FIELDS,
        actor_task_fields=ACTOR_TASK_FIELDS,
        action_order=ACTION_ORDER,
    )
    checkpoint = load_checkpoint(Path(str(policy["checkpoint"])), expected=expected)
    if int(checkpoint.training_transitions) != int(policy["source_training_transitions"]):
        raise ValueError("warm-start checkpoint transition drift")
    identities = {
        "normalizer_sha256": pytree_sha256(checkpoint.observation_normalizer),
        "actor_sha256": pytree_sha256(checkpoint.actor_params),
        "critic_sha256": pytree_sha256(checkpoint.critic_params),
    }
    for field, observed in identities.items():
        if observed != str(policy.get(field)):
            raise ValueError(f"warm-start {field.removesuffix('_sha256')} payload drift")
    return (
        checkpoint.observation_normalizer,
        checkpoint.actor_params,
        checkpoint.critic_params,
    )


def load_unified_policy_formal_config(path: Path) -> UnifiedFormalConfig:
    """Load either a fresh or Actor-only-warm-start unified policy config."""
    raw = read_json(Path(path))
    initialization = raw.get("initialization", {})
    if (
        initialization.get("actor")
        in {"warm_start_frozen_unified", "warm_start_pi_0"}
        and initialization.get("critic") == "fresh"
    ):
        return load_unified_actor_warm_start_config(Path(path))
    return load_unified_formal_config(Path(path))


def _build_unified_formal_environment(
    config: UnifiedFormalConfig,
    *,
    env_factory: Callable[..., Any] = UnifiedTubeRSIEnv,
):
    up_config, down_config, artifact, _ = _load_runtime(config)
    env = env_factory(
        up_config,
        down_config,
        artifact,
        runtime_naccdmax=config.runtime_naccdmax,
        natural_reset_probability=config.reset_mixture.natural_reset_probability,
    )
    if env._bundle.xml_sha256 != up_config.model["xml_sha256"]:
        raise ValueError("unified formal runtime XML identity mismatch")
    return artifact, env


def build_unified_formal_environment(
    config_path: Path,
    *,
    env_factory: Callable[..., Any] = UnifiedTubeRSIEnv,
):
    """Build the stable unified runtime without creating a run or training."""
    config = load_unified_policy_formal_config(config_path)
    artifact, env = _build_unified_formal_environment(config, env_factory=env_factory)
    return config, artifact, env


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
    config = load_unified_policy_formal_config(config_path)
    if backend_name() != "gpu":
        raise RuntimeError("formal unified PPO requires the visible JAX GPU backend")
    artifact, env = _build_unified_formal_environment(config, env_factory=env_factory)
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
        reference_sha256=str(env.resolved_config.model["reference_sha256"]),
        training_transition_ceiling=config.ppo.requested_transitions,
        stopping_conditions=(
            f"stop_at_exact_transition_{config.ppo.requested_transitions}",
            "stop_on_nonfinite_metric",
            "stop_on_cuda_or_oom_error",
            "stop_on_checkpoint_or_train_panel_persistence_failure",
        ),
        resume_command=(
            "PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python "
            + (
                "JIT/cli/train_unified_from_pi0.py "
                if config.raw.get("initialization", {}).get("actor")
                in {"warm_start_frozen_unified", "warm_start_pi_0"}
                else "JIT/cli/train_unified.py "
            )
            + f"--config {Path(config_path)} --run-id {run_id}"
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
            "reset_mixture": config.reset_mixture.as_dict(),
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
        trainer_kwargs = build_unified_formal_trainer_kwargs(config, env, controller)
        if config.raw.get("initialization", {}).get("actor") in {
            "warm_start_frozen_unified", "warm_start_pi_0"
        }:
            trainer_kwargs["restore_params"] = load_frozen_actor_restore_params(config_path)
            trainer_kwargs["restore_value_fn"] = False
        make_policy, _params, final_metrics = trainer(**trainer_kwargs)
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
            "reset_mixture": config.reset_mixture.as_dict(),
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
