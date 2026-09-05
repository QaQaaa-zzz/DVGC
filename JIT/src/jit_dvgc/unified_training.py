"""Fresh one-block PPO pilot for the single unified Tube-RSI policy."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
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
from .config import canonical_sha256, file_sha256, load_config
from .constants import ACTION_ORDER, ACTOR_FRAME_FIELDS, ACTOR_TASK_FIELDS
from .ppo import make_network_factory, wrap_for_jit_training
from .provenance import (
    InteractionAccounting,
    RunDeclaration,
    close_run,
    mark_run_running,
    predeclare_run,
)
from .soft_tube import SoftTubeArtifact, load_soft_tube
from .unified_env import UnifiedTubeRSIEnv


PILOT_SCHEMA = "jit_pi_unified_pilot_v1"


@dataclass(frozen=True)
class UnifiedPPOConfig:
    num_parallel_envs: int
    episode_horizon: int
    unroll_length: int
    batch_size: int
    num_minibatches: int
    num_updates_per_batch: int
    requested_transitions: int
    learning_rate: float
    entropy_cost: float
    reward_scaling: float
    discounting: float
    gae_lambda: float
    clipping_epsilon: float
    max_grad_norm: float
    seed: int

    @property
    def block_transitions(self) -> int:
        return self.unroll_length * self.batch_size * self.num_minibatches


@dataclass(frozen=True)
class UnifiedPilotConfig:
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


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def load_unified_pilot_config(path: Path) -> UnifiedPilotConfig:
    payload = read_json(path)
    if payload.get("schema") != PILOT_SCHEMA:
        raise ValueError("unsupported unified pilot schema")
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("unified pilot inputs are missing")
    try:
        ppo = UnifiedPPOConfig(**payload["ppo"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"invalid unified PPO contract: {exc}") from exc
    runtime = payload.get("runtime")
    if runtime != {"naccdmax": 1024}:
        raise ValueError("unified pilot runtime must declare naccdmax=1024")
    integer_fields = (
        "num_parallel_envs",
        "episode_horizon",
        "unroll_length",
        "batch_size",
        "num_minibatches",
        "num_updates_per_batch",
        "requested_transitions",
        "seed",
    )
    if any(int(getattr(ppo, field)) <= 0 for field in integer_fields):
        raise ValueError("unified PPO integer fields must be positive")
    if ppo.block_transitions != 25_600 or ppo.requested_transitions != 25_600:
        raise ValueError("unified pilot must contain exactly one 25600-transition block")
    if ppo.num_parallel_envs != 1024:
        raise ValueError("unified pilot must use 1024 parallel environments")
    if ppo.batch_size * ppo.num_minibatches % ppo.num_parallel_envs:
        raise ValueError("unified PPO sequence count is not environment-aligned")
    for field in (
        "learning_rate",
        "entropy_cost",
        "reward_scaling",
        "discounting",
        "gae_lambda",
        "clipping_epsilon",
        "max_grad_norm",
    ):
        value = float(getattr(ppo, field))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"unified PPO {field} must be finite and positive")
    expected_initialization = {
        "actor": "fresh",
        "critic": "fresh",
        "optimizer": "fresh",
        "restore_checkpoint": None,
    }
    if payload.get("initialization") != expected_initialization:
        raise ValueError("unified pilot must use wholly fresh initialization")
    boundary = payload.get("claim_boundary", {})
    if boundary.get("engineering_integrity_only") is not True:
        raise ValueError("unified pilot is not bounded to engineering integrity")
    if boundary.get("test_data_used") is not False:
        raise ValueError("unified pilot does not exclude TEST data")
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
    if any(not isinstance(inputs.get(field), str) or not inputs[field] for field in required):
        raise ValueError("unified pilot has an incomplete input identity")
    return UnifiedPilotConfig(
        schema=PILOT_SCHEMA,
        raw=payload,
        config_sha256=canonical_sha256(payload),
        ppo=ppo,
        runtime_naccdmax=1024,
        **{field: inputs[field] for field in required},
    )


def validate_pilot_gate(
    config: UnifiedPilotConfig,
    soft_manifest: Mapping[str, Any],
    smoke_report: Mapping[str, Any],
) -> None:
    if (
        soft_manifest.get("schema") != "jit_soft_tube_v1"
        or soft_manifest.get("status") != "completed"
        or soft_manifest.get("training_guidance_only") is not True
        or soft_manifest.get("certified_safe") is not False
    ):
        raise ValueError("learned Soft Tube gate is not complete")
    if (
        soft_manifest.get("test_data_used") is not False
        or soft_manifest.get("validation_data_used") is not False
    ):
        raise ValueError("Soft Tube TEST exclusion is not closed")
    if soft_manifest.get("manifest_sha256") != config.soft_tube_manifest_sha256:
        raise ValueError("configured Soft Tube identity mismatch")
    if (
        smoke_report.get("schema") != "jit_tube_rsi_smoke_v1"
        or smoke_report.get("status") != "completed"
        or smoke_report.get("tube_rsi_smoke") != "GO"
    ):
        raise ValueError("Tube-RSI smoke is not GO")
    if (
        smoke_report.get("environment_interactions") != 16
        or smoke_report.get("training_transitions") != 0
        or smoke_report.get("expert_switching_used") is not False
    ):
        raise ValueError("Tube-RSI smoke contract mismatch")
    if (
        smoke_report.get("test_data_used") is not False
        or smoke_report.get("validation_data_used") is not False
    ):
        raise ValueError("Tube-RSI smoke TEST exclusion is not closed")
    if smoke_report.get("soft_tube_manifest_sha256") != config.soft_tube_manifest_sha256:
        raise ValueError("Tube-RSI smoke Soft Tube identity mismatch")


def build_unified_trainer_kwargs(
    config: UnifiedPilotConfig,
    environment: Any,
    *,
    progress_fn: Callable[[int, Mapping[str, Any]], None],
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
        "num_evals": 0,
        "num_eval_envs": 8,
        "deterministic_eval": True,
        "run_evals": False,
        "log_training_metrics": True,
        "progress_fn": progress_fn,
        "restore_params": None,
        "restore_value_fn": False,
    }


def checkpoint_identity(
    config: UnifiedPilotConfig, environment: Any
) -> CheckpointIdentity:
    return CheckpointIdentity(
        config_sha256=config.config_sha256,
        xml_sha256=environment._bundle.xml_sha256,
        actor_frame_fields=ACTOR_FRAME_FIELDS,
        actor_task_fields=ACTOR_TASK_FIELDS,
        action_order=ACTION_ORDER,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    array = np.asarray(jax.device_get(value))
    if array.ndim == 0:
        scalar = array.item()
        if isinstance(scalar, float) and not math.isfinite(scalar):
            raise ValueError("nonfinite unified PPO metric")
        return scalar
    return array.tolist()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def run_unified_pilot(
    config_path: Path,
    run_id: str,
    *,
    run_root: Path | None = None,
    trainer: Callable[..., Any] = ppo_train.train,
    env_factory: Callable[..., Any] = UnifiedTubeRSIEnv,
) -> dict[str, Any]:
    config = load_unified_pilot_config(config_path)
    up_config = load_config(Path(config.up_config_path))
    down_config = load_config(Path(config.down_config_path))
    if up_config.config_sha256 != config.up_config_sha256:
        raise ValueError("unified upstream config identity mismatch")
    if down_config.config_sha256 != config.down_config_sha256:
        raise ValueError("unified downstream config identity mismatch")
    artifact: SoftTubeArtifact = load_soft_tube(Path(config.soft_tube_path))
    smoke_path = Path(config.tube_rsi_smoke_report)
    if file_sha256(smoke_path) != config.tube_rsi_smoke_report_sha256:
        raise ValueError("Tube-RSI smoke report SHA-256 mismatch")
    smoke_report = read_json(smoke_path)
    validate_pilot_gate(config, artifact.manifest, smoke_report)
    if jax.default_backend() != "gpu":
        raise RuntimeError("the unified PPO pilot requires the visible JAX GPU backend")

    env = env_factory(
        up_config,
        down_config,
        artifact,
        runtime_naccdmax=config.runtime_naccdmax,
    )
    if env._bundle.xml_sha256 != up_config.model["xml_sha256"]:
        raise ValueError("unified pilot runtime XML identity mismatch")
    root = Path(run_root) if run_root is not None else Path(
        os.environ.get("JIT_RUN_ROOT", "JIT/runs/pi_unified")
    )
    run_dir = root / run_id
    declaration = RunDeclaration(
        run_id=run_id,
        purpose="fresh_single_policy_tube_rsi_engineering_pilot",
        output_dir=run_dir,
        config_sha256=config.config_sha256,
        xml_sha256=env._bundle.xml_sha256,
        reference_sha256=str(up_config.model["reference_sha256"]),
        training_transition_ceiling=config.ppo.requested_transitions,
        stopping_conditions=(
            "stop_after_one_exact_25600_transition_ppo_block",
            "stop_on_nonfinite_metric",
            "stop_on_checkpoint_restore_failure",
        ),
        resume_command=(
            "/home/qy/mujoco_playground/.venv/bin/python "
            "JIT/cli/train_unified.py "
            f"--config {Path(config_path)} --run-id {run_id}"
        ),
    )
    predeclare_run(declaration, resolved_config=config.raw)
    identity = checkpoint_identity(config, env)
    save_checkpoint(
        run_dir / "checkpoints/transition_0",
        CheckpointPayload(identity, 0, None, None, None),
    )
    _write_json(
        run_dir / "pilot_provenance.json",
        {
            "schema": "jit_pi_unified_pilot_provenance_v1",
            "initialization": config.raw["initialization"],
            "policy_count": 1,
            "runtime_naccdmax": config.runtime_naccdmax,
            "expert_switching_used": False,
            "test_data_used": False,
            "soft_tube_manifest_sha256": config.soft_tube_manifest_sha256,
            "tube_rsi_smoke_report_sha256": config.tube_rsi_smoke_report_sha256,
        },
    )
    mark_run_running(
        run_dir,
        process_id=os.getpid(),
        metadata={"stage": "pi_unified", "training_transitions": 0},
    )
    completed_training = 0
    progress_rows: list[dict[str, Any]] = []

    def progress(step: int, metrics: Mapping[str, Any]) -> None:
        nonlocal completed_training
        completed_training = max(completed_training, int(step))
        progress_rows.append(
            {"training_transitions": int(step), "metrics": _json_safe(metrics)}
        )

    try:
        _, params, final_metrics = trainer(
            **build_unified_trainer_kwargs(config, env, progress_fn=progress)
        )
        completed_training = max(completed_training, config.ppo.requested_transitions)
        if completed_training != config.ppo.requested_transitions:
            raise ValueError("unified PPO transition accounting mismatch")
        normalizer, actor, critic = params
        checkpoint = run_dir / f"checkpoints/transition_{completed_training}"
        save_checkpoint(
            checkpoint,
            CheckpointPayload(identity, completed_training, normalizer, actor, critic),
        )
        restored = load_checkpoint(checkpoint, expected=identity)
        if restored.training_transitions != completed_training:
            raise ValueError("unified checkpoint transition mismatch")
        metrics = _json_safe(final_metrics)
        if not metrics and progress_rows:
            metrics = dict(progress_rows[-1]["metrics"])
        if not metrics:
            raise ValueError("unified PPO produced no training metrics")
        report = {
            "schema": "jit_pi_unified_pilot_report_v1",
            "status": "completed",
            "pi_unified_pilot": "GO",
            "training_transitions": completed_training,
            "environment_interactions": completed_training,
            "brax_evaluation_transitions": 0,
            "fixed_evaluation_transitions": 0,
            "diagnostic_transitions": 0,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_restored": True,
            "initialization": config.raw["initialization"],
            "policy_count": 1,
            "runtime_naccdmax": config.runtime_naccdmax,
            "expert_switching_used": False,
            "test_data_used": False,
            "validation_data_used": False,
            "claim_boundary": {
                "engineering_integrity_only": True,
                "learnability_claim": False,
                "final_policy_claim": False,
                "certified_safe_tube_claim": False,
                "jce_jel_claim": False,
            },
            "final_metrics": metrics,
        }
        _write_json(run_dir / "pilot_report.json", report)
        with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as stream:
            for row in progress_rows:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        close_run(
            run_dir,
            status="completed",
            accounting=InteractionAccounting(completed_training, 0, 0, 0),
            reason="one fresh unified PPO block and checkpoint restore completed; engineering integrity only",
        )
        return {"run_dir": str(run_dir.resolve()), "report": report}
    except Exception as exc:
        close_run(
            run_dir,
            status="engineering_error",
            accounting=InteractionAccounting(completed_training, 0, 0, 0),
            reason=f"{type(exc).__name__}: {exc}",
        )
        raise
