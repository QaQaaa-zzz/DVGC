"""Auditable formal Propulsion-Ascent PPO orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from brax.training.agents.ppo import train as ppo_train
import jax
from mujoco_playground._src import wrapper
import numpy as np

from .checkpoint import (
    CheckpointIdentity,
    CheckpointPayload,
    load_checkpoint,
    save_checkpoint,
)
from .config import ResolvedConfig, file_sha256, load_config
from .constants import ACTION_ORDER, ACTOR_FRAME_FIELDS
from .env import TwoPhaseBikeEnv
from .evaluation import capture_episode, save_episode_trace, summarize_phase_u
from .ppo import make_network_factory
from .provenance import (
    InteractionAccounting,
    RunDeclaration,
    close_run,
    mark_run_running,
    predeclare_run,
)
from .video import render_trace


@dataclass(frozen=True)
class PanelResult:
    absolute_transition: int
    environment_transitions: int
    summary: Mapping[str, Any]


@dataclass(frozen=True)
class FormalReport:
    requested_training_transitions: int
    starting_training_transition: int
    completed_training_transitions: int
    segment_training_transitions: int
    brax_evaluation_transitions: int
    fixed_evaluation_transitions: int
    checkpoint_transitions: tuple[int, ...]
    evaluated_transitions: tuple[int, ...]
    final_metrics: Mapping[str, float]
    checkpoint_restored: bool
    resume_semantics: str

    @property
    def total_environment_transitions(self) -> int:
        return (
            self.segment_training_transitions
            + self.brax_evaluation_transitions
            + self.fixed_evaluation_transitions
        )


def validate_formal_report(report: FormalReport) -> FormalReport:
    if report.requested_training_transitions != 998_400:
        raise ValueError("formal requested target must equal 998400")
    if report.completed_training_transitions != report.requested_training_transitions:
        raise ValueError("formal run did not reach the absolute target")
    expected_segment = (
        report.completed_training_transitions - report.starting_training_transition
    )
    if report.segment_training_transitions != expected_segment:
        raise ValueError("formal segment training count mismatch")
    if report.brax_evaluation_transitions != 0:
        raise ValueError("formal Brax evaluation transitions must remain zero")
    full_checkpoints = (0, 102_400, 256_000, 512_000, 742_400, 998_400)
    expected_checkpoints = tuple(
        step for step in full_checkpoints if step >= report.starting_training_transition
    )
    if report.checkpoint_transitions != expected_checkpoints:
        raise ValueError("formal checkpoint schedule mismatch")
    full_evaluations = full_checkpoints[1:]
    expected_evaluations = tuple(
        step for step in full_evaluations if step > report.starting_training_transition
    )
    if report.evaluated_transitions != expected_evaluations:
        raise ValueError("formal evaluation schedule mismatch")
    minimum_fixed = 8 * len(expected_evaluations)
    maximum_fixed = 8 * 200 * len(expected_evaluations)
    if not minimum_fixed <= report.fixed_evaluation_transitions <= maximum_fixed:
        raise ValueError("formal fixed evaluation transition count is invalid")
    for value in report.final_metrics.values():
        if not math.isfinite(float(value)):
            raise ValueError("formal report contains a nonfinite metric")
    if not report.checkpoint_restored:
        raise ValueError("formal final checkpoint restore was not verified")
    expected_resume = (
        "fresh"
        if report.starting_training_transition == 0
        else "parameter_warm_start_optimizer_reset"
    )
    if report.resume_semantics != expected_resume:
        raise ValueError("formal report resume semantics mismatch")
    return report


def _flatten_finite_metrics(
    metrics: Mapping[str, Any], prefix: str = ""
) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for key, value in metrics.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten_finite_metrics(value, name))
            continue
        array = np.asarray(jax.device_get(value))
        if array.ndim != 0:
            continue
        scalar = float(array)
        if not math.isfinite(scalar):
            raise ValueError(f"nonfinite metric encountered: {name}")
        flattened[name] = scalar
    return flattened


class FormalRunController:
    """Converts Brax-relative callbacks into absolute formal-run evidence."""

    def __init__(
        self,
        *,
        config: ResolvedConfig,
        run_dir: Path,
        identity: CheckpointIdentity,
        starting_training_transition: int,
        evaluate_panel: Callable[[int, Any, Any], PanelResult],
        checkpoint_saver: Callable[[Path, CheckpointPayload], None] = save_checkpoint,
        checkpoint_loader: Callable[..., CheckpointPayload] = load_checkpoint,
    ) -> None:
        if config.formal is None:
            raise ValueError("formal controller requires a formal config")
        if starting_training_transition < 0:
            raise ValueError("starting training transition must be nonnegative")
        if starting_training_transition % config.ppo.block_transitions:
            raise ValueError("starting training transition must be block-aligned")
        if starting_training_transition >= config.ppo.requested_transitions:
            raise ValueError("starting training transition must precede target")
        self.config = config
        self.run_dir = Path(run_dir)
        self.identity = identity
        self.starting_training_transition = int(starting_training_transition)
        self.evaluate_panel = evaluate_panel
        self.checkpoint_saver = checkpoint_saver
        self.checkpoint_loader = checkpoint_loader
        self._last_policy_relative: int | None = None
        self._last_progress_relative = 0
        self._completed_segment = 0
        self._checkpoint_transitions: list[int] = []
        self._evaluated_transitions: list[int] = []
        self._fixed_evaluation_transitions = 0
        self._last_metrics: dict[str, float] = {}
        self.run_dir.mkdir(parents=True, exist_ok=True)

    @property
    def completed_training_transitions(self) -> int:
        return self.starting_training_transition + self._completed_segment

    @property
    def segment_training_transitions(self) -> int:
        return self._completed_segment

    @property
    def checkpoint_transitions(self) -> tuple[int, ...]:
        return tuple(self._checkpoint_transitions)

    @property
    def evaluated_transitions(self) -> tuple[int, ...]:
        return tuple(self._evaluated_transitions)

    @property
    def fixed_evaluation_transitions(self) -> int:
        return self._fixed_evaluation_transitions

    @property
    def final_metrics(self) -> Mapping[str, float]:
        return dict(self._last_metrics)

    def on_policy_params(self, relative_step: int, make_policy: Any, params: Any) -> None:
        relative = int(relative_step)
        expected = (
            0
            if self._last_policy_relative is None
            else self._last_policy_relative + self.config.ppo.block_transitions
        )
        if relative != expected:
            raise ValueError(
                f"policy callbacks must be consecutive aligned blocks: expected {expected}, got {relative}"
            )
        absolute = self.starting_training_transition + relative
        if absolute > self.config.ppo.requested_transitions:
            raise ValueError("policy callback exceeded the formal target")
        self._last_policy_relative = relative
        self._completed_segment = max(self._completed_segment, relative)

        formal = self.config.formal
        assert formal is not None
        if absolute in formal.checkpoint_transitions:
            self.checkpoint_saver(
                self.run_dir / "checkpoints" / f"transition_{absolute}",
                CheckpointPayload(
                    identity=self.identity,
                    training_transitions=absolute,
                    observation_normalizer=params[0],
                    actor_params=params[1],
                    critic_params=params[2],
                ),
            )
            self._checkpoint_transitions.append(absolute)
        if (
            absolute in formal.fixed_evaluation_transitions
            and absolute > self.starting_training_transition
        ):
            result = self.evaluate_panel(absolute, make_policy, params)
            if result.absolute_transition != absolute:
                raise ValueError("evaluation panel transition mismatch")
            if result.environment_transitions <= 0:
                raise ValueError("evaluation panel must consume positive transitions")
            self._evaluated_transitions.append(absolute)
            self._fixed_evaluation_transitions += result.environment_transitions

    def on_progress(self, relative_step: int, metrics: Mapping[str, Any]) -> None:
        relative = int(relative_step)
        if self._last_policy_relative != relative or relative <= self._last_progress_relative:
            raise ValueError("progress callback does not match the latest policy callback")
        flattened = _flatten_finite_metrics(metrics)
        absolute = self.starting_training_transition + relative
        row = {"training_transitions": absolute, "metrics": flattened}
        with (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        self._last_progress_relative = relative
        self._completed_segment = max(self._completed_segment, relative)
        self._last_metrics = flattened

    def restore_final_checkpoint(self) -> CheckpointPayload:
        target = self.config.ppo.requested_transitions
        return self.checkpoint_loader(
            self.run_dir / "checkpoints" / f"transition_{target}",
            expected=self.identity,
        )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _default_run_root() -> Path:
    return Path(__file__).resolve().parents[3] / "JIT/runs/phase_u"


def _checkpoint_identity(
    config: ResolvedConfig, xml_sha256: str
) -> CheckpointIdentity:
    return CheckpointIdentity(
        config_sha256=config.config_sha256,
        xml_sha256=xml_sha256,
        actor_frame_fields=ACTOR_FRAME_FIELDS,
        action_order=ACTION_ORDER,
    )


def _evaluate_fixed_panel(
    env: Any,
    run_dir: Path,
    config: ResolvedConfig,
    absolute_transition: int,
    make_policy: Any,
    params: Any,
) -> PanelResult:
    deterministic_policy = make_policy(params, deterministic=True)
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    traces = []
    artifacts = []
    panel_dir = Path(run_dir) / "evaluations" / f"transition_{absolute_transition}"
    panel_dir.mkdir(parents=True, exist_ok=False)
    for seed in config.ppo.held_out_seeds:
        policy_key = jax.random.PRNGKey(seed)

        def policy(observation, *, _key=policy_key):
            action, _ = deterministic_policy(observation, _key)
            return action

        trace = capture_episode(
            env,
            policy,
            seed=seed,
            horizon=config.ppo.episode_horizon,
            reset_fn=reset_fn,
            step_fn=step_fn,
        )
        traces.append(trace)
        artifact = save_episode_trace(trace, panel_dir / f"seed_{seed}")
        artifacts.append(
            {
                "seed": seed,
                "metadata_path": str(artifact.metadata_path),
                "npz_path": str(artifact.npz_path),
                "npz_sha256": artifact.npz_sha256,
                "environment_transitions": artifact.environment_transitions,
                "captured_state_count": artifact.captured_state_count,
            }
        )
    summary = summarize_phase_u(tuple(traces))
    summary.update(
        {
            "absolute_transition": absolute_transition,
            "held_out_seeds": list(config.ppo.held_out_seeds),
            "trace_artifacts": artifacts,
        }
    )
    _write_json(panel_dir / "summary.json", summary)
    if absolute_transition == config.ppo.requested_transitions:
        representative = next(
            (trace for trace in traces if trace.frames[-1].success), traces[0]
        )
        video_report = render_trace(
            env,
            representative,
            panel_dir / "representative.mp4",
            fps=50,
        )
        _write_json(panel_dir / "video_report.json", asdict(video_report))
    return PanelResult(
        absolute_transition=absolute_transition,
        environment_transitions=int(summary["environment_transitions"]),
        summary=summary,
    )


def _verify_restored_inference(
    env: Any,
    make_policy: Any,
    params: Any,
    *,
    seed: int,
) -> None:
    state = env.reset(jax.random.PRNGKey(seed))
    policy = make_policy(params, deterministic=True)
    action, _ = policy(state.obs, jax.random.PRNGKey(seed))
    array = np.asarray(jax.device_get(action))
    if array.shape != (4,) or not np.all(np.isfinite(array)):
        raise ValueError("restored deterministic Actor inference is invalid")


def run_phase_u_formal(
    config_path: Path,
    run_id: str,
    *,
    restore_checkpoint: Path | None = None,
    run_root: Path | None = None,
    trainer: Callable[..., Any] = ppo_train.train,
    env_factory: Callable[[ResolvedConfig], Any] = TwoPhaseBikeEnv,
    panel_evaluator: Callable[..., PanelResult] | None = None,
    backend_name: Callable[[], str] = jax.default_backend,
) -> dict[str, Any]:
    """Runs one formal segment; normal operation is a single 998,400-step segment."""

    config = load_config(Path(config_path))
    if config.formal is None:
        raise ValueError("formal training requires the formal config schema")
    if backend_name() != "gpu":
        raise RuntimeError("formal Phase U training requires the visible JAX GPU backend")

    repository_root = Path(__file__).resolve().parents[3]
    xml_path = repository_root / str(config.model["xml_path"])
    reference_path = repository_root / str(config.model["reference_path"])
    if file_sha256(xml_path) != config.model["xml_sha256"]:
        raise ValueError("authoritative XML identity drift")
    if file_sha256(reference_path) != config.model["reference_sha256"]:
        raise ValueError("reference trajectory identity drift")

    env = env_factory(config)
    identity = _checkpoint_identity(config, env._bundle.xml_sha256)
    restored_payload: CheckpointPayload | None = None
    starting_transition = 0
    parent_checkpoint: str | None = None
    resume_semantics = "fresh"
    if restore_checkpoint is not None:
        parent_path = Path(restore_checkpoint).resolve()
        restored_payload = load_checkpoint(parent_path, expected=identity)
        starting_transition = int(restored_payload.training_transitions)
        parent_checkpoint = str(parent_path)
        resume_semantics = config.formal.resume_semantics
    if starting_transition % config.ppo.block_transitions:
        raise ValueError("restore checkpoint transition must be block-aligned")
    if starting_transition >= config.ppo.requested_transitions:
        raise ValueError("restore checkpoint must precede the formal target")

    remaining = config.ppo.requested_transitions - starting_transition
    segment_seed = config.ppo.seed + starting_transition // config.ppo.block_transitions
    root = Path(run_root) if run_root is not None else Path(
        os.environ.get("JIT_RUN_ROOT", _default_run_root())
    )
    run_dir = root / run_id
    resume_suffix = (
        f" --restore-checkpoint {parent_checkpoint}" if parent_checkpoint else ""
    )
    declaration = RunDeclaration(
        run_id=run_id,
        purpose="formal_propulsion_ascent_ppo",
        output_dir=run_dir,
        config_sha256=config.config_sha256,
        xml_sha256=env._bundle.xml_sha256,
        reference_sha256=str(config.model["reference_sha256"]),
        training_transition_ceiling=remaining,
        stopping_conditions=(
            "stop_at_absolute_transition_998400",
            "stop_on_nonfinite_metric",
            "stop_on_cuda_or_oom_error",
            "stop_on_checkpoint_identity_or_restore_failure",
            "stop_on_trace_persistence_failure",
        ),
        resume_command=(
            "/home/qy/mujoco_playground/.venv/bin/python "
            "JIT/cli/train_phase_expert.py --phase propulsion_ascent "
            f"--config {Path(config_path)} --run-id {run_id} --formal{resume_suffix}"
        ),
        parent_checkpoint=parent_checkpoint,
        starting_training_transition=starting_transition,
        resume_semantics=resume_semantics,
        segment_seed=segment_seed,
    )
    predeclare_run(declaration, resolved_config=config.raw)
    _write_json(
        run_dir / "backend.json",
        {
            "jax_backend": backend_name(),
            "devices": [str(device) for device in jax.devices()],
        },
    )
    mark_run_running(
        run_dir,
        process_id=os.getpid(),
        metadata={
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "training_seed": config.ppo.seed,
            "segment_seed": segment_seed,
            "starting_training_transition": starting_transition,
            "target_training_transition": config.ppo.requested_transitions,
            "resume_semantics": resume_semantics,
        },
    )

    evaluator = panel_evaluator or _evaluate_fixed_panel

    def evaluate_panel(step: int, make_policy: Any, params: Any) -> PanelResult:
        return evaluator(env, run_dir, config, step, make_policy, params)

    controller = FormalRunController(
        config=config,
        run_dir=run_dir,
        identity=identity,
        starting_training_transition=starting_transition,
        evaluate_panel=evaluate_panel,
    )
    restore_params = None
    if restored_payload is not None:
        restore_params = (
            restored_payload.observation_normalizer,
            restored_payload.actor_params,
            restored_payload.critic_params,
        )

    try:
        make_policy, _params, final_metrics = trainer(
            environment=env,
            num_timesteps=remaining,
            max_devices_per_host=1,
            wrap_env=True,
            wrap_env_fn=wrapper.wrap_for_brax_training,
            num_envs=config.ppo.num_parallel_envs,
            episode_length=config.ppo.episode_horizon,
            action_repeat=1,
            learning_rate=config.ppo.learning_rate,
            entropy_cost=config.ppo.entropy_cost,
            discounting=config.ppo.discounting,
            unroll_length=config.ppo.unroll_length,
            batch_size=config.ppo.batch_size,
            num_minibatches=config.ppo.num_minibatches,
            num_updates_per_batch=config.ppo.num_updates_per_batch,
            normalize_observations=True,
            reward_scaling=config.ppo.reward_scaling,
            clipping_epsilon=config.ppo.clipping_epsilon,
            gae_lambda=config.ppo.gae_lambda,
            max_grad_norm=config.ppo.max_grad_norm,
            bootstrap_on_timeout=True,
            network_factory=make_network_factory(),
            seed=segment_seed,
            num_evals=remaining // config.ppo.block_transitions + 1,
            num_eval_envs=config.ppo.num_eval_envs,
            deterministic_eval=True,
            log_training_metrics=True,
            progress_fn=controller.on_progress,
            policy_params_fn=controller.on_policy_params,
            restore_params=restore_params,
            run_evals=False,
        )
        if controller.completed_training_transitions != config.ppo.requested_transitions:
            raise ValueError("formal trainer returned before the absolute target")
        restored = controller.restore_final_checkpoint()
        restored_params = (
            restored.observation_normalizer,
            restored.actor_params,
            restored.critic_params,
        )
        _verify_restored_inference(
            env,
            make_policy,
            restored_params,
            seed=config.ppo.held_out_seeds[0],
        )
        flattened_final = _flatten_finite_metrics(final_metrics)
        report = validate_formal_report(
            FormalReport(
                requested_training_transitions=config.ppo.requested_transitions,
                starting_training_transition=starting_transition,
                completed_training_transitions=controller.completed_training_transitions,
                segment_training_transitions=controller.segment_training_transitions,
                brax_evaluation_transitions=0,
                fixed_evaluation_transitions=controller.fixed_evaluation_transitions,
                checkpoint_transitions=controller.checkpoint_transitions,
                evaluated_transitions=controller.evaluated_transitions,
                final_metrics=flattened_final,
                checkpoint_restored=True,
                resume_semantics=resume_semantics,
            )
        )
        _write_json(run_dir / "formal_report.json", asdict(report))
        close_run(
            run_dir,
            status="completed",
            accounting=InteractionAccounting(
                training=controller.segment_training_transitions,
                brax_evaluation=0,
                fixed_evaluation=controller.fixed_evaluation_transitions,
                diagnostic=0,
            ),
            reason=(
                "formal Phase U target and frozen held-out panels completed; "
                "trained-expert promotion remains evaluation-dependent"
            ),
        )
        return {
            "run_dir": str(run_dir.resolve()),
            "formal_report": asdict(report),
        }
    except Exception as exc:
        close_run(
            run_dir,
            status="engineering_error",
            accounting=InteractionAccounting(
                training=controller.segment_training_transitions,
                brax_evaluation=0,
                fixed_evaluation=controller.fixed_evaluation_transitions,
                diagnostic=0,
            ),
            reason=f"{type(exc).__name__}: {exc}",
        )
        raise
