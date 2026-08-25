"""Auditable one-block Brax PPO engineering runner for Phase U only."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import partial
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from brax.envs.wrappers import training as brax_training
from brax.training.agents.ppo import networks as ppo_networks
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
from .config import ResolvedConfig, load_config
from .constants import ACTION_ORDER, ACTOR_FRAME_FIELDS, ACTOR_TASK_FIELDS
from .env import TwoPhaseBikeEnv
from .evaluation import capture_episode, summarize_phase_u
from .provenance import (
    InteractionAccounting,
    RunDeclaration,
    close_run,
    predeclare_run,
)
from .video import render_trace


@dataclass(frozen=True)
class SmokeReport:
    requested_training_transitions: int
    completed_training_transitions: int
    brax_evaluation_transitions: int
    fixed_evaluation_transitions: int
    diagnostic_transitions: int
    final_metrics: Mapping[str, float]
    checkpoint_restored: bool

    @property
    def total_environment_transitions(self) -> int:
        return (
            self.completed_training_transitions
            + self.brax_evaluation_transitions
            + self.fixed_evaluation_transitions
            + self.diagnostic_transitions
        )


def validate_smoke_report(report: SmokeReport) -> SmokeReport:
    counts = (
        report.requested_training_transitions,
        report.completed_training_transitions,
        report.brax_evaluation_transitions,
        report.fixed_evaluation_transitions,
        report.diagnostic_transitions,
    )
    if any(int(value) < 0 for value in counts):
        raise ValueError("interaction counts must be nonnegative")
    if report.completed_training_transitions != report.requested_training_transitions:
        raise ValueError("smoke training must complete exactly the requested transitions")
    if not report.checkpoint_restored:
        raise ValueError("the final checkpoint restore was not verified")
    if any(not math.isfinite(float(value)) for value in report.final_metrics.values()):
        raise ValueError("final PPO metrics must be finite")
    return report


def make_network_factory():
    """Returns the fixed asymmetric Actor/critic network constructor."""

    return partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(256, 256, 256),
        value_hidden_layer_sizes=(256, 256, 256),
        policy_obs_key="state",
        value_obs_key="privileged_state",
        distribution_type="tanh_normal",
    )


_EPISODE_EVIDENCE_KEY = "AutoResetWrapper_preserve_info"


class _PreserveEpisodeEvidence(wrapper.Wrapper):
    """Copies terminal episode statistics across Playground full resets."""

    @staticmethod
    def _preserve(state):
        evidence = {
            "episode_done": state.info["episode_done"],
            "episode_metrics": state.info["episode_metrics"],
        }
        return state.replace(
            info={**state.info, _EPISODE_EVIDENCE_KEY: evidence}
        )

    def reset(self, rng):
        return self._preserve(self.env.reset(rng))

    def step(self, state, action):
        return self._preserve(self.env.step(state, action))


class _ExposeEpisodeEvidence(wrapper.Wrapper):
    """Restores the terminal statistics expected by Brax's metric logger."""

    @staticmethod
    def _expose(state):
        evidence = state.info[_EPISODE_EVIDENCE_KEY]
        return state.replace(
            info={
                **state.info,
                "episode_done": evidence["episode_done"],
                "episode_metrics": evidence["episode_metrics"],
            }
        )

    def reset(self, rng):
        return self._expose(self.env.reset(rng))

    def step(self, state, action):
        return self._expose(self.env.step(state, action))


def wrap_for_jit_training(
    env: Any,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Any = None,
):
    """Uses real resets so JIT episode events/counters never leak across done."""

    if randomization_fn is None:
        env = brax_training.VmapWrapper(env)
    else:
        env = wrapper.BraxDomainRandomizationVmapWrapper(
            env, randomization_fn
        )
    env = brax_training.EpisodeWrapper(env, episode_length, action_repeat)
    env = _PreserveEpisodeEvidence(env)
    env = wrapper.BraxAutoResetWrapper(env, full_reset=True)
    return _ExposeEpisodeEvidence(env)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    array = np.asarray(jax.device_get(value))
    if array.ndim == 0:
        scalar = array.item()
        if isinstance(scalar, float) and not math.isfinite(scalar):
            raise ValueError("nonfinite metric encountered")
        return scalar
    return array.tolist()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _identity(config: ResolvedConfig, xml_sha256: str) -> CheckpointIdentity:
    return CheckpointIdentity(
        config_sha256=config.config_sha256,
        xml_sha256=xml_sha256,
        actor_frame_fields=ACTOR_FRAME_FIELDS,
        actor_task_fields=ACTOR_TASK_FIELDS,
        action_order=ACTION_ORDER,
    )


def _default_run_root() -> Path:
    return Path(__file__).resolve().parents[3] / "JIT/runs/phase_u"


def run_phase_u_smoke(
    config_path: Path,
    run_id: str,
    *,
    run_root: Path | None = None,
) -> dict[str, Any]:
    """Runs exactly one declared PPO block and one counted diagnostic rollout."""

    config = load_config(Path(config_path))
    if config.phase != "propulsion_ascent":
        raise ValueError("only propulsion_ascent is implemented")
    if config.ppo.requested_transitions != config.ppo.block_transitions:
        raise ValueError("engineering smoke must contain exactly one PPO block")
    if jax.default_backend() != "gpu":
        raise RuntimeError("the Phase U smoke requires the visible JAX GPU backend")

    env = TwoPhaseBikeEnv(config)
    root = Path(run_root) if run_root is not None else Path(
        os.environ.get("JIT_RUN_ROOT", _default_run_root())
    )
    run_dir = root / run_id
    declaration = RunDeclaration(
        run_id=run_id,
        purpose="compile_update_checkpoint_restore_engineering_smoke",
        output_dir=run_dir,
        config_sha256=config.config_sha256,
        xml_sha256=env._bundle.xml_sha256,
        reference_sha256=str(config.model["reference_sha256"]),
        training_transition_ceiling=config.ppo.requested_transitions,
        stopping_conditions=(
            "stop_after_one_exact_ppo_block",
            "stop_on_nonfinite_metric",
            "stop_on_checkpoint_restore_failure",
        ),
        resume_command=(
            "/home/qy/mujoco_playground/.venv/bin/python "
            "JIT/cli/train_phase_expert.py --phase propulsion_ascent "
            f"--config {Path(config_path)} --run-id {run_id} --smoke"
        ),
    )
    predeclare_run(declaration, resolved_config=config.raw)
    _write_json(
        run_dir / "backend.json",
        {
            "jax_backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
        },
    )
    identity = _identity(config, env._bundle.xml_sha256)
    save_checkpoint(
        run_dir / "checkpoints/transition_0",
        CheckpointPayload(
            identity=identity,
            training_transitions=0,
            observation_normalizer=None,
            actor_params=None,
            critic_params=None,
        ),
    )

    progress_rows: list[dict[str, Any]] = []
    completed_training = 0
    diagnostic_transitions = 0

    def progress(step: int, metrics: Mapping[str, Any]) -> None:
        nonlocal completed_training
        completed_training = max(completed_training, int(step))
        progress_rows.append({"training_transitions": int(step), "metrics": metrics})

    try:
        make_policy, params, final_metrics = ppo_train.train(
            environment=env,
            num_timesteps=config.ppo.requested_transitions,
            max_devices_per_host=1,
            wrap_env=True,
            wrap_env_fn=wrap_for_jit_training,
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
            seed=config.ppo.seed,
            num_evals=config.ppo.num_evals,
            num_eval_envs=config.ppo.num_eval_envs,
            deterministic_eval=True,
            log_training_metrics=True,
            progress_fn=progress,
            run_evals=False,
        )
        completed_training = max(completed_training, config.ppo.requested_transitions)
        normalizer_params, actor_params, critic_params = params
        final_checkpoint = run_dir / f"checkpoints/transition_{completed_training}"
        save_checkpoint(
            final_checkpoint,
            CheckpointPayload(
                identity=identity,
                training_transitions=completed_training,
                observation_normalizer=normalizer_params,
                actor_params=actor_params,
                critic_params=critic_params,
            ),
        )
        restored = load_checkpoint(final_checkpoint, expected=identity)
        restored_params = (
            restored.observation_normalizer,
            restored.actor_params,
            restored.critic_params,
        )
        deterministic_policy = make_policy(restored_params, deterministic=True)
        policy_key = jax.random.PRNGKey(config.ppo.held_out_seeds[0])

        def diagnostic_policy(observation):
            action, _ = deterministic_policy(observation, policy_key)
            return action

        trace = capture_episode(
            env,
            diagnostic_policy,
            seed=config.ppo.held_out_seeds[0],
            horizon=config.ppo.episode_horizon,
            reset_fn=jax.jit(env.reset_natural),
            step_fn=jax.jit(env.step),
        )
        diagnostic_transitions = trace.environment_transitions
        diagnostic_summary = summarize_phase_u((trace,))
        _write_json(run_dir / "diagnostic_summary.json", diagnostic_summary)
        video_report = render_trace(
            env,
            trace,
            run_dir / "diagnostic.mp4",
            fps=50,
            reward_scaling=config.ppo.reward_scaling,
        )
        _write_json(run_dir / "video_report.json", asdict(video_report))

        flat_metrics = {
            str(key): float(np.asarray(jax.device_get(value)))
            for key, value in final_metrics.items()
            if np.asarray(jax.device_get(value)).ndim == 0
        }
        report = validate_smoke_report(
            SmokeReport(
                requested_training_transitions=config.ppo.requested_transitions,
                completed_training_transitions=completed_training,
                brax_evaluation_transitions=0,
                fixed_evaluation_transitions=0,
                diagnostic_transitions=diagnostic_transitions,
                final_metrics=flat_metrics,
                checkpoint_restored=True,
            )
        )
        _write_json(run_dir / "smoke_report.json", asdict(report))
        with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as stream:
            for row in progress_rows:
                stream.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")
        close_run(
            run_dir,
            status="completed",
            accounting=InteractionAccounting(
                training=completed_training,
                brax_evaluation=0,
                fixed_evaluation=0,
                diagnostic=diagnostic_transitions,
            ),
            reason="one exact PPO block and restored-policy diagnostic completed; engineering integrity only",
        )
        return {
            "run_dir": str(run_dir.resolve()),
            "smoke_report": _json_safe(asdict(report)),
            "diagnostic_summary": diagnostic_summary,
        }
    except Exception as exc:
        close_run(
            run_dir,
            status="engineering_error",
            accounting=InteractionAccounting(
                training=completed_training,
                brax_evaluation=0,
                fixed_evaluation=0,
                diagnostic=diagnostic_transitions,
            ),
            reason=f"{type(exc).__name__}: {exc}",
        )
        raise
