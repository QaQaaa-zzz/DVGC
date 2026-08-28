"""Round-1 unified PPO with natural-reset coverage as the sole training change.

The Round-0 causal diagnostic showed that frozen pi_up_star reaches Apex from
the canonical natural start while the 10M unified policy fails before the jump
zone.  This module therefore changes exactly one method variable: per-episode
reset distribution.  Ten percent of episodes use the existing Phase-U natural
reset and ninety percent keep the original learned Soft-Tube RSI sampler.
Everything else is inherited from the locked unified formal contract.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from brax.training.agents.ppo import train as ppo_train
import jax
from jax import numpy as jp
import numpy as np

from .descent_semantics import initial_descent_events
from .env import TwoPhaseBikeEnv
from .formal_training import PanelResult
from .provenance import (
    InteractionAccounting,
    RunDeclaration,
    close_run,
    mark_run_running,
    predeclare_run,
)
from .tube_rsi import PHASE_DOWNSTREAM, PHASE_UPSTREAM
from .unified_diagnostic import _load_runtime
from .unified_env import UnifiedTubeRSIEnv
from .unified_formal import (
    UnifiedFormalController,
    _evaluate_train_panel,
    _verify_restored_policy,
    _write_json,
    build_unified_formal_trainer_kwargs,
    load_unified_formal_config,
)
from .unified_training import checkpoint_identity, read_json


ROUND1_RESET_SCHEMA = "jit_pi_unified_round1_reset_mix_v1"
ROUND1_NATURAL_PROBABILITY = 0.10
ROUND1_SOFT_TUBE_PROBABILITY = 0.90
ROUND1_PURPOSE = "formal_pi_unified_round1_mixed_reset_ppo"


def validate_round1_reset_contract(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    mix = payload.get("reset_mixture")
    if not isinstance(mix, dict):
        raise ValueError("Round-1 config is missing reset_mixture")
    expected = {
        "schema": ROUND1_RESET_SCHEMA,
        "selection": "bernoulli_per_episode",
        "natural_reset_probability": ROUND1_NATURAL_PROBABILITY,
        "soft_tube_probability": ROUND1_SOFT_TUBE_PROBABILITY,
        "natural_reset_semantics": "existing_phase_u_natural_reset",
        "soft_tube_semantics": "existing_phase_balanced_value_weighted_tube_rsi",
        "single_variable": "reset_distribution_only",
    }
    if mix != expected:
        raise ValueError("Round-1 reset-mixture contract drift")
    if not math.isclose(
        float(mix["natural_reset_probability"])
        + float(mix["soft_tube_probability"]),
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("Round-1 reset probabilities must sum to one")
    boundary = payload.get("claim_boundary", {})
    if boundary.get("round1_single_variable_iteration") is not True:
        raise ValueError("Round-1 single-variable claim boundary is missing")
    if boundary.get("round0_failure_evidence") != "PI_UP_APEX_UNIFIED_PRE_JUMP_FAIL":
        raise ValueError("Round-1 config is not bound to the locked Round-0 diagnosis")
    return mix


def load_round1_config(path: Path):
    raw = read_json(Path(path))
    validate_round1_reset_contract(raw)
    return load_unified_formal_config(Path(path))


class Round1MixedResetUnifiedEnv(UnifiedTubeRSIEnv):
    """Unified environment with fixed 10% natural / 90% Soft-Tube resets."""

    def __init__(
        self,
        up_config,
        down_config,
        artifact,
        *,
        runtime_naccdmax: int | None = None,
        natural_reset_probability: float = ROUND1_NATURAL_PROBABILITY,
    ):
        probability = float(natural_reset_probability)
        if not math.isclose(
            probability,
            ROUND1_NATURAL_PROBABILITY,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("Round-1 natural reset probability is locked at 0.10")
        self._round1_natural_reset_probability = probability
        super().__init__(
            up_config,
            down_config,
            artifact,
            runtime_naccdmax=runtime_naccdmax,
        )

    def _mark_round1_reset_source(self, state, *, soft_tube: jax.Array | bool):
        soft = jp.asarray(soft_tube, dtype=bool)
        natural = ~soft
        start_phase = jp.asarray(state.info["start_phase"], dtype=jp.int32)
        info = {
            **state.info,
            "round1_reset_from_soft_tube": soft,
        }
        metrics = {
            **state.metrics,
            "reset/source_soft_tube": soft.astype(jp.float32),
            "reset/source_natural": natural.astype(jp.float32),
            "reset/tube_phase_upstream": (
                soft & (start_phase == PHASE_UPSTREAM)
            ).astype(jp.float32),
            "reset/tube_phase_downstream": (
                soft & (start_phase == PHASE_DOWNSTREAM)
            ).astype(jp.float32),
        }
        return state.replace(info=info, metrics=metrics)

    def _reset_round1_tube(self, rng: jax.Array):
        state = self._reset_from_tube_sample(self._tube_pool.sample(rng))
        return self._mark_round1_reset_source(state, soft_tube=True)

    def _reset_round1_natural(self, rng: jax.Array):
        phase_state = TwoPhaseBikeEnv.reset_natural(self, rng)
        up_events = phase_state.info["events"]
        active_phase = jp.asarray(PHASE_UPSTREAM, jp.int32)
        false = jp.asarray(False)
        root_x = phase_state.data.qpos[self._bundle.model_index.root_qpos_address]

        # Match UnifiedTubeRSIEnv's info pytree exactly, plus one Round-1 source
        # flag that is also present on the Tube branch.  Do not retain the
        # Phase-U-only reset_source_airborne_rsi field.
        info = {
            "rng": phase_state.info["rng"],
            "history": phase_state.info["history"],
            "up_events": up_events,
            "down_events": initial_descent_events(root_x),
            "active_phase": active_phase,
            "start_phase": active_phase,
            "phase_transitioned": false,
            "expert_switching_used": false,
            "last_action": phase_state.info["last_action"],
            "reward_state": phase_state.info["reward_state"],
            "episode_step": phase_state.info["episode_step"],
            "phase_episode_step": jp.asarray(0, jp.int32),
            "source_tick": jp.asarray(-1, jp.int32),
            "parent_group_index": jp.asarray(-1, jp.int32),
            "tube_entry_index": jp.asarray(-1, jp.int32),
            "tube_global_index": jp.asarray(-1, jp.int32),
            "terminated": phase_state.info["terminated"],
            "truncated": phase_state.info["truncated"],
            "time_out": phase_state.info["time_out"],
            "end_code": phase_state.info["end_code"],
            "success": phase_state.info["success"],
            "physical_failure": phase_state.info["physical_failure"],
            "roll_limit": phase_state.info["roll_limit"],
            "pitch_limit": phase_state.info["pitch_limit"],
            "jump_zone_missed": phase_state.info["jump_zone_missed"],
            "stuck": phase_state.info["stuck"],
            "yaw_limit": phase_state.info["yaw_limit"],
            "timeout": phase_state.info["timeout"],
            "episode_return": phase_state.info["episode_return"],
        }
        metrics = self._unified_zero_metrics()
        for key, value in phase_state.metrics.items():
            if key in metrics:
                metrics[key] = value
        metrics.update(
            {
                "reset/source_soft_tube": jp.asarray(0.0, jp.float32),
                "reset/tube_phase_upstream": jp.asarray(0.0, jp.float32),
                "reset/tube_phase_downstream": jp.asarray(0.0, jp.float32),
                "event/tube_phase_transition": jp.asarray(0.0, jp.float32),
                "state/active_phase": active_phase.astype(jp.float32),
            }
        )
        state = phase_state.replace(info=info, metrics=metrics)
        return self._mark_round1_reset_source(state, soft_tube=False)

    def reset(self, rng: jax.Array):
        decision_key, reset_key = jax.random.split(rng)
        use_natural = jax.random.bernoulli(
            decision_key, self._round1_natural_reset_probability
        )
        return jax.lax.cond(
            use_natural,
            self._reset_round1_natural,
            self._reset_round1_tube,
            reset_key,
        )

    def reset_tube_index(self, phase_index, entry_index):
        state = super().reset_tube_index(phase_index, entry_index)
        return self._mark_round1_reset_source(state, soft_tube=True)

    def step(self, state, action):
        next_state = super().step(state, action)
        soft = jp.asarray(
            state.info["round1_reset_from_soft_tube"], dtype=bool
        )
        natural = ~soft
        start_phase = jp.asarray(state.info["start_phase"], dtype=jp.int32)
        metrics = {
            **next_state.metrics,
            "reset/source_soft_tube": soft.astype(jp.float32),
            "reset/source_natural": natural.astype(jp.float32),
            "reset/tube_phase_upstream": (
                soft & (start_phase == PHASE_UPSTREAM)
            ).astype(jp.float32),
            "reset/tube_phase_downstream": (
                soft & (start_phase == PHASE_DOWNSTREAM)
            ).astype(jp.float32),
        }
        return next_state.replace(metrics=metrics)


def audit_round1_reset_sampler(env: Round1MixedResetUnifiedEnv, *, sample_count: int = 256):
    """Compile reset only and prove both source branches share one valid pytree."""
    if sample_count <= 1:
        raise ValueError("Round-1 reset smoke requires multiple samples")
    reset = jax.jit(env.reset)
    counts = {"natural": 0, "soft_tube": 0}
    phase_counts = {"downstream": 0, "upstream": 0}
    for seed in range(9_500_001, 9_500_001 + int(sample_count)):
        state = reset(jax.random.PRNGKey(seed))
        jax.block_until_ready(state)
        qpos = np.asarray(jax.device_get(state.data.qpos))
        qvel = np.asarray(jax.device_get(state.data.qvel))
        if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
            raise ValueError("Round-1 reset smoke produced a nonfinite state")
        if bool(np.asarray(jax.device_get(state.info["expert_switching_used"]))):
            raise ValueError("Round-1 reset smoke used expert switching")
        natural = float(
            np.asarray(jax.device_get(state.metrics["reset/source_natural"]))
        )
        soft = float(
            np.asarray(jax.device_get(state.metrics["reset/source_soft_tube"]))
        )
        if (natural, soft) not in {(0.0, 1.0), (1.0, 0.0)}:
            raise ValueError("Round-1 reset source flags are not one-hot")
        if natural == 1.0:
            counts["natural"] += 1
            if int(np.asarray(jax.device_get(state.info["active_phase"]))) != PHASE_UPSTREAM:
                raise ValueError("natural Round-1 reset must start upstream")
        else:
            counts["soft_tube"] += 1
            phase = int(np.asarray(jax.device_get(state.info["active_phase"])))
            phase_counts["upstream" if phase == PHASE_UPSTREAM else "downstream"] += 1
    if counts["natural"] == 0 or counts["soft_tube"] == 0:
        raise ValueError("Round-1 fixed reset smoke did not exercise both source branches")
    return {
        "schema": "jit_pi_unified_round1_reset_smoke_v1",
        "status": "completed",
        "sample_count": int(sample_count),
        "natural_count": counts["natural"],
        "soft_tube_count": counts["soft_tube"],
        "observed_natural_fraction": counts["natural"] / float(sample_count),
        "soft_tube_phase_counts": phase_counts,
        "configured_natural_probability": ROUND1_NATURAL_PROBABILITY,
        "configured_soft_tube_probability": ROUND1_SOFT_TUBE_PROBABILITY,
        "environment_interactions": 0,
        "training_transitions": 0,
        "expert_switching_used": False,
        "validation_data_used": False,
        "test_data_used": False,
    }


def build_round1_environment(config_path: Path):
    config = load_round1_config(Path(config_path))
    up_config, down_config, artifact, _ = _load_runtime(config)
    env = Round1MixedResetUnifiedEnv(
        up_config,
        down_config,
        artifact,
        runtime_naccdmax=config.runtime_naccdmax,
        natural_reset_probability=ROUND1_NATURAL_PROBABILITY,
    )
    if env._bundle.xml_sha256 != up_config.model["xml_sha256"]:
        raise ValueError("Round-1 runtime XML identity mismatch")
    return config, artifact, env


def run_unified_round1(
    config_path: Path,
    run_id: str,
    *,
    run_root: Path | None = None,
    trainer: Callable[..., Any] = ppo_train.train,
    backend_name: Callable[[], str] = jax.default_backend,
):
    """Run one fresh 10M Round-1 job; reset distribution is the only change."""
    config, artifact, env = build_round1_environment(config_path)
    if backend_name() != "gpu":
        raise RuntimeError("Round-1 unified PPO requires the visible JAX GPU backend")
    root = (
        Path(run_root)
        if run_root is not None
        else Path(os.environ.get("JIT_RUN_ROOT", "JIT/runs/pi_unified"))
    )
    run_dir = root / run_id
    declaration = RunDeclaration(
        run_id=run_id,
        purpose=ROUND1_PURPOSE,
        output_dir=run_dir,
        config_sha256=config.config_sha256,
        xml_sha256=env._bundle.xml_sha256,
        reference_sha256=str(config.raw["inputs"].get("reference_sha256", "") or config.raw.get("reference_sha256", "") or env.resolved_config.model["reference_sha256"]),
        training_transition_ceiling=config.ppo.requested_transitions,
        stopping_conditions=(
            f"stop_at_exact_transition_{config.ppo.requested_transitions}",
            "stop_on_nonfinite_metric",
            "stop_on_cuda_or_oom_error",
            "stop_on_checkpoint_or_train_panel_persistence_failure",
        ),
        resume_command=(
            "PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python "
            "JIT/cli/train_unified_round1.py "
            f"--config {Path(config_path)} --run-id {run_id}"
        ),
        segment_seed=config.ppo.seed,
    )
    predeclare_run(declaration, resolved_config=config.raw)
    _write_json(
        run_dir / "round1_declaration.json",
        {
            "schema": "jit_pi_unified_round1_declaration_v1",
            "single_changed_method_variable": "reset_distribution",
            "round0_causal_classification": "PI_UP_APEX_UNIFIED_PRE_JUMP_FAIL",
            "reset_mixture": config.raw["reset_mixture"],
            "all_other_formal_hyperparameters_locked": True,
            "policy_count": 1,
            "expert_switching_used": False,
            "soft_tube_manifest_sha256": config.soft_tube_manifest_sha256,
            "test_data_used": False,
            "validation_data_used": False,
        },
    )
    _write_json(
        run_dir / "backend.json",
        {"jax_backend": backend_name(), "devices": [str(x) for x in jax.devices()]},
    )
    mark_run_running(
        run_dir,
        process_id=os.getpid(),
        metadata={
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "training_seed": config.ppo.seed,
            "target_training_transition": config.ppo.requested_transitions,
            "resume_semantics": "fresh",
            "round1_natural_reset_probability": ROUND1_NATURAL_PROBABILITY,
        },
    )

    identity = checkpoint_identity(config, env)

    def evaluate(step: int, make_policy: Any, params: Any) -> PanelResult:
        # Deliberately retain the exact Round-0 fixed TRAIN Tube panel so the
        # diagnostic itself is not another changed variable.
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
        if controller.completed_training_transitions != config.ppo.requested_transitions:
            raise ValueError("Round-1 trainer returned before the exact target")
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
        if not metrics or not all(math.isfinite(v) for v in metrics.values()):
            raise ValueError("Round-1 PPO produced no finite final metrics")
        report = {
            "schema": "jit_pi_unified_round1_formal_report_v1",
            "status": "completed",
            "requested_training_transitions": config.ppo.requested_transitions,
            "completed_training_transitions": controller.completed_training_transitions,
            "checkpoint_transitions": list(controller.checkpoint_transitions),
            "train_panel_transitions": list(controller.train_panel_transitions),
            "train_panel_interactions": controller.train_panel_interactions,
            "brax_evaluation_transitions": 0,
            "reset_mixture": config.raw["reset_mixture"],
            "single_changed_method_variable": "reset_distribution",
            "test_data_used": False,
            "validation_data_used": False,
            "expert_switching_used": False,
            "checkpoint_restored": True,
            "final_metrics": metrics,
        }
        _write_json(run_dir / "round1_report.json", report)
        close_run(
            run_dir,
            status="completed",
            accounting=InteractionAccounting(
                config.ppo.requested_transitions,
                0,
                0,
                controller.train_panel_interactions,
            ),
            reason="Round-1 10pct natural plus 90pct learned Soft-Tube RSI completed at the fixed 10M target; no checkpoint selection performed",
        )
        return {"run_dir": str(run_dir.resolve()), "round1_report": report}
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
