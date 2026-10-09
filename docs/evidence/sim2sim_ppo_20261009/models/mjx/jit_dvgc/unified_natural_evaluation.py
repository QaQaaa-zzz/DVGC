"""Independent canonical natural-start evaluation for a frozen unified policy.

The current authoritative natural reset is intentionally deterministic. This
module therefore audits reset diversity before taking any environment step and
refuses to manufacture a statistical success rate from repeated copies of the
same physical initial state. Under the current contract it evaluates exactly
one canonical full-chain rollout. A later, separately declared reachable
initial-state bank is required for statistical Final-Recovery/JCE claims.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping, Sequence

import jax
import numpy as np

from .checkpoint import load_checkpoint
from .config import file_sha256
from .constants import END_REASONS
from .evaluation import EpisodeTrace, capture_episode, save_episode_trace
from .ppo import make_checkpoint_policy
from .unified_diagnostic import _load_runtime
from .unified_env import JUMP_START_X_M, UnifiedTubeRSIEnv
from .unified_formal import load_unified_formal_config
from .unified_training import canonical_sha256, checkpoint_identity


EVALUATION_SCHEMA = "jit_pi_unified_canonical_natural_eval_v1"
JUMP_START_EVALUATION_SCHEMA = "jit_pi_unified_canonical_jump_start_eval_v1"
RESET_AUDIT_SEED_START = 9_400_001
RESET_AUDIT_COUNT = 64
CANONICAL_ROLLOUT_SEED = 9_400_001
CANONICAL_POLICY_KEY = 9_410_001
EXPECTED_UNIQUE_NATURAL_RESETS = 1


class NaturalStartUnifiedEvalEnv(UnifiedTubeRSIEnv):
    """Unified runtime whose evaluation reset is the Phase-U natural reset.

    Training semantics remain untouched in :class:`UnifiedTubeRSIEnv`. This
    subclass exists only for independent evaluation and reuses the production
    unified natural-reset adapter so evaluation cannot drift from the step
    info/metric contract. The underlying physical reset remains the existing
    Phase-U natural reset.
    """

    def reset(self, rng: jax.Array):
        return self.reset_natural(rng)

    def reset_natural(self, rng: jax.Array):
        return self._reset_natural_unified(rng)


class JumpStartUnifiedEvalEnv(UnifiedTubeRSIEnv):
    """Unified runtime evaluated from the fixed ground jump-start reset."""

    def reset(self, rng: jax.Array):
        return self.reset_jump_start(rng)

    def reset_jump_start(self, rng: jax.Array):
        return self._reset_jump_start_unified(rng)


def jump_start_evaluation_contract() -> dict[str, Any]:
    """Declare the conditional start semantics without natural-start claims."""
    return {
        "schema": JUMP_START_EVALUATION_SCHEMA,
        "start_kind": "fixed_ground_jump_start",
        "jump_start_x_m": JUMP_START_X_M,
        "natural_start_connected": False,
        "tube_or_rsi_reset_used": False,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _repo_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _physical_reset_sha256(state: Any) -> str:
    digest = hashlib.sha256()
    for name in ("qpos", "qvel"):
        array = np.asarray(jax.device_get(getattr(state.data, name)))
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _source_training_provenance(config: Any) -> dict[str, Any]:
    declaration = config.raw.get("run_declaration")
    run_id = None
    if isinstance(declaration, Mapping):
        run_id = declaration.get("run_id")
    return {
        "source_training_run_id": run_id,
        "source_training_reset_mixture": config.reset_mixture.as_dict(),
    }


def audit_natural_reset_diversity(
    env: Any,
    seeds: Sequence[int],
    *,
    reset_fn: Callable[[jax.Array], Any] | None = None,
) -> dict[str, Any]:
    """Audit physical reset diversity without stepping the environment."""
    if not seeds:
        raise ValueError("natural reset audit requires at least one seed")
    reset = jax.jit(env.reset_natural) if reset_fn is None else reset_fn
    hashes: list[str] = []
    records: list[dict[str, Any]] = []
    for seed in seeds:
        state = reset(jax.random.PRNGKey(int(seed)))
        qpos = np.asarray(jax.device_get(state.data.qpos))
        qvel = np.asarray(jax.device_get(state.data.qvel))
        if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
            raise ValueError("natural reset audit produced a nonfinite physical state")
        if bool(np.asarray(jax.device_get(state.info.get("expert_switching_used", False)))):
            raise ValueError("natural reset audit unexpectedly used expert switching")
        soft_reset = float(
            np.asarray(jax.device_get(state.metrics.get("reset/source_soft_tube", 0.0)))
        )
        if soft_reset != 0.0:
            raise ValueError("natural reset audit unexpectedly used a Soft-Tube reset")
        state_hash = _physical_reset_sha256(state)
        hashes.append(state_hash)
        records.append(
            {
                "seed": int(seed),
                "physical_state_sha256": state_hash,
                "root_x": float(qpos[0]),
                "root_z": float(qpos[2]),
                "forward_velocity": float(qvel[0]),
                "vertical_velocity": float(qvel[2]),
            }
        )
    counts = Counter(hashes)
    return {
        "seed_count": len(seeds),
        "unique_physical_state_count": len(counts),
        "duplicate_seed_count": len(seeds) - len(counts),
        "physical_state_multiplicities": dict(sorted(counts.items())),
        "records": records,
        "environment_interactions": 0,
    }


def _ever(trace: EpisodeTrace, metric: str) -> bool:
    return any(frame.metrics.get(metric, 0.0) > 0.5 for frame in trace.frames)


def _first_metric_frame(trace: EpisodeTrace, metric: str) -> int | None:
    for index, frame in enumerate(trace.frames):
        if frame.metrics.get(metric, 0.0) > 0.5:
            return index
    return None


def _state_at(trace: EpisodeTrace, index: int | None) -> dict[str, float] | None:
    if index is None:
        return None
    frame = trace.frames[index]
    metrics = frame.metrics
    return {
        "frame_index": int(index),
        "root_x": float(metrics.get("signal/root_x", frame.qpos[0])),
        "root_z": float(metrics.get("signal/root_z", frame.qpos[2])),
        "forward_velocity": float(metrics.get("signal/forward_velocity", frame.qvel[0])),
        "vertical_velocity": float(metrics.get("signal/vertical_velocity", frame.qvel[2])),
        "roll": float(metrics.get("signal/roll", 0.0)),
        "pitch": float(metrics.get("signal/pitch", 0.0)),
        "roll_rate": float(metrics.get("signal/roll_rate", 0.0)),
        "pitch_rate": float(metrics.get("signal/pitch_rate", 0.0)),
    }


def summarize_canonical_natural_trace(trace: EpisodeTrace) -> dict[str, Any]:
    """Summarize the complete natural-start event chain for one frozen policy."""
    if not trace.frames:
        raise ValueError("canonical natural trace is empty")
    terminal = trace.frames[-1]
    jump_zone = _ever(trace, "event/jump_zone_seen")
    ascending = _ever(trace, "event/ascending_seen")
    height = _ever(trace, "event/height_seen")
    apex = _ever(trace, "event/apex_seen")
    phase_transition = _ever(trace, "event/tube_phase_transition")
    valid_contact = _ever(trace, "event/descent_valid_contact_seen")
    descent_success = _ever(trace, "terminal/descent_success")
    apex_index = _first_metric_frame(trace, "event/apex_seen")
    contact_index = _first_metric_frame(trace, "event/descent_valid_contact_seen")
    pre_landing_physical_failure = (
        True
        if contact_index is None
        else any(frame.physical_failure for frame in trace.frames[: contact_index + 1])
    )
    jump_trajectory_success = bool(
        jump_zone
        and ascending
        and height
        and apex
        and phase_transition
        and valid_contact
        and apex_index is not None
        and contact_index is not None
        and apex_index < contact_index
        and not pre_landing_physical_failure
    )
    full_recovery = bool(
        jump_trajectory_success
        and descent_success
        and terminal.success
    )
    end_reason = END_REASONS.get(int(terminal.end_code), f"unknown_{terminal.end_code}")

    root_z = [
        float(frame.metrics.get("signal/root_z", frame.qpos[2])) for frame in trace.frames
    ]
    root_x = [
        float(frame.metrics.get("signal/root_x", frame.qpos[0])) for frame in trace.frames
    ]
    abs_roll = [abs(float(frame.metrics.get("signal/roll", 0.0))) for frame in trace.frames]
    abs_pitch = [abs(float(frame.metrics.get("signal/pitch", 0.0))) for frame in trace.frames]
    angular_speed = [
        float(frame.metrics.get("signal/angular_speed", 0.0)) for frame in trace.frames
    ]
    actions = (
        np.stack([frame.action for frame in trace.frames[1:]])
        if len(trace.frames) > 1
        else np.zeros((0, 4))
    )
    action_abs = np.abs(actions)

    return {
        "seed": int(trace.seed),
        "environment_interactions": int(trace.environment_transitions),
        "captured_state_count": len(trace.frames),
        "jump_zone_seen": bool(jump_zone),
        "ascending_seen": bool(ascending),
        "height_seen": bool(height),
        "apex_seen": bool(apex),
        "phase_transitioned": bool(phase_transition),
        "valid_landing_contact_seen": bool(valid_contact),
        "jump_trajectory_success": jump_trajectory_success,
        "pre_landing_physical_failure": bool(pre_landing_physical_failure),
        "stable_recovery_success": bool(descent_success),
        "full_recovery_success": full_recovery,
        "terminal_success": bool(terminal.success),
        "terminal_physical_failure": bool(terminal.physical_failure),
        "terminal_timeout": bool(terminal.timeout),
        "terminal_end_code": int(terminal.end_code),
        "terminal_reason": end_reason,
        "start_x": root_x[0],
        "final_x": root_x[-1],
        "forward_displacement": root_x[-1] - root_x[0],
        "maximum_root_height": max(root_z),
        "maximum_abs_roll": max(abs_roll),
        "maximum_abs_pitch": max(abs_pitch),
        "maximum_angular_speed": max(angular_speed),
        "apex_state": _state_at(trace, apex_index),
        "first_valid_landing_state": _state_at(trace, contact_index),
        "action_statistics": {
            "mean_abs": action_abs.mean(axis=0).tolist() if action_abs.size else [0.0] * 4,
            "max_abs": action_abs.max(axis=0).tolist() if action_abs.size else [0.0] * 4,
            "saturation_fraction": float(np.mean(action_abs >= 0.999))
            if action_abs.size
            else 0.0,
        },
        "liftoff_metric_available": False,
    }


def _run_canonical_fixed_start_evaluation(
    config_path: Path,
    checkpoint: Path,
    output_dir: Path,
    *,
    contract: Mapping[str, Any],
    env_factory: Callable[..., Any],
    reset_method_name: str,
    backend_name: Callable[[], str],
    rollout_seed: int = CANONICAL_ROLLOUT_SEED,
) -> dict[str, Any]:
    """Run one fixed-start full-chain gate for a frozen unified policy."""
    config_path = Path(config_path)
    checkpoint_path = Path(checkpoint)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"evaluation output already exists: {output}")

    config = load_unified_formal_config(config_path)
    up_config, down_config, artifact, _ = _load_runtime(config)
    if backend_name() != "gpu":
        raise RuntimeError("canonical fixed-start evaluation requires the visible JAX GPU")
    env = env_factory(
        up_config,
        down_config,
        artifact,
        runtime_naccdmax=config.runtime_naccdmax,
    )
    payload = load_checkpoint(checkpoint_path, expected=checkpoint_identity(config, env))
    if payload.training_transitions != config.ppo.requested_transitions:
        raise ValueError("canonical evaluation requires the completed formal checkpoint")

    payload_sha = file_sha256(checkpoint_path / "payload.pkl")
    identity = json.loads((checkpoint_path / "identity.json").read_text(encoding="utf-8"))
    if identity.get("payload_sha256") != payload_sha:
        raise ValueError("canonical evaluation checkpoint payload identity drift")

    output.mkdir(parents=True, exist_ok=False)
    audit_seeds = tuple(
        range(RESET_AUDIT_SEED_START, RESET_AUDIT_SEED_START + RESET_AUDIT_COUNT)
    )
    source_training = _source_training_provenance(config)
    start_kind = str(contract["start_kind"])
    natural_start = bool(contract["natural_start_connected"])
    jump_start = start_kind == "fixed_ground_jump_start"
    reset_protocol = (
        "existing Phase-U natural reset converted to unified state without physical-state mutation"
        if natural_start
        else (
            "fixed ground jump-start reset at x=2.5 m; default keyframe pose and declared "
            "initial velocity; no Tube or RSI restoration"
        )
    )
    declaration = {
        "schema": str(contract["schema"]),
        "status": "predeclared",
        "purpose": (
            "canonical natural-start full-chain evaluation of the fixed completed unified policy"
            if natural_start
            else "canonical fixed jump-start full-chain evaluation of the frozen unified policy"
        ),
        "repository_head": _repo_head(),
        "formal_config": str(config_path.resolve()),
        "formal_config_sha256": config.config_sha256,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_payload_sha256": payload_sha,
        "checkpoint_training_transitions": int(payload.training_transitions),
        "xml_sha256": env._bundle.xml_sha256,
        "soft_tube_manifest_sha256": artifact.manifest["manifest_sha256"],
        **source_training,
        "start_contract": dict(contract),
        "reset_protocol": reset_protocol,
        "reset_audit_seed_start": RESET_AUDIT_SEED_START,
        "reset_audit_count": RESET_AUDIT_COUNT,
        "expected_unique_fixed_resets": EXPECTED_UNIQUE_NATURAL_RESETS,
        **(
            {"expected_unique_natural_resets": EXPECTED_UNIQUE_NATURAL_RESETS}
            if natural_start
            else {"expected_unique_jump_start_resets": EXPECTED_UNIQUE_NATURAL_RESETS}
        ),
        "canonical_rollout_seed": int(rollout_seed),
        "episode_horizon": int(config.ppo.episode_horizon),
        "jump_trajectory_definition": (
            "jump zone seen AND ascent/height/Apex seen AND unified phase transition "
            "AND first valid landing reached before any physical failure"
        ),
        "full_recovery_definition": "Apex seen AND unified phase transition AND valid descent contact AND descent stable-recovery terminal success",
        "selection_policy": "transition_10009600 fixed before evaluation; no checkpoint selection after TRAIN-panel inspection",
        "statistical_success_rate_claim": False,
        "training_transitions": 0,
        "expert_switching_used": False,
        "soft_tube_reset_used": False,
        "natural_reset_used": natural_start,
        "jump_start_reset_used": jump_start,
        "validation_data_used": False,
        "test_data_used": False,
    }
    declaration["protocol_sha256"] = canonical_sha256(declaration)
    _write_json(output / "declaration.json", declaration)

    try:
        reset_fn = jax.jit(getattr(env, reset_method_name))
        reset_audit = audit_natural_reset_diversity(
            env, audit_seeds, reset_fn=reset_fn
        )
        _write_json(output / "reset_diversity.json", reset_audit)
        if reset_audit["unique_physical_state_count"] != EXPECTED_UNIQUE_NATURAL_RESETS:
            raise ValueError(
                "fixed reset diversity contract changed; stop and predeclare a new evaluation protocol"
            )
        if jump_start and any(
            abs(float(record["root_x"]) - JUMP_START_X_M) > 1.0e-6
            for record in reset_audit["records"]
        ):
            raise ValueError("jump-start reset x drift")

        checkpoint_policy = make_checkpoint_policy(env, payload, deterministic=True)

        def deterministic_policy(observation):
            action, _ = checkpoint_policy(
                observation, jax.random.PRNGKey(CANONICAL_POLICY_KEY)
            )
            return action

        trace = capture_episode(
            env,
            deterministic_policy,
            seed=int(rollout_seed),
            horizon=config.ppo.episode_horizon,
            reset_fn=reset_fn,
            step_fn=jax.jit(env.step),
        )
        trace_artifact = save_episode_trace(trace, output / "canonical_trace")
        summary = summarize_canonical_natural_trace(trace)
        if not math.isfinite(float(summary["forward_displacement"])):
            raise ValueError("canonical evaluation produced nonfinite summary")

        report = {
            "schema": str(contract["schema"]),
            "status": "completed",
            "protocol_sha256": declaration["protocol_sha256"],
            "checkpoint_payload_sha256": payload_sha,
            **source_training,
            "reset_diversity": {
                "seed_count": reset_audit["seed_count"],
                "unique_physical_state_count": reset_audit[
                    "unique_physical_state_count"
                ],
                "duplicate_seed_count": reset_audit["duplicate_seed_count"],
            },
            "canonical_rollout": summary,
            "canonical_trace_npz": str(trace_artifact.npz_path.resolve()),
            "canonical_trace_metadata": str(trace_artifact.metadata_path.resolve()),
            "canonical_trace_npz_sha256": trace_artifact.npz_sha256,
            "environment_interactions": int(trace.environment_transitions),
            "training_transitions": 0,
            "expert_switching_used": False,
            "soft_tube_reset_used": False,
            "natural_reset_used": natural_start,
            "jump_start_reset_used": jump_start,
            "start_contract": dict(contract),
            "validation_data_used": False,
            "test_data_used": False,
            "statistical_success_rate_available": False,
            "statistical_success_rate_reason": (
                "the authoritative natural reset has one unique physical initial state across the 64-seed audit"
                if natural_start
                else "the fixed jump-start reset has one unique physical initial state across the 64-seed audit"
            ),
            "canonical_gate": (
                (
                    "CANONICAL_FULL_RECOVERY_GO"
                    if natural_start
                    else "CANONICAL_JUMP_START_TRAJECTORY_GO"
                )
                if (
                    summary["full_recovery_success"]
                    if natural_start
                    else summary["jump_trajectory_success"]
                )
                else (
                    "CANONICAL_FULL_RECOVERY_FAIL"
                    if natural_start
                    else "CANONICAL_JUMP_START_TRAJECTORY_FAIL"
                )
            ),
            "pi_unified_star_freeze_ready": False,
            "next_scientific_gate": (
                "build a separately predeclared real-dynamics reachable initial-state evaluation bank before any statistical Final-Recovery/JCE claim"
                if natural_start
                else (
                    "lock the real-frame jump-start centerline before conditional capability acquisition"
                    if summary["jump_trajectory_success"]
                    else "stop: frozen policy did not complete the fixed jump-start task"
                )
            ),
        }
        _write_json(output / "report.json", report)
        return report
    except BaseException as exc:
        _write_json(
            output / "failure.json",
            {
                **declaration,
                "status": "engineering_error",
                "error": f"{type(exc).__name__}: {exc}",
                "training_transitions": 0,
            },
        )
        raise


def run_canonical_natural_evaluation(
    config_path: Path,
    checkpoint: Path,
    output_dir: Path,
    *,
    env_factory: Callable[..., Any] = NaturalStartUnifiedEvalEnv,
    backend_name: Callable[[], str] = jax.default_backend,
) -> dict[str, Any]:
    """Run the canonical natural-start full-chain gate for a fixed policy."""
    return _run_canonical_fixed_start_evaluation(
        config_path,
        checkpoint,
        output_dir,
        contract={
            "schema": EVALUATION_SCHEMA,
            "start_kind": "canonical_natural_ground_start",
            "natural_start_connected": True,
            "tube_or_rsi_reset_used": False,
        },
        env_factory=env_factory,
        reset_method_name="reset_natural",
        backend_name=backend_name,
    )


def run_canonical_jump_start_evaluation(
    config_path: Path,
    checkpoint: Path,
    output_dir: Path,
    *,
    env_factory: Callable[..., Any] = JumpStartUnifiedEvalEnv,
    backend_name: Callable[[], str] = jax.default_backend,
    rollout_seed: int = CANONICAL_ROLLOUT_SEED,
) -> dict[str, Any]:
    """Run the canonical fixed-ground jump-start full-chain gate."""
    return _run_canonical_fixed_start_evaluation(
        config_path,
        checkpoint,
        output_dir,
        contract=jump_start_evaluation_contract(),
        env_factory=env_factory,
        reset_method_name="reset_jump_start",
        backend_name=backend_name,
        rollout_seed=rollout_seed,
    )


# Compatibility alias for older callers; there is one implementation above.
run_round0_canonical_natural_evaluation = run_canonical_natural_evaluation
