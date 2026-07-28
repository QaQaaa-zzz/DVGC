"""Shared frozen-policy rollout helpers used by certification and evaluation."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jp
import numpy as np

from .config import STAGE_ID
from .snapshot_timing import LEGACY_SNAPSHOT_SCHEMA, REPLAY_MODES, SNAPSHOT_SCHEMA_NAME


def inferred_apex_seen(record: dict[str, Any]) -> int:
    explicit = record.get("apex_seen")
    if explicit is not None:
        return int(explicit)
    index = record.get("reference_index")
    if index is None:
        index = record.get("source_index")
    return int(index is not None and int(index) >= 220)


def _logged_replay_sidecars(record: dict[str, Any], logged: bool):
    """Read logged tensors only for the explicitly logged restore mode."""
    if not logged:
        return None, None
    return (
        np.asarray(record["actor_observation_t"], np.float32),
        np.asarray(record["current_frame_t"], np.float32),
    )


def _legacy_restore(env, record: dict[str, Any], rng, *, use_logged_observation: bool):
    ps = dict(record.get("policy_state", {}))
    phase = int(record.get("oracle_phase", STAGE_ID[record["source_phase"]]))
    history = ps.get("obs_history")
    actor_observation = ps.get("actor_observation") if use_logged_observation else None
    if use_logged_observation and actor_observation is None:
        raise ValueError("legacy_logged_replay requires policy_state.actor_observation")
    return env.reset_from_snapshot(
        jp.asarray(record["qpos"]), jp.asarray(record["qvel"]), jp.asarray(record["ctrl"]), rng,
        jp.asarray(phase, jp.int32), jp.asarray(int(record.get("had_airborne", 0)), jp.int32),
        jp.asarray(int(record.get("had_valid_landing", 0)), jp.int32),
        jp.asarray(int(record.get("contact_age", 0)), jp.int32),
        jp.asarray(ps.get("last_action", np.zeros(env.action_size, np.float32))),
        estimated_phase=jp.asarray(int(ps.get("filter_phase", phase)), jp.int32),
        phase_probs=jp.asarray(ps.get("phase_probs", np.eye(4, dtype=np.float32)[phase])),
        airborne_count=jp.asarray(int(record.get("airborne_count", 0)), jp.int32),
        prelaunch_airborne_count=jp.asarray(int(record.get("prelaunch_airborne_count", 0)), jp.int32),
        landing_bounce_count=jp.asarray(int(record.get("landing_bounce_count", 0)), jp.int32),
        invalid_wheel_count=jp.asarray(int(record.get("invalid_wheel_count", 0)), jp.int32),
        recovery_count=jp.asarray(int(record.get("recovery_count", 0)), jp.int32),
        prev_acc_z=jp.asarray(float(ps.get("prev_acc_z", np.nan)), jp.float32),
        prev_vz=jp.asarray(float(ps.get("prev_vz", np.nan)), jp.float32),
        obs_history=None if history is None else jp.asarray(history),
        obs_history_valid=jp.asarray(history is not None),
        # Preserve the historical continuation contract even in the explicit
        # R0 diagnostic.  The independently rebuilt *current* packet may be a
        # hybrid, but subsequent history evolution remains legacy-identical.
        continuation_obs_history=None if history is None else jp.asarray(history),
        actor_observation=None if actor_observation is None else jp.asarray(actor_observation),
        actor_observation_valid=jp.asarray(actor_observation is not None),
        qacc_warmstart=jp.asarray(record["qacc_warmstart"]),
        stage_entry_ever=jp.asarray(int(record.get("stage_entry_ever", 0)), jp.int32),
        apex_seen=jp.asarray(inferred_apex_seen(record), jp.int32),
        jump_signal_latched=jp.asarray(bool(record.get("jump_signal_latched", record.get("had_airborne", 0)))),
        jump_window_start_x=jp.asarray(float(record.get("jump_window_start_x", record["qpos"][0])), jp.float32),
        jump_window_end_x=jp.asarray(float(record.get("jump_window_end_x", record["qpos"][0] + 1.0)), jp.float32),
    )


def _timing_explicit_restore(env, record: dict[str, Any], rng, *, logged: bool):
    if record.get("schema_name") != SNAPSHOT_SCHEMA_NAME or int(record.get("schema_version", -1)) != 4:
        raise ValueError("timing-explicit replay requires a v4 snapshot")
    physical = record["physical_state_t"]
    pre = np.asarray(record["obs_history_pre_t"], np.float32)
    post = np.asarray(record["obs_history_post_t"], np.float32)
    estimator_pre = record["estimator_state_pre_t"]
    estimator_post = record["estimator_state_post_t"]
    actor, logged_current_frame = _logged_replay_sidecars(record, logged)
    state = env.reset_from_snapshot(
        jp.asarray(physical["qpos"]), jp.asarray(physical["qvel"]), jp.asarray(physical["ctrl_previous"]), rng,
        jp.asarray(int(estimator_pre["phase"]), jp.int32),
        jp.asarray(int(estimator_pre["had_airborne"]), jp.int32),
        jp.asarray(int(estimator_pre["had_valid_landing"]), jp.int32),
        jp.asarray(int(estimator_pre["contact_age"]), jp.int32),
        jp.asarray(record["last_normalized_command_t"], jp.float32),
        estimated_phase=jp.asarray(int(estimator_pre["estimated_phase"]), jp.int32),
        phase_probs=jp.asarray(estimator_pre["phase_probs"], jp.float32),
        airborne_count=jp.asarray(int(estimator_pre["airborne_count"]), jp.int32),
        prelaunch_airborne_count=jp.asarray(int(estimator_pre["prelaunch_airborne_count"]), jp.int32),
        landing_bounce_count=jp.asarray(int(estimator_pre["landing_bounce_count"]), jp.int32),
        invalid_wheel_count=jp.asarray(int(estimator_pre["invalid_wheel_count"]), jp.int32),
        recovery_count=jp.asarray(int(estimator_pre["recovery_count"]), jp.int32),
        prev_acc_z=jp.asarray(float(estimator_pre["prev_acc_z"]), jp.float32),
        prev_vz=jp.asarray(float(estimator_pre["prev_vz"]), jp.float32),
        obs_history=jp.asarray(pre), obs_history_valid=jp.asarray(True),
        continuation_obs_history=jp.asarray(post),
        actor_packet_fifo=jp.asarray(record["actor_packet_fifo_t"]),
        actor_packet_fifo_valid=jp.asarray(3, jp.int32),
        actor_observation=None if actor is None else jp.asarray(actor),
        actor_observation_valid=jp.asarray(logged),
        observation_rng=jp.asarray(record["observation_rng_t"]),
        data_act=jp.asarray(physical["act"]),
        data_sensordata=jp.asarray(physical["sensordata"]),
        data_time=jp.asarray(physical["time"]),
        qacc_warmstart=jp.asarray(physical["qacc_warmstart"]),
        stage_entry_ever=jp.asarray(int(estimator_pre["stage_entry_ever"]), jp.int32),
        apex_seen=jp.asarray(int(estimator_pre["apex_seen"]), jp.int32),
        jump_signal_latched=jp.asarray(bool(estimator_pre["jump_signal_latched"])),
        jump_window_start_x=jp.asarray(float(estimator_pre["jump_window_start_x"]), jp.float32),
        jump_window_end_x=jp.asarray(float(estimator_pre["jump_window_end_x"]), jp.float32),
    )
    info = dict(state.info)
    reconstructed_current_frame = info["actor_current_frame"]
    for name, value in estimator_post.items():
        if name in info:
            info[name] = jp.asarray(value, dtype=info[name].dtype)
    info.update({
        "rng": jp.asarray(record["rng_state_t"]),
        "obs_history": jp.asarray(post),
        "actor_obs_history_pre": jp.asarray(pre),
        "actor_current_frame": (
            jp.asarray(logged_current_frame)
            if logged else reconstructed_current_frame
        ),
        "actor_obs_history_post": jp.asarray(post),
        "actor_packet_fifo": jp.asarray(record["actor_packet_fifo_t"]),
        "actor_packet_fifo_valid": jp.asarray(3, jp.int32),
        "actor_observation_rng": jp.asarray(record.get("observation_rng_t", record["rng_state_t"])),
        "actor_frame_prev_acc_z": jp.asarray(float(estimator_pre["prev_acc_z"]), jp.float32),
    })
    return state.replace(info=info)


def restore_snapshot_mode(env, record: dict[str, Any], rng, *, observation_mode: str):
    """Restore only through one of the three declared authority modes."""
    if observation_mode not in REPLAY_MODES:
        raise ValueError(f"observation_mode must be one of {REPLAY_MODES}")
    if observation_mode == "legacy_logged_replay":
        return _legacy_restore(env, record, rng, use_logged_observation=True)
    return _timing_explicit_restore(
        env, record, rng,
        logged=observation_mode == "timing_explicit_logged_replay",
    )


def restore_snapshot_logged(env, record: dict[str, Any], rng):
    """Deprecated alias for explicit legacy logged replay."""
    return restore_snapshot_mode(env, record, rng, observation_mode="legacy_logged_replay")


def restore_snapshot_reconstructed(env, record: dict[str, Any], rng):
    """Deprecated legacy hybrid diagnostic; never valid for authority."""
    return _legacy_restore(env, record, rng, use_logged_observation=False)


def restore_snapshot(env, record: dict[str, Any], rng):
    """Deprecated compatibility fallback; forbidden in new authority paths."""
    schema = record.get("schema_name", LEGACY_SNAPSHOT_SCHEMA)
    if schema == SNAPSHOT_SCHEMA_NAME:
        return restore_snapshot_mode(
            env, record, rng,
            observation_mode="timing_explicit_independent_reconstruction",
        )
    if record.get("policy_state", {}).get("actor_observation") is not None:
        return restore_snapshot_mode(env, record, rng, observation_mode="legacy_logged_replay")
    return _legacy_restore(env, record, rng, use_logged_observation=False)


def frozen_rollout(
    env, inference_fn, state, rng, *, horizon: int,
    action_noise_std: float = 0.0, step_fn: Callable | None = None,
):
    step_fn = jax.jit(env.step) if step_fn is None else step_fn
    trace = []
    for _ in range(int(horizon)):
        rng, action_key, noise_key = jax.random.split(rng, 3)
        action, _ = inference_fn(state.obs, action_key)
        if action_noise_std > 0:
            action = jp.clip(action + jax.random.normal(noise_key, action.shape) * float(action_noise_std), -1.0, 1.0)
        state = step_fn(state, action)
        trace.append(state)
        if float(np.asarray(jax.device_get(state.done))) > 0.5:
            break
    return state, {
        "chain": int(np.asarray(jax.device_get(state.info.get("chain_ever", state.info.get("chain_success", 0))))),
        "final": int(np.asarray(jax.device_get(state.info.get("recovery_success", 0)))),
        "terminated": int(np.asarray(jax.device_get(state.info.get("terminated", 0)))),
        "truncated": int(np.asarray(jax.device_get(state.info.get("truncated", 0)))),
        "end_code": int(np.asarray(jax.device_get(state.info.get("end_code", 0)))),
        "steps": len(trace),
    }
