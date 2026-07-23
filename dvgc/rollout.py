"""Shared frozen-policy rollout helpers used by certification and evaluation."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jp
import numpy as np

from .config import STAGE_ID


def inferred_apex_seen(record: dict[str, Any]) -> int:
    """Restore explicit latch or infer it without evaluating ``int(None)``."""
    explicit = record.get("apex_seen")
    if explicit is not None:
        return int(explicit)
    index = record.get("reference_index")
    if index is None:
        index = record.get("source_index")
    return int(index is not None and int(index) >= 220)


def restore_snapshot(env, record: dict[str, Any], rng):
    ps = dict(record.get("policy_state", {}))
    phase = int(record.get("oracle_phase", STAGE_ID[record["source_phase"]]))
    history = ps.get("obs_history")
    actor_observation = ps.get("actor_observation")
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
        obs_history=(None if history is None else jp.asarray(history)),
        obs_history_valid=jp.asarray(history is not None),
        actor_observation=(None if actor_observation is None else jp.asarray(actor_observation)),
        actor_observation_valid=jp.asarray(actor_observation is not None),
        qacc_warmstart=jp.asarray(record["qacc_warmstart"]),
        stage_entry_ever=jp.asarray(int(record.get("stage_entry_ever",0)),jp.int32),
        apex_seen=jp.asarray(inferred_apex_seen(record), jp.int32),
        jump_signal_latched=jp.asarray(bool(record.get("jump_signal_latched",record.get("had_airborne",0)))),
        jump_window_start_x=jp.asarray(float(record.get("jump_window_start_x",record["qpos"][0])),jp.float32),
        jump_window_end_x=jp.asarray(float(record.get("jump_window_end_x",record["qpos"][0]+1.0)),jp.float32),
    )


def frozen_rollout(
    env,
    inference_fn,
    state,
    rng,
    *,
    horizon: int,
    action_noise_std: float = 0.0,
    step_fn: Callable | None = None,
):
    """Roll out a frozen policy, optionally reusing a caller-owned JIT.

    Sequential evaluation and certification call this helper many times.  The
    caller should compile one step function per dynamics environment and pass
    it here so those runs do not accumulate equivalent MJX/Warp executables.
    """
    step_fn = jax.jit(env.step) if step_fn is None else step_fn
    trace = []
    for step in range(int(horizon)):
        rng, action_key, noise_key = jax.random.split(rng, 3)
        action, _ = inference_fn(state.obs, action_key)
        if action_noise_std > 0:
            action = jp.clip(action + jax.random.normal(noise_key, action.shape) * float(action_noise_std), -1.0, 1.0)
        state = step_fn(state, action)
        trace.append(state)
        if float(np.asarray(jax.device_get(state.done))) > 0.5:
            break
    chain = int(np.asarray(jax.device_get(state.info.get("chain_ever", state.info.get("chain_success", 0)))))
    final = int(np.asarray(jax.device_get(state.info.get("recovery_success", 0))))
    terminated = int(np.asarray(jax.device_get(state.info.get("terminated", 0))))
    truncated = int(np.asarray(jax.device_get(state.info.get("truncated", 0))))
    end_code = int(np.asarray(jax.device_get(state.info.get("end_code", 0))))
    return state, {
        "chain": chain,
        "final": final,
        "terminated": terminated,
        "truncated": truncated,
        "end_code": end_code,
        "steps": len(trace),
    }
