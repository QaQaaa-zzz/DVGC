"""Frozen-policy complete-packet delay sensitivity rollouts."""
from __future__ import annotations

from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from dvgc.descent_probe import formal_dynamic_margin
from dvgc.runtime import build_inference


def make_packet_delay_rollout(
    env: Any,
    params: Any,
    delay_schedule: Sequence[int],
    *,
    horizon: int = 24,
    ticks_per_knot: int = 4,
    residual_ticks: int = 8,
):
    """Return a batched rollout using whole causal observation packets.

    ``packet_queue`` is ordered ``[t-2, t-1, t]``.  Each policy evaluation
    consumes exactly one complete packet; individual frames or signals are
    never independently shifted.
    """
    schedule = jnp.asarray(tuple(int(value) for value in delay_schedule), jnp.int32)
    if schedule.shape != (int(horizon),) or bool(jnp.any((schedule < 0) | (schedule > 2))):
        raise ValueError("delay schedule must contain horizon values in {0,1,2}")
    inference = build_inference(env, params, deterministic=True)
    step = jax.vmap(env.step)
    feature_fn = jax.vmap(env._physical_feature)
    cfg = env._config

    def rollout(state, residual_knots, packet_queue, key):
        count = residual_knots.shape[0]
        active = jnp.ones((count,), bool)
        survival = jnp.zeros((count,), jnp.int32)
        minimum_margin = jnp.full((count,), jnp.inf, jnp.float32)
        terminal_margin = jnp.full((count,), jnp.inf, jnp.float32)
        end_codes = jnp.zeros((count,), jnp.int32)
        actions = jnp.zeros((horizon, count, env.action_size), jnp.float32)
        active_action_mask = jnp.zeros((horizon, count), bool)
        termination_tick = jnp.full((count,), horizon, jnp.int32)
        phase_trace = jnp.zeros((horizon, count), jnp.int32)
        contact_age_trace = jnp.zeros((horizon, count), jnp.int32)
        landing_entry = jnp.zeros((count,), bool)
        chain = jnp.zeros((count,), bool)
        recovery = jnp.zeros((count,), bool)

        def body(carry, tick):
            (state, queue, active, survival, minimum_margin, terminal_margin,
             end_codes, actions, active_action_mask, termination_tick,
             phase_trace, contact_age_trace, landing_entry, chain, recovery) = carry
            delay = schedule[tick]
            actor_packet = jax.lax.dynamic_index_in_dim(queue, 2 - delay, axis=1, keepdims=False)
            policy_obs = dict(state.obs)
            policy_obs["state"] = actor_packet
            action, _ = inference(policy_obs, jax.random.fold_in(key, tick))
            residual = residual_knots[:, jnp.minimum(tick // ticks_per_knot, residual_knots.shape[1] - 1)]
            residual = jnp.where(tick < residual_ticks, residual, jnp.zeros_like(residual))
            commanded = jnp.clip(action + residual, -1.0, 1.0)
            next_state = step(state, commanded)
            feature = feature_fn(next_state.data)
            margin = formal_dynamic_margin(feature, cfg)
            alive = active & (~next_state.done.astype(bool))
            survival = survival + alive.astype(jnp.int32)
            minimum_margin = jnp.where(active, jnp.minimum(minimum_margin, margin), minimum_margin)
            terminal_margin = jnp.where(active, margin, terminal_margin)
            end_codes = jnp.where(active & next_state.done.astype(bool), next_state.info["end_code"], end_codes)
            actions = actions.at[tick].set(commanded)
            active_action_mask = active_action_mask.at[tick].set(active)
            termination_tick = jnp.where(active & next_state.done.astype(bool), tick + 1, termination_tick)
            phase_trace = phase_trace.at[tick].set(next_state.info["phase"])
            contact_age_trace = contact_age_trace.at[tick].set(next_state.info["contact_age"])
            landing_entry = landing_entry | (active & (next_state.metrics["event/landing"] > 0.5))
            chain = chain | (active & (next_state.info["chain_ever"] > 0))
            recovery = recovery | (active & (next_state.info["recovery_success"] > 0))
            queue = jnp.concatenate((queue[:, 1:], next_state.obs["state"][:, None]), axis=1)
            return ((next_state, queue, alive, survival, minimum_margin,
                     terminal_margin, end_codes, actions, active_action_mask,
                     termination_tick, phase_trace, contact_age_trace,
                     landing_entry, chain, recovery), None)

        initial = (state, packet_queue, active, survival, minimum_margin,
                   terminal_margin, end_codes, actions, active_action_mask,
                   termination_tick, phase_trace, contact_age_trace,
                   landing_entry, chain, recovery)
        final, _ = jax.lax.scan(body, initial, jnp.arange(horizon))
        (_, _, _, survival, minimum_margin, terminal_margin, end_codes,
         actions, active_action_mask, termination_tick, phase_trace,
         contact_age_trace, landing_entry, chain, recovery) = final
        return {
            "survival": survival,
            "minimum_margin": minimum_margin,
            "terminal_margin": terminal_margin,
            "end_code": end_codes,
            "actions": actions,
            "active_action_mask": active_action_mask,
            "termination_tick": termination_tick,
            "landing_entry": landing_entry,
            "chain": chain,
            "recovery_success": recovery,
            "final_recovery": recovery,
            "phase_trace": phase_trace,
            "contact_age_trace": contact_age_trace,
        }

    return jax.jit(rollout)


def active_prefix_repeat_comparison(first: dict[str, Any], second: dict[str, Any], index: int = 0) -> dict[str, Any]:
    """Compare every authoritative field only through the terminal transition."""
    scalar_fields = ("survival", "minimum_margin", "terminal_margin", "end_code", "termination_tick", "landing_entry", "chain", "recovery_success", "final_recovery")
    scalar = {name: bool(np.array_equal(np.asarray(first[name])[index], np.asarray(second[name])[index])) for name in scalar_fields}
    tick = int(np.asarray(first["termination_tick"])[index])
    tick2 = int(np.asarray(second["termination_tick"])[index])
    prefix = min(tick, tick2)
    trace_fields = ("actions", "active_action_mask", "phase_trace", "contact_age_trace")
    trace = {
        name: bool(np.array_equal(
            np.asarray(first[name])[:prefix, index],
            np.asarray(second[name])[:prefix, index],
        ))
        for name in trace_fields
    }
    failed = [name for name, value in {**scalar, **trace}.items() if not value]
    return {"exact": not failed, "active_prefix_ticks": prefix, "scalar_fields": scalar, "trace_fields": trace, "failed_fields": failed}
