"""Frozen-policy complete-packet delay sensitivity rollouts."""
from __future__ import annotations

from typing import Any, Sequence

import jax
import jax.numpy as jnp

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

        def body(carry, tick):
            state, queue, active, survival, minimum_margin, terminal_margin, end_codes, actions = carry
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
            queue = jnp.concatenate((queue[:, 1:], next_state.obs["state"][:, None]), axis=1)
            return (next_state, queue, alive, survival, minimum_margin, terminal_margin, end_codes, actions), None

        initial = (state, packet_queue, active, survival, minimum_margin, terminal_margin, end_codes, actions)
        final, _ = jax.lax.scan(body, initial, jnp.arange(horizon))
        _, _, _, survival, minimum_margin, terminal_margin, end_codes, actions = final
        return {
            "survival": survival,
            "minimum_margin": minimum_margin,
            "terminal_margin": terminal_margin,
            "end_code": end_codes,
            "actions": actions,
        }

    return jax.jit(rollout)
