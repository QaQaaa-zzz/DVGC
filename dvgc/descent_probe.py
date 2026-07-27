"""Batched physical controllability probes for the bounded Descent bank."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from dvgc.descent_pilot import REWARD_KEYS
from dvgc.rollout import restore_snapshot
from dvgc.runtime import build_inference


def formal_dynamic_margin(feature: jax.Array, cfg: Any) -> jax.Array:
    """Signed margin to continuous formal hard-failure boundaries.

    Angular velocity is a Recovery-quality gate, not a hard-failure terminal
    in ``OrangeBikeDVGC.step``; it must therefore not steer the CEM's formal
    failure-margin objective.
    """
    roll = jnp.deg2rad(float(cfg.max_roll_deg)) - jnp.abs(feature[..., 3])
    pitch = jnp.deg2rad(float(cfg.max_pitch_deg)) - jnp.abs(feature[..., 4])
    return jnp.minimum(roll, pitch)


def make_residual_rollout(
    env: Any, params: Any, *, horizon: int = 24,
    ticks_per_knot: int = 4, residual_ticks: int | None = None,
):
    """Return a JIT batched closed-loop residual rollout callable."""
    inference = build_inference(env, params, deterministic=True)
    step = jax.vmap(env.step)
    feature_fn = jax.vmap(env._physical_feature)
    cfg = env._config

    def rollout(state: Any, residual_knots: jax.Array, key: jax.Array):
        count = residual_knots.shape[0]
        active = jnp.ones((count,), bool)
        survival = jnp.zeros((count,), jnp.int32)
        minimum_margin = jnp.full((count,), jnp.inf, jnp.float32)
        terminal_margin = jnp.full((count,), jnp.inf, jnp.float32)
        reward_sum = jnp.zeros((count,), jnp.float32)
        component_sum = jnp.zeros((count, len(REWARD_KEYS)), jnp.float32)
        first_action_tick = jnp.full((count,), -1, jnp.int32)
        actions = jnp.zeros((horizon, count, env.action_size), jnp.float32)
        features = jnp.zeros((horizon, count, 16), jnp.float32)
        end_codes = jnp.zeros((count,), jnp.int32)

        def body(carry, tick):
            (state, active, survival, minimum_margin, terminal_margin,
             reward_sum, component_sum, first_action_tick, actions, features, end_codes) = carry
            action, _ = inference(state.obs, jax.random.fold_in(key, tick))
            residual = residual_knots[:, jnp.minimum(tick // ticks_per_knot, residual_knots.shape[1] - 1)]
            if residual_ticks is not None:
                residual = jnp.where(tick < residual_ticks, residual, jnp.zeros_like(residual))
            commanded = jnp.clip(action + residual, -1.0, 1.0)
            next_state = step(state, commanded)
            feature = feature_fn(next_state.data)
            margin = formal_dynamic_margin(feature, cfg)
            alive = active & (~next_state.done.astype(bool))
            survival = survival + alive.astype(jnp.int32)
            minimum_margin = jnp.where(active, jnp.minimum(minimum_margin, margin), minimum_margin)
            terminal_margin = jnp.where(active, margin, terminal_margin)
            reward_sum = reward_sum + jnp.where(active, next_state.reward, 0.0)
            terms = jnp.stack([next_state.metrics[name] for name in REWARD_KEYS], axis=-1)
            component_sum = component_sum + jnp.where(active[:, None], terms, 0.0)
            meaningful = jnp.max(jnp.abs(residual), axis=-1) > 1e-8
            first_action_tick = jnp.where((first_action_tick < 0) & meaningful, tick, first_action_tick)
            actions = actions.at[tick].set(commanded)
            features = features.at[tick].set(feature)
            end_codes = jnp.where(active & next_state.done.astype(bool), next_state.info["end_code"], end_codes)
            # MJX-Warp stores some contact metadata flattened across the batch,
            # so generic per-row tree masking is invalid.  Terminal rows may
            # continue numerically, but `active` excludes every later value
            # from objectives, rewards, end codes and survival accounting.
            return ((next_state, alive, survival, minimum_margin, terminal_margin,
                     reward_sum, component_sum, first_action_tick, actions, features, end_codes), None)

        initial = (state, active, survival, minimum_margin, terminal_margin,
                   reward_sum, component_sum, first_action_tick, actions, features, end_codes)
        final, _ = jax.lax.scan(body, initial, jnp.arange(horizon))
        (_, _, survival, minimum_margin, terminal_margin, reward_sum,
         component_sum, first_action_tick, actions, features, end_codes) = final
        effort = jnp.sqrt(jnp.mean(residual_knots * residual_knots, axis=(1, 2)))
        return {"survival": survival, "minimum_margin": minimum_margin,
                "terminal_margin": terminal_margin, "reward_return": reward_sum,
                "reward_components": component_sum, "residual_rms": effort,
                "residual_max": jnp.max(jnp.abs(residual_knots), axis=(1, 2)),
                "first_action_tick": first_action_tick, "actions": actions,
                "features": features, "end_code": end_codes}
    return jax.jit(rollout)


def base_state(env: Any, record: Mapping[str, Any], seed: int) -> Any:
    return restore_snapshot(env, record, jax.random.PRNGKey(seed))


def batched_base_state(env: Any, record: Mapping[str, Any], seed: int, count: int) -> Any:
    """Restore via vmap so MJX-Warp internal metadata gets legal batch axes."""
    keys = jax.random.split(jax.random.PRNGKey(seed), int(count))
    return jax.jit(jax.vmap(lambda key: restore_snapshot(env, record, key)))(keys)


def lexicographic_order(result: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.lexsort((result["residual_rms"], -result["terminal_margin"],
                       -result["minimum_margin"], -result["survival"]))


def exact_replay_matches(
    first: Mapping[str, np.ndarray], second: Mapping[str, np.ndarray],
    selected_summary: Mapping[str, Any],
) -> bool:
    """Requires repeatability *and* identity with the selected CEM result."""
    repeated = all(np.array_equal(np.asarray(first[key]), np.asarray(second[key]))
                   for key in ("survival", "minimum_margin", "terminal_margin", "end_code", "actions", "features"))
    if not repeated:
        return False
    scalar_matches = all(np.array_equal(np.asarray(first[key])[0], np.asarray(selected_summary[key]))
                         for key in ("survival", "minimum_margin", "terminal_margin", "end_code"))
    action_matches = np.array_equal(np.asarray(first["actions"])[:, 0], np.asarray(selected_summary["actions"]))
    feature_matches = np.array_equal(np.asarray(first["features"])[:, 0], np.asarray(selected_summary["features"]))
    return bool(scalar_matches and action_matches and feature_matches)


def cem_search(
    rollout: Any, state_factory: Any, *, bound: float, seed: int,
    generations: int = 5, samples: int = 256, elite_count: int = 32,
    knot_count: int = 6,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    """Run one fixed-budget 24-D residual CEM level."""
    rng = np.random.default_rng(seed); mean = np.zeros((knot_count, 4), np.float32)
    std = np.full((knot_count, 4), bound * .5, np.float32); all_rows = []; best = None
    for generation in range(generations):
        knots = np.clip(rng.normal(mean, std, size=(samples, knot_count, 4)), -bound, bound).astype(np.float32)
        state = state_factory(samples)
        result = jax.device_get(rollout(state, jnp.asarray(knots), jax.random.PRNGKey(seed + generation)))
        order = lexicographic_order(result); elite = knots[order[:elite_count]]
        mean = elite.mean(axis=0); std = np.maximum(elite.std(axis=0), bound * .02)
        for index in range(samples):
            all_rows.append({"generation": generation, "sample": index,
                "survival": int(result["survival"][index]),
                "minimum_margin": float(result["minimum_margin"][index]),
                "terminal_margin": float(result["terminal_margin"][index]),
                "reward_return": float(result["reward_return"][index]),
                "reward_components": np.asarray(result["reward_components"][index]).tolist(),
                "residual_rms": float(result["residual_rms"][index]),
                "residual_max": float(result["residual_max"][index]),
                "end_code": int(result["end_code"][index])})
        candidate = int(order[0])
        payload = {
            key: np.asarray(value[:, candidate] if key in {"actions", "features"} else value[candidate])
            for key, value in result.items()
        }
        if best is None or tuple((-payload["survival"], -payload["minimum_margin"],
                                  -payload["terminal_margin"], payload["residual_rms"])) < best[0]:
            best = ((-payload["survival"], -payload["minimum_margin"],
                     -payload["terminal_margin"], payload["residual_rms"]), knots[candidate].copy(), payload)
    assert best is not None
    summary = {key: (value.tolist() if np.asarray(value).ndim else float(value))
               for key, value in best[2].items() if key not in {"actions", "features"}}
    summary["actions"] = best[2]["actions"].tolist();summary["features"] = best[2]["features"].tolist()
    return best[1], summary, all_rows
