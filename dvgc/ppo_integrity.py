"""Small, auditable PPO loop for update-integrity experiments.

This module intentionally covers only the bounded single-device Descent pilot.
It keeps one observation-normalizer snapshot for rollout and loss, and exposes
the optimizer/RNG/environment state needed for deterministic continuation.
"""
from __future__ import annotations

import hashlib
import pickle
from functools import partial
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from dvgc.runtime import _atomic_pickle_dump, build_network_factory
from dvgc.wrappers import wrap_for_training


def tree_hash(tree: Any) -> str:
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(jax.device_get(tree)):
        value = np.ascontiguousarray(np.asarray(leaf))
        digest.update(str(value.dtype).encode())
        digest.update(repr(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def tree_norm(tree: Any) -> float:
    return float(np.sqrt(sum(float(np.sum(np.asarray(x, np.float64) ** 2))
                             for x in jax.tree_util.tree_leaves(jax.device_get(tree)))))


def tree_delta(after: Any, before: Any) -> dict[str, float]:
    delta = jax.tree_util.tree_map(lambda x, y: x - y, after, before)
    absolute, base = tree_norm(delta), tree_norm(before)
    return {"l2": absolute, "relative_l2": absolute / max(base, 1e-12)}


def normalizer_summary(state: Any) -> dict[str, Any]:
    def stats(tree):
        values = np.concatenate([np.asarray(x, np.float64).reshape(-1)
                                 for x in jax.tree_util.tree_leaves(jax.device_get(tree))])
        return {"min": float(values.min()), "mean": float(values.mean()),
                "max": float(values.max()), "l2": float(np.linalg.norm(values))}
    count = state.count
    if hasattr(count, "hi"):
        count_value = int(np.asarray(count.hi)) * 2**32 + int(np.asarray(count.lo))
    else:
        count_value = int(np.asarray(count))
    return {"sha256": tree_hash(state), "count": count_value,
            "mean": stats(state.mean), "std": stats(state.std)}


def _network(obs_shape: Any, action_size: int):
    from brax.training.acme import running_statistics
    return build_network_factory()(obs_shape, action_size,
                                   preprocess_observations_fn=running_statistics.normalize)


def make_optimizer(learning_rate: float, max_grad_norm: float = .75):
    """Build the exact bounded-pilot Adam optimizer."""
    import optax
    return optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate))


def prepare(env: Any, initial_params: Any, *, seed: int = 0, num_envs: int = 50,
            episode_length: int = 24, learning_rate: float = 1e-4) -> dict[str, Any]:
    """Create the exact initial single-device learner state and first rollout key."""
    from brax.training.agents.ppo import losses as ppo_losses

    key = jax.random.PRNGKey(seed)
    global_key, local_key = jax.random.split(key)
    del global_key
    local_key = jax.random.fold_in(local_key, jax.process_index())
    local_key, key_env, _ = jax.random.split(local_key, 3)
    wrapped = wrap_for_training(env, episode_length=episode_length, action_repeat=1)
    key_envs = jax.random.split(key_env, num_envs)
    env_state = jax.jit(wrapped.reset)(key_envs)
    obs_shape = jax.tree_util.tree_map(lambda x: x.shape[1:], env_state.obs)
    network = _network(obs_shape, env.action_size)
    learner_params = ppo_losses.PPONetworkParams(policy=initial_params[1], value=initial_params[2])
    optimizer = make_optimizer(learning_rate)
    return {"env": wrapped, "env_state": env_state, "key": local_key,
            "key_envs": key_envs, "network": network, "params": learner_params,
            "normalizer": initial_params[0], "optimizer": optimizer,
            "optimizer_state": optimizer.init(learner_params), "env_steps": 0}


def collect_first_rollout(state: Mapping[str, Any], *, unroll_length: int = 32):
    """Collect 50x32 unique transitions using the rollout normalizer snapshot."""
    from brax.training import acting
    from brax.training.agents.ppo import networks as ppo_networks
    key_epoch, next_local_key = jax.random.split(state["key"])
    key_sgd, key_unroll, next_epoch_key = jax.random.split(key_epoch, 3)
    inference_params = (state["normalizer"], state["params"].policy, state["params"].value)
    policy = ppo_networks.make_inference_fn(state["network"])(inference_params)
    final_state, data = acting.generate_unroll(
        state["env"], state["env_state"], policy, key_unroll, unroll_length,
        extra_fields=("truncation", "episode_metrics", "episode_done", "reset_parent"),
    )
    # Brax loss convention: [sequence/environment, time, ...].
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
    return data, final_state, key_sgd, next_epoch_key, next_local_key


def update_normalizer(normalizer: Any, observations: Any) -> Any:
    from brax.training.acme import running_statistics
    return running_statistics.update(normalizer, observations, pmap_axis_name=None)


def logprob_audit(network: Any, params: Any, normalizer: Any, data: Any) -> dict[str, Any]:
    logits = network.policy_network.apply(normalizer, params.policy, data.observation)
    dist = network.parametric_action_distribution
    raw = data.extras["policy_extras"]["raw_action"]
    stored = data.extras["policy_extras"]["log_prob"]
    recomputed = dist.log_prob(logits, raw)
    old_logits = data.extras["policy_extras"]["distribution_params"]
    ratio = jnp.exp(recomputed - stored)
    analytic = jnp.mean(dist.create_dist(logits).kl_divergence(dist.create_dist(old_logits)))
    error = np.asarray(recomputed - stored)
    ratio_np = np.asarray(ratio)
    return {
        "stored_recomputed_max_abs_error": float(np.max(np.abs(error))),
        "stored_recomputed_mean_error": float(np.mean(error)),
        "sample_mean_kl": float(np.mean(np.asarray(stored - recomputed))),
        "analytic_distribution_kl_mean": float(analytic),
        "ratio": {"min": float(ratio_np.min()), "p05": float(np.quantile(ratio_np, .05)),
                  "median": float(np.median(ratio_np)), "p95": float(np.quantile(ratio_np, .95)),
                  "max": float(ratio_np.max())},
        "distribution_params_max_abs_error": float(np.max(np.abs(np.asarray(logits-old_logits)))),
        "log_prob_semantics": "NormalTanhDistribution.log_prob(raw_pre_squash); Jacobian applied once by distribution",
        "kl_reduction": "analytic distribution KL summed over action event dimensions, mean over samples/time",
    }


def _global_norm(tree: Any) -> float:
    return tree_norm(tree)


def optimize_batch(state: Mapping[str, Any], data: Any, normalizer: Any, key: Any,
                   *, passes: int = 2, num_minibatches: int = 2,
                   rollback_kl: float | None = None) -> tuple[Any, Any, dict[str, Any]]:
    """Apply the authorized PPO update while keeping `normalizer` immutable."""
    from brax.training.agents.ppo import losses as ppo_losses
    import optax

    loss_fn = partial(ppo_losses.compute_ppo_loss, ppo_network=state["network"],
                      entropy_cost=.001, discounting=.995, reward_scaling=.1,
                      gae_lambda=.97, clipping_epsilon=.10)
    value_grad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
    params, opt_state = state["params"], state["optimizer_state"]
    records = []
    for pass_index in range(passes):
        key, perm_key = jax.random.split(key)
        order = np.asarray(jax.random.permutation(perm_key, data.reward.shape[0]))
        for minibatch_index, indices in enumerate(np.array_split(order, num_minibatches)):
            minibatch = jax.tree_util.tree_map(lambda x: x[indices], data)
            key, loss_key = jax.random.split(key)
            (loss, metrics), grads = value_grad(params, normalizer, minibatch, loss_key)
            raw_norm = _global_norm(grads)
            actor_norm = _global_norm(grads.policy)
            critic_norm = _global_norm(grads.value)
            log_std = [x for path, x in jax.tree_util.tree_flatten_with_path(grads.policy)[0]
                       if "scale_parameter" in "/".join(str(p) for p in path)]
            log_std_norm = _global_norm(log_std) if log_std else 0.0
            before, before_opt = params, opt_state
            updates, candidate_opt = state["optimizer"].update(grads, opt_state, params)
            candidate_params = optax.apply_updates(params, updates)
            post_kl = logprob_audit(state["network"], candidate_params, normalizer, data)[
                "analytic_distribution_kl_mean"]
            rolled_back = rollback_kl is not None and (not np.isfinite(post_kl) or post_kl > rollback_kl)
            if rolled_back:
                params, opt_state = before, before_opt
            else:
                params, opt_state = candidate_params, candidate_opt
            records.append({"pass": pass_index, "minibatch": minibatch_index,
                            "loss": float(loss), "gradient_norm_before_clip": raw_norm,
                            "gradient_norm_after_clip_upper_bound": min(raw_norm, .75),
                            "actor_gradient_norm": actor_norm, "critic_gradient_norm": critic_norm,
                            "log_std_gradient_norm": log_std_norm,
                            "candidate_post_update_analytic_kl": post_kl,
                            "rolled_back": rolled_back,
                            "rollback_threshold": rollback_kl,
                            "parameter_delta": tree_delta(params, before),
                            **{k: float(v) for k, v in metrics.items()}})
    return params, opt_state, {"gradient_steps": len(records),
                               "accepted_gradient_steps": sum(not row["rolled_back"] for row in records),
                               "rolled_back_gradient_steps": sum(row["rolled_back"] for row in records),
                               "steps": records,
                               "final": records[-1]}


def save_training_state(path: str | Path, state: Mapping[str, Any]) -> None:
    payload = {key: value for key, value in state.items()
               if key not in {"env", "network", "optimizer"}}
    payload["schema"] = "dvgc_integrity_ppo_state_v1"
    _atomic_pickle_dump(Path(path), jax.device_get(payload))


def load_training_state(path: str | Path, runtime_state: Mapping[str, Any]) -> dict[str, Any]:
    with Path(path).open("rb") as stream:
        payload = pickle.load(stream)
    if payload.pop("schema", None) != "dvgc_integrity_ppo_state_v1":
        raise ValueError("invalid integrity PPO checkpoint")
    result = dict(runtime_state)
    result.update(payload)
    return result
