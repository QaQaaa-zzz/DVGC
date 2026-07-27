"""Bounded supervised utilities for the eight-teacher Descent probe."""
from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dvgc.config import file_sha256
from dvgc.descent_probe import formal_dynamic_margin
from dvgc.env import END_REASON
from dvgc.rollout import restore_snapshot
from dvgc.runtime import make_dvgc_ppo_networks


def build_actor_tools(env: Any, frozen_params: Any):
    try:
        from brax.training.acme import running_statistics
    except ImportError:
        from brax.training import running_statistics
    sample = env.reset(jax.random.PRNGKey(0)).obs
    networks = make_dvgc_ppo_networks(
        {key: tuple(value.shape) for key, value in sample.items()}, env.action_size,
        running_statistics.normalize,
    )
    distribution = networks.parametric_action_distribution
    normalizer = frozen_params[0]

    def action(policy, observation):
        logits = networks.policy_network.apply(normalizer, policy, {"state": observation})
        return distribution.mode(logits)

    def loc_scale(policy, observation):
        logits = networks.policy_network.apply(normalizer, policy, {"state": observation})
        dist = distribution.create_dist(logits)
        return dist.loc, dist.scale

    return networks, jax.jit(action), jax.jit(loc_scale)


def replace_trainable(base_policy: Mapping[str, Any], trainable: Mapping[str, Any], mode: str):
    params = base_policy["params"]
    trunk = params["trunk"]
    if mode == "head":
        new_trunk = trunk
    elif mode == "last_block":
        new_trunk = {**trunk, "hidden_2": trainable["hidden_2"]}
    else:
        raise ValueError(mode)
    return {"params": {
        "neutral_loc": trainable["neutral_loc"],
        "scale_parameter": params["scale_parameter"],
        "trunk": new_trunk,
    }}


def extract_trainable(base_policy: Mapping[str, Any], mode: str):
    result = {"neutral_loc": copy.deepcopy(base_policy["params"]["neutral_loc"])}
    if mode == "last_block":
        result["hidden_2"] = copy.deepcopy(base_policy["params"]["trunk"]["hidden_2"])
    return result


def train_supervised(
    *, base_policy: Mapping[str, Any], actor_action: Any,
    teacher_observation: np.ndarray, teacher_target: np.ndarray,
    anchor_observation: np.ndarray, anchor_target: np.ndarray,
    learning_rate: float, steps: int = 500, mode: str = "head",
    callback: Any | None = None,
):
    """Full-batch 1:1 teacher/anchor Huber fitting with frozen nonactor assets."""
    teacher_obs = jnp.asarray(teacher_observation); teacher_y = jnp.asarray(teacher_target)
    anchor_obs = jnp.asarray(anchor_observation); anchor_y = jnp.asarray(anchor_target)
    trainable = extract_trainable(base_policy, mode)
    optimizer = optax.adam(float(learning_rate)); opt_state = optimizer.init(trainable)

    def loss_fn(value):
        policy = replace_trainable(base_policy, value, mode)
        teacher_action = actor_action(policy, teacher_obs)
        anchor_action = actor_action(policy, anchor_obs)
        teacher_loss = jnp.mean(optax.huber_loss(teacher_action, teacher_y, delta=.05))
        anchor_loss = jnp.mean(optax.huber_loss(anchor_action, anchor_y, delta=.05))
        return .5 * teacher_loss + .5 * anchor_loss, (teacher_loss, anchor_loss)

    update = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
    history = []
    for step in range(1, int(steps) + 1):
        (loss, aux), grads = update(trainable)
        updates, opt_state = optimizer.update(grads, opt_state, trainable)
        trainable = optax.apply_updates(trainable, updates)
        if step % 25 == 0:
            policy = replace_trainable(base_policy, trainable, mode)
            row = {"step":step,"loss":float(loss),"teacher_loss":float(aux[0]),"anchor_loss":float(aux[1])}
            if callback is not None: row.update(callback(step, policy))
            history.append(row)
    return replace_trainable(base_policy, trainable, mode), history


def make_fast_rollout(env: Any, frozen_params: Any):
    networks, _, _ = build_actor_tools(env, frozen_params)
    distribution = networks.parametric_action_distribution; normalizer = frozen_params[0]

    def rollout(policy, initial_state):
        active = jnp.asarray(True); survived = jnp.asarray(0, jnp.int32)
        minimum_margin = jnp.asarray(jnp.inf, jnp.float32); end_code = jnp.asarray(0, jnp.int32)
        saturation = jnp.asarray(0, jnp.int32); action_count = jnp.asarray(0, jnp.int32)

        def body(carry, tick):
            state, active, survived, minimum_margin, end_code, saturation, action_count = carry
            logits = networks.policy_network.apply(normalizer, policy, state.obs)
            action = distribution.mode(logits)
            next_state = env.step(state, action)
            margin = formal_dynamic_margin(env._physical_feature(next_state.data), env._config)
            alive = active & (~next_state.done.astype(bool))
            survived = survived + alive.astype(jnp.int32)
            minimum_margin = jnp.where(active, jnp.minimum(minimum_margin, margin), minimum_margin)
            end_code = jnp.where(active & next_state.done.astype(bool), next_state.info["end_code"], end_code)
            saturation += jnp.where(active, jnp.sum(jnp.abs(action) >= .95), 0)
            action_count += jnp.where(active, action.size, 0)
            return (next_state, alive, survived, minimum_margin, end_code, saturation, action_count), None

        final, _ = jax.lax.scan(body, (initial_state, active, survived, minimum_margin,
                                      end_code, saturation, action_count), jnp.arange(24))
        _, _, survived, minimum_margin, end_code, saturation, action_count = final
        return survived, minimum_margin, end_code, saturation, action_count
    return jax.jit(rollout)


def evaluate_policy(env: Any, rollout: Any, policy: Any, records: Sequence[Mapping[str, Any]], seed: int):
    rows = []
    for index, record in enumerate(records):
        state = restore_snapshot(env, record, jax.random.PRNGKey(seed + index * 1000))
        survived, margin, code, saturation, count = jax.device_get(rollout(policy, state))
        rows.append({
            "candidate_id":record["id"], "survived_ticks":int(survived),
            "minimum_formal_margin":float(margin), "end_code":int(code),
            "termination_reason":END_REASON.get(int(code), "pilot_horizon_reached" if int(survived) == 24 else "unknown"),
            "saturation_fraction":float(saturation) / max(int(count), 1),
        })
    ticks = np.asarray([row["survived_ticks"] for row in rows])
    return {"rows":rows,"summary":{
        "states":len(rows), "survival_counts":{str(h):int(np.sum(ticks >= h)) for h in (8,12,16,24)},
        "median":float(np.median(ticks)), "lower_quartile":float(np.quantile(ticks,.25)),
        "failure_reasons":dict(sorted(Counter(row["termination_reason"] for row in rows).items())),
        "saturation_fraction":float(np.mean([row["saturation_fraction"] for row in rows])),
    }}


def action_audit(actor_action: Any, base_policy: Any, policy: Any,
                 observation: np.ndarray, target: np.ndarray | None = None):
    obs = jnp.asarray(observation)
    base = np.asarray(actor_action(base_policy, obs)); current = np.asarray(actor_action(policy, obs))
    delta = current - base
    result = {"delta_rms":float(np.sqrt(np.mean(delta * delta))),"delta_max":float(np.max(np.abs(delta)))}
    if target is not None:
        error = current - np.asarray(target)
        result["imitation_rms"] = float(np.sqrt(np.mean(error * error)))
        residual = np.asarray(target) - base
        result["residual_direction_agreement"] = float(np.mean(np.sum(delta * residual, axis=-1) > 0))
    return result


def parameter_drift(base_policy: Any, policy: Any) -> float:
    before = jax.tree.leaves(base_policy); after = jax.tree.leaves(policy)
    numerator = sum(float(np.sum((np.asarray(a)-np.asarray(b))**2)) for a,b in zip(after,before))
    denominator = sum(float(np.sum(np.asarray(b)**2)) for b in before)
    return float(np.sqrt(numerator / max(denominator, 1e-30)))
