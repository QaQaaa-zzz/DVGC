"""Privileged PPO explorer with an actor-only frozen-policy warm start.

The explorer consumes the existing critic observation, whose prefix is the full
three-frame actor observation. No critic or optimizer state crosses this helper.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from brax.training.acme import running_statistics
from flax.core import FrozenDict, freeze, unfreeze
import jax
import jax.numpy as jp

from .constants import ACTOR_OBSERVATION_SIZE, PRIVILEGED_OBSERVATION_SIZE
from .ppo import make_network_factory


def make_exploration_network_factory():
    """Use the same 256x3, four-action tanh-normal PPO architecture."""
    factory = make_network_factory()
    factory.keywords.update(policy_obs_key="privileged_state", value_obs_key="privileged_state")
    return factory


def widen_actor_warm_start(
    observation_normalizer: Any,
    actor_params: Any,
    *,
    actor_observation_size: int = ACTOR_OBSERVATION_SIZE,
    privileged_observation_size: int = PRIVILEGED_OBSERVATION_SIZE,
) -> tuple[Any, Any]:
    """Return Brax actor-only restore params for the privileged explorer.

    Preserve all original actor weights, normalizer count, and actor statistics.
    Extra input rows are zero. Privileged normalization uses the actor statistics
    in its shared prefix and a unit-variance, zero-mean prior for extra features.
    The prior summed variance is consistent with the inherited global count, so
    a subsequent running-statistics update does not collapse the new variance.
    The caller must initialize a fresh value network and optimizer (for Brax PPO,
    use ``restore_value_fn=False`` with this two-element restore tuple).
    """
    if not 0 < actor_observation_size < privileged_observation_size:
        raise ValueError("warm start requires a strictly wider privileged observation")
    if not isinstance(observation_normalizer, running_statistics.RunningStatisticsState):
        raise ValueError("warm start requires Brax RunningStatisticsState")
    stats = observation_normalizer
    for field in ("mean", "std", "summed_variance"):
        tree = getattr(stats, field)
        if not isinstance(tree, Mapping) or "state" not in tree or "privileged_state" not in tree:
            raise ValueError("normalizer requires state and privileged_state statistics")
        if tree["state"].shape != (actor_observation_size,) or tree["privileged_state"].shape != (privileged_observation_size,):
            raise ValueError("normalizer observation shape mismatch")
    params = unfreeze(actor_params) if isinstance(actor_params, FrozenDict) else jax.tree.map(lambda x: x, actor_params)
    try:
        kernel = params["params"]["hidden_0"]["kernel"]
    except (KeyError, TypeError) as exc:
        raise ValueError("unsupported frozen actor parameter structure") from exc
    if kernel.shape != (actor_observation_size, 256):
        raise ValueError("frozen actor first-layer shape mismatch")
    extra = privileged_observation_size - actor_observation_size
    params["params"]["hidden_0"]["kernel"] = jp.concatenate(
        (kernel, jp.zeros((extra, kernel.shape[1]), dtype=kernel.dtype)), axis=0
    )
    if isinstance(actor_params, FrozenDict):
        params = freeze(params)
    count = stats.count
    count_float = (jp.float32(count.hi) * jp.float32(2.0**32) + jp.float32(count.lo)
                   if hasattr(count, "hi") else jp.asarray(count, dtype=jp.float32))
    # Welford stores accumulated variance; EMA stores variance directly.
    scale = count_float if stats.mode == running_statistics.NormalizationMode.WELFORD else jp.float32(1)
    variance_prior = scale * jp.maximum(1.0 - stats.std_eps, 0.0)
    updates = {}
    for field, fill in (("mean", 0.0), ("std", 1.0), ("summed_variance", variance_prior)):
        original = getattr(stats, field)
        tree = dict(original)
        prefix = original["state"]
        tree["privileged_state"] = jp.concatenate((prefix, jp.full((extra,), fill, dtype=prefix.dtype)))
        updates[field] = freeze(tree) if isinstance(original, FrozenDict) else tree
    return stats.replace(**updates), params
