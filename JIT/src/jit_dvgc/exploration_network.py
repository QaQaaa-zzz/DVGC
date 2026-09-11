"""Privileged PPO networks and frozen-policy residual action composition.

The residual actor is fresh with a zero mean head and uses the existing critic
observation. The historical full-action widening helper remains available for
reproducing prior experiments; it is not the residual initialization contract.
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


def initialize_residual_actor(networks: Any, key: Any) -> Any:
    """Initialize a fresh privileged delta actor with zero deterministic output.

    The canonical tanh-normal output is four means followed by four scale
    parameters. Only the mean head is zeroed; hidden layers and stochastic scale
    retain their fresh canonical initialization. No frozen actor is copied.
    Pass the frozen checkpoint's existing 106-feature privileged normalizer
    alongside this result; do not call ``widen_actor_warm_start`` for residuals.
    """
    original = networks.policy_network.init(key)
    params = unfreeze(original) if isinstance(original, FrozenDict) else jax.tree.map(lambda x: x, original)
    try:
        first = params['params']['hidden_0']['kernel']
        head = params['params']['hidden_3']
        kernel, bias = head['kernel'], head['bias']
    except (KeyError, TypeError) as exc:
        raise ValueError('residual actor requires canonical privileged 256x3 network') from exc
    if first.shape != (PRIVILEGED_OBSERVATION_SIZE, 256) or kernel.shape != (256, 8) or bias.shape != (8,):
        raise ValueError('residual actor requires canonical privileged 256x3 four-action network')
    head['kernel'] = kernel.at[:, :4].set(0)
    head['bias'] = bias.at[:4].set(0)
    return freeze(params) if isinstance(original, FrozenDict) else params


def validate_residual_delta_limit(delta_limit: Any) -> tuple[float, float, float, float]:
    """Validate explicit per-channel normalized-action bounds on the host."""
    import numpy as np

    try:
        limit = np.asarray(delta_limit, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('delta_limit requires four finite values in (0, 2]') from exc
    if limit.shape != (4,) or not np.isfinite(limit).all() or np.any(limit <= 0) or np.any(limit > 2):
        raise ValueError('delta_limit requires four finite values in (0, 2]')
    return tuple(float(x) for x in limit)


def compose_residual_action(base_action: Any, normalized_delta: Any, delta_limit: Any) -> tuple[Any, Any, Any]:
    """Compose frozen pi and bounded four-channel delta, including saturation.

    Validate limits once with ``validate_residual_delta_limit`` before tracing.
    Inputs are normalized action coordinates; arbitrary leading batch dimensions
    are supported. The returned requested delta precedes actuator clipping, and
    effective delta is the actual change after clipping. Base-policy gradients
    are stopped explicitly; PPO optimizes only its normalized delta distribution.
    """
    base = jax.lax.stop_gradient(jp.asarray(base_action))
    delta = jp.asarray(normalized_delta)
    limit = jp.asarray(delta_limit)
    if base.ndim < 1 or base.shape[-1] != 4 or delta.shape != base.shape or limit.shape != (4,):
        raise ValueError('residual action requires matching (..., 4) actions and delta_limit shape (4,)')
    requested = jp.clip(delta, -1., 1.) * limit
    executed = jp.clip(base + requested, -1., 1.)
    return executed, requested, executed - base
