"""Runtime helpers for the final DVGC-Physical workflow.

This module keeps three artifacts separate:

* ``params.pkl``: the Brax PPO bundle ``(obs_normalizer, actor, critic)``;
* ``bank.pkl``: physical reset snapshots plus policy-conditioned Beta labels;
* Orbax files: implementation-level periodic Brax checkpoints.

The public command-line workflow should use ``params.pkl`` for PPO warm starts,
certification, rollout rendering and evaluation.  The bank is never a policy.
"""
from __future__ import annotations

import functools
import inspect
import json
import math
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jp
import numpy as np


# The actor consumes a short deployable sensor history.  The critic receives a
# separate privileged state vector returned by the environment during training.
# Keep these sizes in one module so training and frozen-policy inference always
# rebuild the same architecture.
POLICY_HIDDEN_LAYER_SIZES = (256, 256, 256)
VALUE_HIDDEN_LAYER_SIZES = (256, 256, 256)
POLICY_OBS_KEY = "state"
VALUE_OBS_KEY = "privileged_state"
POLICY_INITIAL_ACTION_STD = 0.05


def require_training_stack() -> Tuple[Any, Any, Any]:
    """Import Brax/Playground lazily with a useful error message."""
    try:
        from brax.training.agents.ppo import networks as ppo_networks
        from brax.training.agents.ppo import train as ppo_train_module
        ppo_train = getattr(ppo_train_module, "train", ppo_train_module)
        if not callable(ppo_train):
            raise TypeError(
                "Brax PPO import did not expose a callable train function: "
                f"got {type(ppo_train_module)!r}."
            )
        from mujoco_playground import wrapper
    except ImportError as exc:  # pragma: no cover - depends on user's install
        raise RuntimeError(
            "DVGC-MJX training needs brax, mujoco, mujoco-mjx and "
            "mujoco-playground in the same Python environment."
        ) from exc
    return ppo_networks, ppo_train, wrapper


def make_dvgc_ppo_networks(
    observation_size: Any,
    action_size: int,
    preprocess_observations_fn: Callable,
) -> Any:
    """Build a bounded actor with a neutral, low-variance control prior.

    Landing candidates already contain a large zero-action recoverable subset.
    Brax's generic tanh-normal head initializes both distribution halves with a
    random dense layer, yielding roughly 0.69 action standard deviation and a
    non-neutral deterministic mean.  That destroys the 25-step recovery hold
    before PPO sees useful successes.  This head keeps tanh bounds while making
    the initial mode exactly zero and the initial scale explicitly auditable.
    """
    ppo_networks, _, _ = require_training_stack()
    from brax.training import distribution
    from brax.training import networks as brax_networks
    from flax import linen

    target_scale_parameter = math.log(math.expm1(POLICY_INITIAL_ACTION_STD - 0.001))

    class NeutralTanhActor(linen.Module):
        @linen.compact
        def __call__(self, obs: jax.Array) -> jax.Array:
            hidden = brax_networks.MLP(
                layer_sizes=POLICY_HIDDEN_LAYER_SIZES,
                activation=linen.swish,
                activate_final=True,
                name="trunk",
            )(obs)
            loc = linen.Dense(
                int(action_size),
                kernel_init=jax.nn.initializers.zeros,
                bias_init=jax.nn.initializers.zeros,
                name="neutral_loc",
            )(hidden)
            scale_parameter = self.param(
                "scale_parameter",
                lambda _key, shape: jp.full(shape, target_scale_parameter, jp.float32),
                (int(action_size),),
            )
            scale_parameter = jp.broadcast_to(scale_parameter, loc.shape)
            return jp.concatenate((loc, scale_parameter), axis=-1)

    actor_module = NeutralTanhActor()
    actor_obs_size = observation_size[POLICY_OBS_KEY]
    actor_obs_size = int(math.prod(actor_obs_size))
    dummy_obs = jp.zeros((1, actor_obs_size), jp.float32)

    def actor_apply(processor_params: Any, policy_params: Any, observation: Any) -> jax.Array:
        actor_obs = observation[POLICY_OBS_KEY] if isinstance(observation, dict) else observation
        selected_params = (
            brax_networks.normalizer_select(processor_params, POLICY_OBS_KEY)
            if processor_params is not None and isinstance(observation, dict)
            else processor_params
        )
        actor_obs = preprocess_observations_fn(actor_obs, selected_params)
        return actor_module.apply(policy_params, actor_obs)

    policy_network = brax_networks.FeedForwardNetwork(
        init=lambda key: actor_module.init(key, dummy_obs),
        apply=actor_apply,
    )
    base = ppo_networks.make_ppo_networks(
        observation_size=observation_size,
        action_size=int(action_size),
        preprocess_observations_fn=preprocess_observations_fn,
        policy_hidden_layer_sizes=POLICY_HIDDEN_LAYER_SIZES,
        value_hidden_layer_sizes=VALUE_HIDDEN_LAYER_SIZES,
        policy_obs_key=POLICY_OBS_KEY,
        value_obs_key=VALUE_OBS_KEY,
    )
    return ppo_networks.PPONetworks(
        policy_network=policy_network,
        value_network=base.value_network,
        parametric_action_distribution=distribution.NormalTanhDistribution(
            event_size=int(action_size)
        ),
    )


def build_network_factory() -> Callable[..., Any]:
    """Build asymmetric PPO networks for deployable actor / privileged critic."""
    return make_dvgc_ppo_networks


def _atomic_pickle_dump(path: Path, payload: Any) -> None:
    """Write a pickle atomically so Ctrl+C never leaves a half-written model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def save_params(path: str | Path, params: Any) -> None:
    """Save the complete PPO inference bundle: normalizer + actor + critic."""
    path = Path(path)
    _atomic_pickle_dump(path, jax.device_get(params))


def save_normalizer(path: str | Path, params: Any) -> None:
    """Export a readable duplicate of the observation normalizer state.

    It is diagnostic only.  Always load the full ``params.pkl`` for inference,
    certification or PPO continuation so actor and normalizer cannot mismatch.
    """
    path = Path(path)
    if not isinstance(params, (tuple, list)) or len(params) < 1:
        raise ValueError("Expected Brax PPO params tuple (normalizer, actor, critic).")
    _atomic_pickle_dump(path, jax.device_get(params[0]))


def load_params(path: str | Path) -> Any:
    path = Path(path)
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, (tuple, list)) or len(payload) < 3:
        raise ValueError(
            f"{path} is not a complete Brax PPO bundle. Expected "
            "(obs_normalizer, actor_params, critic_params)."
        )
    return payload


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, default=_json_default)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.ndarray, jax.Array)):
        return np.asarray(value).tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return str(value)


def build_inference(env: Any, params: Any, *, deterministic: bool = True) -> Callable[[Any, jax.Array], Any]:
    """Recreate a saved PPO policy with its actor/critic normalization state.

    Frozen policy evaluation still passes the complete observation dictionary:
    the actor consumes ``state`` while the PPO bundle retains a separate
    normalizer slot for ``privileged_state``.  This prevents an asymmetric
    critic from changing inference-time actor preprocessing.
    """
    ppo_networks, _, _ = require_training_stack()
    # Brax moved this module from ``brax.training`` to
    # ``brax.training.acme``.  Keep a fallback for older Playground stacks
    # so frozen-policy probes remain compatible with the training environment.
    try:
        from brax.training.acme import running_statistics
    except ImportError:
        try:
            from brax.training import running_statistics
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Installed Brax exposes neither brax.training.acme.running_statistics "
                "nor brax.training.running_statistics. Reinstall the Brax version "
                "used by Mujoco Playground."
            ) from exc

    sample_obs = env.reset(jax.random.PRNGKey(0)).obs
    required = {POLICY_OBS_KEY, VALUE_OBS_KEY}
    missing = required.difference(sample_obs)
    if missing:
        raise RuntimeError(f"Environment observations are missing required PPO keys: {sorted(missing)}")
    observation_size = {key: tuple(value.shape) for key, value in sample_obs.items()}
    networks = make_dvgc_ppo_networks(
        observation_size=observation_size,
        action_size=int(env.action_size),
        preprocess_observations_fn=running_statistics.normalize,
    )
    make_inference_fn = ppo_networks.make_inference_fn(networks)
    return jax.jit(make_inference_fn(params, deterministic=deterministic))


def build_policy_distribution(env: Any, params: Any) -> Callable[[Any], Any]:
    """Return pre-tanh Gaussian loc/scale and deterministic action for probes."""
    try:
        from brax.training.acme import running_statistics
    except ImportError:
        from brax.training import running_statistics
    sample_obs=env.reset(jax.random.PRNGKey(0)).obs
    observation_size={key:tuple(value.shape) for key,value in sample_obs.items()}
    networks=make_dvgc_ppo_networks(
        observation_size=observation_size,action_size=int(env.action_size),
        preprocess_observations_fn=running_statistics.normalize,
    )
    def apply(obs):
        logits=networks.policy_network.apply(params[0],params[1],obs)
        dist=networks.parametric_action_distribution.create_dist(logits)
        return dist.loc,dist.scale,jp.tanh(dist.loc)
    return jax.jit(apply)


def ppo_rollout_block_steps(*, unroll_length: int, batch_size: int, num_minibatches: int) -> int:
    """Return one Brax PPO rollout/update quantum in environment steps."""
    values = (int(unroll_length), int(batch_size), int(num_minibatches))
    if any(v <= 0 for v in values):
        raise ValueError(f"PPO rollout dimensions must be positive, got {values}")
    return math.prod(values)


def ppo_effective_timesteps(
    requested_timesteps: int,
    *,
    unroll_length: int,
    batch_size: int,
    num_minibatches: int,
    num_evals: int = 1,
) -> int:
    """Mirror Brax epoch alignment and report scheduled training steps."""
    requested = int(requested_timesteps)
    if requested <= 0:
        raise ValueError(f"num_timesteps must be positive, got {requested}")
    block = ppo_rollout_block_steps(
        unroll_length=unroll_length, batch_size=batch_size, num_minibatches=num_minibatches
    )
    epochs = max(int(num_evals) - 1, 1)
    steps_per_epoch = int(math.ceil(requested / (epochs * block)))
    return int(epochs * steps_per_epoch * block)


def validate_ppo_batch_layout(*, num_envs: int, batch_size: int, num_minibatches: int) -> None:
    """Validate Brax PPO's rollout tensor layout before expensive compilation."""
    num_envs = int(num_envs)
    batch_size = int(batch_size)
    num_minibatches = int(num_minibatches)
    if min(num_envs, batch_size, num_minibatches) <= 0:
        raise ValueError("num_envs, batch_size and num_minibatches must all be positive")
    total_sequences = batch_size * num_minibatches
    if total_sequences % num_envs:
        suggestions = []
        for candidate_batch in (128, 256, 512, 1024, 2048, 4096, 8192):
            for candidate_minibatches in (4, 8, 16, 32, 64):
                product = candidate_batch * candidate_minibatches
                if product >= num_envs and product % num_envs == 0:
                    suggestions.append((product, candidate_batch, candidate_minibatches))
        suggestions.sort()
        hint = ""
        if suggestions:
            _, candidate_batch, candidate_minibatches = suggestions[0]
            hint = (
                f" For --num-envs {num_envs}, a compact valid choice is "
                f"--batch-size {candidate_batch} --num-minibatches {candidate_minibatches}."
            )
        raise ValueError(
            "Brax PPO requires batch_size * num_minibatches to be divisible "
            f"by num_envs, but {batch_size} * {num_minibatches} = "
            f"{total_sequences} is not divisible by {num_envs}." + hint
        )


def frozen_normalizer_training_params(params: Any) -> Any:
    """Convert a restored normalizer into an update-free PPO training state.

    The installed Brax release applies ``normalize_until_count`` only to its
    EMA path.  With a restored Welford state that option silently continues to
    change mean/std, which can move actions outside an immutable Tube retention
    trust region even when the actor weights themselves remain inside it.

    The conversion preserves mean/std and changes only the internal variance
    representation.  It must be paired with ``normalize_until_count=0``.  The
    exact original normalizer is restored in the published policy bundle after
    PPO, so this training-only state never becomes an artifact identity.
    """
    try:
        from brax.training.acme import running_statistics
    except ImportError:  # pragma: no cover - older supported Brax layout
        from brax.training import running_statistics

    if not isinstance(params, tuple) or len(params) != 3:
        raise ValueError("PPO params must be (normalizer, actor, critic)")
    normalizer, actor, critic = params
    variance = jax.tree_util.tree_map(
        lambda std: jp.maximum(jp.square(std) - normalizer.std_eps, 0.0),
        normalizer.std,
    )
    frozen = normalizer.replace(
        summed_variance=variance,
        mode=running_statistics.NormalizationMode.EMA,
    )
    return frozen, actor, critic


def normalizer_max_abs_difference(left: Any, right: Any) -> Dict[str, float]:
    """Return auditable mean/std differences between normalizer states."""
    def maximum(a: Any, b: Any) -> float:
        values = [
            float(np.max(np.abs(np.asarray(x) - np.asarray(y))))
            for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))
        ]
        return max(values, default=0.0)

    return {"mean": maximum(left.mean, right.mean), "std": maximum(left.std, right.std)}


def make_ppo_train_fn(
    *,
    timesteps: int,
    episode_length: int,
    num_envs: int,
    num_eval_envs: int,
    num_evals: int,
    seed: int,
    learning_rate: float,
    entropy_cost: float,
    reward_scaling: float,
    checkpoint_dir: str | Path | None,
    unroll_length: int = 32,
    batch_size: int = 1024,
    num_minibatches: int = 32,
    num_updates_per_batch: int = 2,
    discounting: float = 0.995,
    gae_lambda: float = 0.97,
    clipping_epsilon: float = 0.10,
    max_grad_norm: float = 0.75,
    log_training_metrics: bool = True,
    training_metrics_steps: Optional[int] = None,
    normalize_until_count: Optional[int] = None,
    restore_params: Optional[Any] = None,
    restore_checkpoint_path: Optional[str | Path] = None,
    policy_params_fn: Optional[Callable[..., None]] = None,
    full_reset: bool = False,
    run_evals: bool = True,
) -> Callable[..., Tuple[Any, Any, Any]]:
    """Return a Brax PPO training callable with normalized observations."""
    _, ppo_train, wrapper = require_training_stack()
    if full_reset:
        from .wrappers import wrap_for_training
        wrap_env_fn=wrap_for_training
    else:
        wrap_env_fn=wrapper.wrap_for_brax_training
    kwargs: Dict[str, Any] = dict(
        num_timesteps=int(timesteps),
        num_evals=int(num_evals),
        reward_scaling=float(reward_scaling),
        episode_length=int(episode_length),
        normalize_observations=True,
        normalize_observations_std_eps=1e-6,
        normalize_observations_mode="welford",
        normalize_until_count=(None if normalize_until_count is None
                               else int(normalize_until_count)),
        action_repeat=1,
        unroll_length=int(unroll_length),
        num_minibatches=int(num_minibatches),
        num_updates_per_batch=int(num_updates_per_batch),
        discounting=float(discounting),
        learning_rate=float(learning_rate),
        entropy_cost=float(entropy_cost),
        num_envs=int(num_envs),
        num_eval_envs=int(num_eval_envs),
        batch_size=int(batch_size),
        log_training_metrics=bool(log_training_metrics),
        training_metrics_steps=(None if training_metrics_steps is None else int(training_metrics_steps)),
        max_grad_norm=float(max_grad_norm),
        clipping_epsilon=float(clipping_epsilon),
        network_factory=build_network_factory(),
        seed=int(seed),
        save_checkpoint_path=(
            None
            if checkpoint_dir is None
            else str(Path(checkpoint_dir).expanduser().resolve())
        ),
        wrap_env_fn=wrap_env_fn,
        policy_params_fn=(policy_params_fn if policy_params_fn is not None else (lambda *_: None)),
    )
    signature = inspect.signature(ppo_train)
    if "gae_lambda" not in signature.parameters:
        raise RuntimeError("Installed Brax PPO does not expose gae_lambda; use the configured MuJoCo Playground training environment.")
    kwargs["gae_lambda"] = float(gae_lambda)
    # Evaluation is a policy-selection signal.  Use the mean action rather
    # than a newly sampled exploratory action at every evaluation rollout.
    # This keeps it distinct from the stochastic on-policy data collection.
    if "deterministic_eval" in signature.parameters:
        kwargs["deterministic_eval"] = True
    if "run_evals" in signature.parameters:
        kwargs["run_evals"] = bool(run_evals)
    elif not run_evals:
        raise RuntimeError("Installed Brax PPO does not expose run_evals")
    if restore_checkpoint_path is not None:
        if "restore_checkpoint_path" not in signature.parameters:
            raise RuntimeError("Installed Brax PPO does not support restore_checkpoint_path.")
        kwargs["restore_checkpoint_path"] = str(Path(restore_checkpoint_path).expanduser().resolve())
    if restore_params is not None:
        if "restore_params" not in signature.parameters:
            raise RuntimeError("Installed Brax PPO does not expose restore_params.")
        kwargs["restore_params"] = restore_params
    return functools.partial(ppo_train, **kwargs)


def assert_brax_metric_contract(env: Any) -> None:
    """Host-side preflight for Brax EvalWrapper's static metric dictionary."""
    state = env.reset(jax.random.PRNGKey(0))
    reset_metrics = dict(state.metrics)
    if "reward" not in reset_metrics:
        raise RuntimeError("Raw DVGC reset metrics omit Brax-required metrics['reward'].")
    reset_metrics["reward"] = state.reward
    wrapped_like_state = state.replace(metrics=reset_metrics)
    action = jp.zeros((int(env.action_size),), dtype=jp.float32)
    # Keep Warp on the same JIT custom-call path used by rollouts and PPO.
    # An eager call creates a distinct Warp execution path whose shared
    # scratch state can perturb a later, otherwise exact snapshot replay.
    next_state = jax.jit(env.step)(wrapped_like_state, action)
    jax.block_until_ready(next_state)
    before, after = set(reset_metrics), set(next_state.metrics)
    if before != after:
        raise RuntimeError(
            "DVGC metric dictionary changes across raw env.step under Brax "
            f"EvalWrapper semantics. missing={sorted(before - after)}, "
            f"unexpected={sorted(after - before)}"
        )


def scalar(value: Any) -> float:
    return float(np.asarray(jax.device_get(value)))


def integer(value: Any) -> int:
    return int(np.asarray(jax.device_get(value)))


def metric_row(state: Any) -> Dict[str, float]:
    """Convert a raw MJX State's scalar metrics and terminal code to host data."""
    row: Dict[str, float] = {}
    for key, value in state.metrics.items():
        try:
            row[key] = scalar(value)
        except Exception:
            pass
    for key in ("phase", "end_code", "recovery_success", "chain_success", "episode_step", "contact_age"):
        if key in state.info:
            try:
                row[f"info/{key}"] = float(integer(state.info[key]))
            except Exception:
                pass
    row["done"] = scalar(state.done)
    return row
