"""Fresh single-Actor distillation and opt-in fixed-data PPO supervision.

Offline actions are supervised targets only, never off-policy PPO transitions.
The caller owns provenance, TRAIN/development separation, and teacher selection.
"""
from pathlib import Path
import hashlib
import json
import math
import numpy as np


def load_dataset(path):
    with np.load(path, allow_pickle=False) as raw:
        obs = np.asarray(raw['observations'], dtype=np.float32)
        actions = np.asarray(raw['actions'], dtype=np.float32)
        weights = np.asarray(raw['weights'], dtype=np.float64)
    if obs.ndim != 2 or len(obs) == 0 or not np.isfinite(obs).all():
        raise ValueError('observations must be finite nonempty NxD')
    if actions.shape != (len(obs), 4) or not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1:
        raise ValueError('actions must be finite normalized Nx4')
    if weights.shape != (len(obs),) or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError('weights must be finite nonnegative with positive total')
    return obs, actions, (weights / weights.sum()).astype(np.float32)


def _count_float(count):
    import jax.numpy as jp
    if hasattr(count, 'hi') and hasattr(count, 'lo'):
        return jp.asarray(count.hi, jp.float32) * 4294967296. + jp.asarray(count.lo, jp.float32)
    return jp.asarray(count, jp.float32)


def _write(path, result):
    Path(path).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


def distill(dataset_path, output_checkpoint, *, identity, privileged_observation_size,
            seed=0, updates=2000, batch_size=256, learning_rate=3e-4):
    """Save a standard checkpoint with fresh actor/critic, zero RL transitions.

    Actor normalization is estimated from weighted TRAIN observations. Privileged
    normalization starts as identity; downstream PPO must use a fresh critic.
    """
    import jax
    import jax.numpy as jp
    import optax
    from brax.training.acme import running_statistics as rs
    from .ppo import make_network_factory
    from .checkpoint import CheckpointPayload, save_checkpoint, load_checkpoint
    if any(type(x) is not int or x <= 0 for x in (updates, batch_size, privileged_observation_size)):
        raise ValueError('updates, batch size, privileged size must be positive integers')
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('learning rate must be finite and positive')
    output = Path(output_checkpoint)
    if output.exists():
        raise FileExistsError(output)
    observations, actions, weights = load_dataset(dataset_path)
    obs, targets, probs = map(jp.asarray, (observations, actions, weights))
    size = {'state': obs.shape[1], 'privileged_state': privileged_observation_size}
    networks = make_network_factory()(size, 4, preprocess_observations_fn=rs.normalize)
    key, actor_key, critic_key = jax.random.split(jax.random.PRNGKey(seed), 3)
    actor = networks.policy_network.init(actor_key)
    critic = networks.value_network.init(critic_key)
    normalizer = rs.init_state({k: jp.zeros(v) for k, v in size.items()})
    actor_stats = rs.update(rs.init_state(jp.zeros(obs.shape[1])), obs,
                           weights=probs * len(obs))
    # Sparse successful teacher data contains constant channels. Unit minimum
    # scale prevents tiny empirical variance amplifying new student states.
    actor_std = jp.maximum(actor_stats.std, 1.)
    # Preserve a coherent tree and weighted count while privileged observations
    # remain unobserved. The critic is reinitialized by downstream PPO.
    normalizer = normalizer.replace(
        count=actor_stats.count,
        mean={**normalizer.mean, 'state': actor_stats.mean},
        std={**normalizer.std, 'state': actor_std},
        summed_variance={'privileged_state': jp.ones(privileged_observation_size) * _count_float(actor_stats.count),
                         'state': jp.square(actor_std) * _count_float(actor_stats.count)})
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(actor)

    def prediction(params, x):
        logits = networks.policy_network.apply(normalizer, params, {'state': x})
        return networks.parametric_action_distribution.mode(logits)

    @jax.jit
    def step(params, state, rng):
        indices = jax.random.choice(rng, len(obs), (batch_size,), p=probs)
        def loss(p):
            return jp.mean(jp.square(prediction(p, obs[indices]) - targets[indices]))
        value, grad = jax.value_and_grad(loss)(params)
        delta, state = optimizer.update(grad, state, params)
        return optax.apply_updates(params, delta), state, value

    def full_score(params):
        total = 0.
        for begin in range(0, len(obs), batch_size):
            end = begin + batch_size
            error = np.asarray(jp.mean(jp.square(prediction(params, obs[begin:end]) - targets[begin:end]), axis=-1))
            total += float(np.sum(error * weights[begin:end]))
        return total

    initial = full_score(actor)
    output.parent.mkdir(parents=True, exist_ok=True)
    logs = output.parent / (output.name + '_distillation_metrics.jsonl')
    with logs.open('x') as stream:
        for iteration in range(updates):
            key, sample_key = jax.random.split(key)
            actor, opt_state, loss = step(actor, opt_state, sample_key)
            if iteration % 100 == 0 or iteration == updates - 1:
                value = float(loss)
                if not math.isfinite(value):
                    raise ValueError('nonfinite distillation loss')
                stream.write(json.dumps({'update': iteration + 1, 'action_mse': value}) + '\n')
                stream.flush()
    final = full_score(actor)
    if not math.isfinite(final) or not all(np.isfinite(x).all() for x in jax.tree.leaves(jax.device_get(actor))):
        raise ValueError('nonfinite distilled actor')
    payload = CheckpointPayload(identity, 0, jax.device_get(normalizer),
                                jax.device_get(actor), jax.device_get(critic))
    save_checkpoint(output, payload)
    load_checkpoint(output, expected=identity)
    summary = dict(schema='jit_policy_distillation_v1', dataset=str(Path(dataset_path).resolve()),
        dataset_sha256=hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest(),
        samples=len(obs), actor_observation_dimension=obs.shape[1], seed=seed,
        updates=updates, batch_size=batch_size, learning_rate=learning_rate,
        initial_action_mse=initial, final_action_mse=final,
        actor_initialization='fresh_random', critic_initialization='fresh_random',
        actor_normalizer='weighted_train_observations_unit_minimum_scale',
        privileged_normalizer='zero_mean_unit_variance_pseudocount_matching_actor',
        environment_interactions=0,
        checkpoint=str(output.resolve()), checkpoint_restored=True,
        score_role='training_fit_not_generalization')
    _write(output / 'distillation.json', summary)
    return summary


def wrap_trainer(trainer, dataset_path, *, coefficient, batch_size=256,
                 decay_transitions=None, checkpoint_path=None, identity=None):
    """Add fixed supervised actions to PPO; optional dynamic linear decay.

    The decay clock is the increase in Brax normalization sample count during
    this invocation (environment observations), not Python/JIT trace calls.
    A checkpoint injects actor+normalizer restoration and a fresh PPO critic.
    """
    if not math.isfinite(coefficient) or coefficient < 0 or type(batch_size) is not int or batch_size <= 0:
        raise ValueError('invalid imitation coefficient or batch size')
    if decay_transitions is not None and (not math.isfinite(decay_transitions) or decay_transitions <= 0):
        raise ValueError('decay_transitions must be positive')
    observations, actions, weights = load_dataset(dataset_path)

    def train(**kwargs):
        import jax
        import jax.numpy as jp
        from brax.training.agents.ppo import losses
        if checkpoint_path is not None:
            from .checkpoint import load_checkpoint
            if identity is None:
                raise ValueError('checkpoint identity required')
            restored = load_checkpoint(checkpoint_path, expected=identity)
            kwargs['restore_params'] = (restored.observation_normalizer, restored.actor_params, restored.critic_params)
            kwargs['restore_value_fn'] = False
        if kwargs.get('restore_params') is None:
            raise ValueError('distilled restore_params required')
        start_count = _count_float(kwargs['restore_params'][0].count)
        obs, targets, probs = map(jp.asarray, (observations, actions, weights))
        original = losses.compute_ppo_loss

        def loss(params, normalizer_params, data, rng, ppo_network, **options):
            base, metrics = original(params, normalizer_params, data, rng, ppo_network, **options)
            indices = jax.random.choice(jax.random.fold_in(rng, 173), len(obs), (batch_size,), p=probs)
            logits = ppo_network.policy_network.apply(normalizer_params, params.policy, {'state': obs[indices]})
            predicted = ppo_network.parametric_action_distribution.mode(logits)
            mse = jp.mean(jp.square(predicted - targets[indices]))
            scale = coefficient
            if decay_transitions is not None:
                scale = coefficient * jp.clip(1 - (_count_float(normalizer_params.count)-start_count)/decay_transitions, 0, 1)
            total = base + scale * mse if coefficient else base
            return total, {**metrics, 'ppo_loss': base, 'distillation_action_mse': mse,
                           'distillation_coefficient': scale, 'total_loss': total}
        losses.compute_ppo_loss = loss
        try:
            return trainer(**kwargs)
        finally:
            losses.compute_ppo_loss = original
    return train
