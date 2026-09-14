"""Opt-in frozen-source action supervision, separate from on-policy PPO data."""
from pathlib import Path
import math
import numpy as np


def action_mse(student, teacher):
    import jax
    import jax.numpy as jp
    return jp.mean(jp.square(student - jax.lax.stop_gradient(teacher)))


def anchor_weights(rows):
    if not rows or any(r.get('witnessed') is not True or
            r['phase'] not in ('upstream', 'downstream') or
            not math.isfinite(r['sampling_weight']) or r['sampling_weight'] <= 0 for r in rows):
        raise ValueError('positive witnessed anchor rows required')
    totals = {phase: sum(r['sampling_weight'] for r in rows if r['phase'] == phase)
              for phase in ('upstream', 'downstream')}
    if min(totals.values()) <= 0:
        raise ValueError('both anchor phases required')
    return np.asarray([.5 * r['sampling_weight'] / totals[r['phase']] for r in rows])


def paired_counts(old, new):
    if len(old) != len(new):
        raise ValueError('paired labels must have equal length')
    result = dict(retained=0, lost=0, gained=0, unchanged_negative=0,
                  unknown=0, old_positive=sum(x == 1 for x in old))
    for a, b in zip(old, new):
        name = {(1, 1): 'retained', (1, 0): 'lost', (0, 1): 'gained',
                (0, 0): 'unchanged_negative'}.get((a, b), 'unknown')
        result[name] += 1
    return result


def load_anchor(raw, source):
    from .jump_evidence_validation import read, file_sha, verify_hash
    from .unified_envelope_snapshot import load_unified_envelope_snapshot as load_snapshot
    if raw.get('schema') != 'jit_source_action_retention_v1':
        raise ValueError('unsupported action retention schema')
    if (type(raw.get('coefficient')) not in (float, int) or
            not math.isfinite(raw['coefficient']) or raw['coefficient'] < 0 or
            type(raw.get('batch_size')) is not int or raw['batch_size'] <= 0):
        raise ValueError('finite nonnegative coefficient and positive anchor batch required')
    path = Path(raw['support'])
    if file_sha(path) != raw['support_file_sha256']:
        raise ValueError('anchor support hash changed')
    support = read(path)
    verify_hash(support, 'support_sha256')
    if support.get('role') != 'train' or support.get('final_test_used') is not False:
        raise ValueError('only TRAIN anchors permitted')
    for path, digest in support['inputs'].items():
        if file_sha(Path(path)) != digest:
            raise ValueError('anchor input hash changed')
    rows = support['entries']
    weights = anchor_weights(rows)
    if any(r.get('labels', {}).get(source) != 1 for r in rows):
        raise ValueError('every anchor must have a positive source-policy witness')
    observations = np.stack([load_snapshot(Path(r['snapshot'])).observation for r in rows])
    if observations.ndim != 2 or not np.isfinite(observations).all():
        raise ValueError('finite complete actor observations required')
    return observations, weights


def wrap_trainer(trainer, contract, source_name, run_dir):
    """Scope the Brax loss adapter to one synchronous trainer invocation.

    No site-package files are changed. Each experiment runs in its own process;
    the original loss is restored even when training fails.
    """
    observations, weights = load_anchor(contract, source_name)

    def train(**kwargs):
        import jax
        import jax.numpy as jp
        from brax.training.agents.ppo import losses
        from .jump_evidence_validation import write
        source_normalizer, source_actor, _ = kwargs['restore_params']
        original = losses.compute_ppo_loss
        obs, probs = jp.asarray(observations), jp.asarray(weights)
        coefficient = contract['coefficient']

        def loss(params, normalizer_params, data, rng, ppo_network, **options):
            base, metrics = original(params, normalizer_params, data, rng, ppo_network, **options)
            indices = jax.random.choice(jax.random.fold_in(rng, 719), obs.shape[0],
                (contract['batch_size'],), p=probs)
            anchor = {'state': obs[indices]}
            apply = ppo_network.policy_network.apply
            distribution = ppo_network.parametric_action_distribution
            teacher = distribution.mode(apply(source_normalizer, source_actor, anchor))
            student = distribution.mode(apply(normalizer_params, params.policy, anchor))
            mse = action_mse(student, teacher)
            penalty = coefficient * mse
            # Keep coefficient-zero gradient path identical to ordinary PPO.
            total = base + penalty if coefficient else base
            return total, {**metrics, 'ppo_loss': base, 'retention_action_mse': mse,
                'retention_penalty': penalty, 'total_loss': total}

        write(Path(run_dir) / 'retention_hyperparameters.json', {
            **contract, 'source_policy': source_name, 'anchor_count': len(observations),
            'actor_observation_dimension': observations.shape[1],
            'teacher_normalizer': 'frozen_source', 'student_normalizer': 'current',
            'actor_parameter_count': sum(x.size for x in jax.tree.leaves(source_actor)),
            'critic_parameter_count': sum(x.size for x in jax.tree.leaves(kwargs['restore_params'][2])),
            'loss': 'PPO + coefficient * mean squared deterministic normalized action difference',
            'ppo_replay_used': False, 'teacher_trainable': False,
            'environment_interactions': 0})
        progress = kwargs.get('progress_fn')
        milestones = set(contract.get('plot_transitions', []))
        def on_progress(step, metrics):
            if progress is not None:
                progress(step, metrics)
            if int(step) in milestones and any(not str(k).startswith('episode/') for k in metrics):
                from .retention_experiment import export_training
                export_training(run_dir)
        kwargs['progress_fn'] = on_progress
        losses.compute_ppo_loss = loss
        try:
            return trainer(**kwargs)
        finally:
            losses.compute_ppo_loss = original
            from .retention_experiment import export_training
            export_training(run_dir)
    return train
