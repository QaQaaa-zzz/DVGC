"""T0: immutable saved-TRAIN inference and gradients, never simulation or learning.

The independent forward implementation reconstructs normalization, Swish layers,
tanh mode and legal clipping.  It is compared with the production Brax network
and saved GPU actions.  Only pi0 reconstruction calibrates the fixed tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np

SCHEMA = 'jit_retention_T0_saved_diagnostics_v1'
NODES = (0, 100, 500, 1000)
ACTION_ORDER = ('steer', 'rear_wheel_drive', 'hip', 'knee')
FOCUS = ('TRAIN-1010233101-30', 'TRAIN-1010233101-68', 'TRAIN-1010233101-55')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class SourceRegistry:
    def __init__(self):
        self.files = {}

    def register(self, path, expected=None):
        path = Path(path).resolve()
        if any(part.upper() in ('TEST', 'JCE', 'JEL') for part in path.parts):
            raise ValueError('TEST/JCE/JEL is not an allowed input')
        key = str(path)
        digest = self.files.get(key)
        if digest is None:
            digest = file_sha(path)
            self.files[key] = digest
        if expected is not None and digest != expected:
            raise ValueError(f'input hash mismatch: {path}')
        return path

    def json(self, path, expected=None):
        return json.loads(self.register(path, expected).read_text())

    def arrays(self, path, keys=None, expected=None):
        with np.load(self.register(path, expected), allow_pickle=False) as data:
            return {k: data[k].copy() for k in (keys if keys is not None else data.files)}

    def verify_unchanged(self):
        for path, digest in self.files.items():
            if file_sha(path) != digest:
                raise ValueError(f'source changed during T0: {path}')


def create_output(path, sources):
    path = Path(path).resolve()
    if any(path == Path(s).resolve() or Path(s).resolve() in path.parents for s in sources):
        raise ValueError('output must be outside original source runs')
    path.mkdir(parents=True, exist_ok=False)
    return path


def require_train(record):
    if (str(record.get('role', record.get('data_role', ''))).upper() != 'TRAIN'
            or record.get('DEV_used', False)):
        raise ValueError('only explicit TRAIN data is permitted')


def calibration_tolerance(errors):
    if not errors:
        raise ValueError('nonempty pi0 calibration required')
    a = np.asarray(errors)
    if not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError('finite nonnegative calibration errors required')
    return max(1e-6, 10. * float(a.max()))


def chain_gate(records, tolerance):
    failures = [r for r in records if not np.isfinite(r['maxabs']) or r['maxabs'] > tolerance]
    if not records:
        failures = [{'name': 'missing_execution_chain', 'maxabs': None}]
    failures = [{**r, 'maxabs': float(r['maxabs']) if r['maxabs'] is not None and np.isfinite(r['maxabs']) else None} for r in failures]
    return dict(schema=SCHEMA, phase='blocked' if failures else 'passed',
                t1_allowed=not bool(failures), tolerance=tolerance,
                blocking_failures=failures, simulation_steps=0, supervised_updates=0)


def error_statistics(prediction, target):
    diff = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    if diff.ndim != 2 or diff.shape[1] != 4 or not len(diff) or not np.isfinite(diff).all():
        raise ValueError('finite nonempty four-action error arrays required')
    return dict(n=len(diff), mse=float(np.mean(diff ** 2)),
                mse_by_action=np.mean(diff ** 2, axis=0).tolist(),
                rmse=float(np.sqrt(np.mean(diff ** 2))), maxabs=float(np.max(np.abs(diff))),
                p95_frame_rmse=float(np.quantile(np.sqrt(np.mean(diff ** 2, axis=1)), .95)))


def reconstruction_statistics(independent, native, saved):
    return dict(**error_statistics(independent, saved),
                native_cpu_vs_saved_maxabs=error_statistics(native, saved)['maxabs'],
                independent_cpu_vs_native_maxabs=error_statistics(independent, native)['maxabs'])


def independent_forward(actor, mean, std, observations):
    """Direct JAX arithmetic; no production preprocessor/network apply is called."""
    import jax.numpy as jp
    x = (jp.asarray(observations, dtype=jp.float32) - jp.asarray(mean)) / jp.asarray(std)
    layers = actor['params']
    names = sorted(layers, key=lambda k: int(k.rsplit('_', 1)[1]))
    for i, name in enumerate(names):
        layer = layers[name]
        x = x @ jp.asarray(layer['kernel']) + jp.asarray(layer['bias'])
        if i < len(names) - 1:
            x = x / (1. + jp.exp(-x))
    return jp.clip(jp.tanh(x[..., :4]), -1., 1.)


def gradient_cosine(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm > 1e-20 else None


def tree_sha(tree):
    import jax
    leaves, definition = jax.tree_util.tree_flatten(tree)
    digest = hashlib.sha256(str(definition).encode())
    for leaf in leaves:
        a = np.asarray(leaf)
        digest.update(str(a.dtype).encode('ascii'))
        digest.update(repr(tuple(a.shape)).encode('ascii'))
        digest.update(np.ascontiguousarray(a).tobytes())
    return digest.hexdigest()


TRACE_KEYS = ('actor_observation_before', 'normalized_action_executed', 'mask',
              'first_valid_contact', 'success', 'physical_failure', 'qpos', 'qvel',
              'roll', 'time_before', 'episode_step')


def saved_trace(registry, attempt):
    arrays = registry.arrays(attempt['trace'], TRACE_KEYS, attempt.get('trace_sha256'))
    lane = attempt['trace_lane']
    mask = arrays['mask'][:, lane].astype(bool)
    result = {k: v[mask, lane] for k, v in arrays.items() if k != 'mask'}
    if len(result['actor_observation_before']) != attempt['steps']:
        raise ValueError('trace length does not match successful/failed attempt receipt')
    return result


def contact_phase(trace):
    seen = np.maximum.accumulate(trace['first_valid_contact'].astype(bool))
    # Observations precede the transition: the contact transition itself is pre-contact.
    before = np.r_[False, seen[:-1]]
    return np.where(before, 'post_contact', 'pre_contact')


@dataclass
class Corpus:
    name: str
    observations: np.ndarray
    actions: np.ndarray
    weights: np.ndarray
    roots: np.ndarray
    segments: np.ndarray
    phases: np.ndarray
    trajectories: np.ndarray

    def validate(self):
        n = len(self.observations)
        if self.observations.shape != (n, 76) or self.actions.shape != (n, 4) or not n:
            raise ValueError(f'invalid corpus dimensions: {self.name}')
        if not all(len(x) == n for x in (self.weights, self.roots, self.segments, self.phases, self.trajectories)):
            raise ValueError('corpus metadata length mismatch')
        if not all(np.isfinite(x).all() for x in (self.observations, self.actions, self.weights)):
            raise ValueError('nonfinite corpus')
        if np.max(np.abs(self.actions)) > 1.000001 or np.any(self.weights < 0) or self.weights.sum() <= 0:
            raise ValueError('invalid normalized targets/weights')
        return self


def load_inputs(control, intervention, registry):
    runs = {'control': Path(control).resolve(), 'intervention': Path(intervention).resolve()}
    plans = {name: registry.json(r / 'plan.json') for name, r in runs.items()}
    rows = {name: registry.json(r / 'B_checkpoint_results.json') for name, r in runs.items()}
    diagnostics = {name: registry.json(r / 'selected_diagnostics.json') for name, r in runs.items()}
    workers = {name: registry.json(r / 'worker_inputs.json') for name, r in runs.items()}
    for name, r in runs.items():
        if registry.json(r / 'status.json')['phase'] != 'completed':
            raise ValueError('T0 requires completed frozen runs')
        if not all(str(u) in rows[name] for u in NODES):
            raise ValueError('missing matched checkpoint node')
        if plans[name]['models']['pi0']['policy']['action_order'] != list(ACTION_ORDER):
            raise ValueError('action channel order mismatch')
    old = registry.json(plans['control']['demo_manifest'])
    new = registry.json(plans['intervention']['demo_manifest'])
    require_train(old); require_train(new)
    a = registry.arrays(old['path'], expected=old['sha256'])
    b = registry.arrays(new['path'], expected=new['sha256'])
    if old['count'] != 525 or new['old_count'] != 525 or new['new_count'] != 228:
        raise ValueError('frozen 525+228 demo contract mismatch')
    if not all(np.array_equal(a[k], b[k][:525]) for k in ('observations', 'actions')):
        raise ValueError('original teacher observations/actions changed')
    teacher_manifest = registry.json(old['teacher_manifest'])
    teachers = {}
    old_phases = []
    for lesson in teacher_manifest['TRAIN']:
        require_train(lesson)
        trace = saved_trace(registry, lesson['trace'])
        ids = np.array(old['sample_roots']) == lesson['root_id']
        if not np.array_equal(a['observations'][ids], trace['actor_observation_before']):
            raise ValueError('teacher state provenance mismatch')
        if not np.array_equal(a['actions'][ids], trace['normalized_action_executed']):
            raise ValueError('teacher executed target mismatch')
        teachers[lesson['root_id']] = trace
    for rid in dict.fromkeys(old['sample_roots']):
        old_phases.extend(contact_phase(teachers[rid]))
    new_phases, new_trajectories = [], []
    cursor = 525
    for entry in new['new_entries']:
        trace = saved_trace(registry, entry)
        n = entry['count']
        if not np.array_equal(b['observations'][cursor:cursor+n], trace['actor_observation_before']):
            raise ValueError('new teacher state provenance mismatch')
        if not np.array_equal(b['actions'][cursor:cursor+n], trace['normalized_action_executed']):
            raise ValueError('new teacher action provenance mismatch')
        new_phases.extend(contact_phase(trace)); new_trajectories.extend([entry['trajectory_id']] * n)
        cursor += n
    if cursor != 753:
        raise ValueError('incomplete new demo frame coverage')
    corpora = [Corpus('old_demo', a['observations'], a['actions'], a['weights'],
                      np.array(old['sample_roots']), np.array(old['sample_origins']),
                      np.array(old_phases), np.array(old['sample_roots'])).validate(),
               Corpus('added_demo', b['observations'][525:], b['actions'][525:], b['weights'][525:],
                      np.array(new['sample_roots'][525:]), np.array(new['sample_origins'][525:]),
                      np.array(new_phases), np.array(new_trajectories)).validate()]
    keep_receipt = registry.json(workers['control']['keep_receipt'], workers['control']['keep_receipt_sha256'])
    require_train(keep_receipt)
    if workers['control']['keep_data_sha256'] != workers['intervention']['keep_data_sha256']:
        raise ValueError('keep data differs between stages')
    keep = registry.arrays(workers['control']['keep_data'], expected=workers['control']['keep_data_sha256'])
    kr, kg, kp = [], [], []
    for ep in keep_receipt['episodes']:
        require_train(ep)
        if not ep['success']:
            raise ValueError('unsuccessful keep trajectory admitted')
        n, contact = ep['steps'], ep['first_valid_contact']
        kr.extend([ep['ancestor']] * n); kg.extend([ep['group']] * n)
        kp.extend(['post_contact' if contact is not None and i > contact else 'pre_contact' for i in range(n)])
    if len(keep['actor_observation_before']) != 6289 or len(kr) != 6289:
        raise ValueError('fixed keep6289 provenance mismatch')
    # Targets are filled from independently reconstructed pi0, never a DEV label.
    corpora.append(Corpus('keep', keep['actor_observation_before'], np.zeros((6289, 4), np.float32),
                          keep['weights'], np.array(kr), np.array(kg), np.array(kp), np.array(kr)).validate())
    stable = []
    selected = diagnostics['control']['TRAIN_confirmation_update']
    for entry in rows['control'][str(selected)]['train']:
        rid = entry['root_id']
        records = [entry] + [next(x for x in rep if x['root_id'] == rid) for rep in diagnostics['control']['TRAIN_confirmations']]
        if all(x['combinations']['independent']['label'] == 1 for x in records):
            stable.append(records[-1])
    if len(stable) != 4 or selected != 1000:
        raise ValueError('old four stable roots/selected1000 contract changed')
    obs, targets, weights, roots, phases = [], [], [], [], []
    for entry in stable:
        require_train(entry)
        trace = saved_trace(registry, entry['combinations']['independent']['attempt'])
        n = len(trace['actor_observation_before'])
        obs.extend(trace['actor_observation_before']); targets.extend(trace['normalized_action_executed'])
        roots.extend([entry['root_id']] * n); phases.extend(contact_phase(trace)); weights.extend([1 / (4 * n)] * n)
    corpora.append(Corpus('old_stable_student', np.array(obs), np.array(targets), np.array(weights),
                          np.array(roots), np.repeat('independent_student', len(obs)), np.array(phases), np.array(roots)).validate())
    return runs, plans, rows, diagnostics, corpora, teachers


def load_models(runs, plans, registry):
    models, identities = {}, {}
    for stage, run in runs.items():
        for step in NODES:
            metadata = registry.json(run / 'B' / f'learner_update_{step:04d}.json')
            registry.register(metadata['path'], metadata['sha256'])
            path = registry.register(run / 'B' / f'update_{step:04d}.pkl')
            with path.open('rb') as stream:
                params = pickle.load(stream)
            hashes = [tree_sha(x) for x in params]
            expected = [metadata[k] for k in ('normalizer_sha256', 'actor_sha256', 'critic_sha256')]
            if hashes != expected:
                raise ValueError('inference/learner identity mismatch')
            pi0 = plans[stage]['models']['pi0']['policy']
            if hashes[0] != pi0['normalizer_sha256'] or hashes[2] != pi0['critic_sha256']:
                raise ValueError('frozen normalizer/critic mismatch')
            if step == 0 and hashes[1] != pi0['actor_sha256']:
                raise ValueError('zero checkpoint is not pi0')
            if not np.isfinite(params[0].std['state']).all() or np.any(params[0].std['state'] <= 0):
                raise ValueError('invalid normalizer standard deviation')
            key = f'{stage}_{step}'
            models[key] = params
            identities[key] = dict(path=str(path), file_sha256=registry.files[str(path)],
                                   actor_sha256=hashes[1], normalizer_sha256=hashes[0], critic_sha256=hashes[2])
    if identities['control_0']['actor_sha256'] != identities['intervention_0']['actor_sha256']:
        raise ValueError('pi0 initializers differ')
    return models, identities


def make_predictors():
    import jax
    import jax.numpy as jp
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks
    network = networks.make_ppo_networks({'state': 76, 'privileged_state': 106}, 4,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=(256, 256, 256), value_hidden_layer_sizes=(256, 256, 256),
        policy_obs_key='state', value_obs_key='privileged_state', distribution_type='tanh_normal')
    @jax.jit
    def production(norm, actor, obs):
        return network.parametric_action_distribution.mode(network.policy_network.apply(norm, actor, {'state': obs}))
    direct = jax.jit(independent_forward)
    def batched(params, obs, independent=False):
        chunks = []
        for start in range(0, len(obs), 256):
            piece = np.asarray(obs[start:start+256], dtype=np.float32)
            n = len(piece); padded = np.pad(piece, ((0, 256-n), (0, 0)))
            result = direct(params[1], params[0].mean['state'], params[0].std['state'], jp.asarray(padded)) if independent else production(params[0], params[1], jp.asarray(padded))
            chunks.append(np.asarray(result)[:n])
        return np.concatenate(chunks)
    return batched, production


def corpus_groups(corpus):
    yield 'all', np.ones(len(corpus.observations), bool)
    for field in ('roots', 'segments', 'phases', 'trajectories'):
        values = getattr(corpus, field)
        for value in dict.fromkeys(values):
            yield f'{field}/{value}', values == value
    if corpus.name != 'keep':
        for root in dict.fromkeys(corpus.roots):
            for segment in dict.fromkeys(corpus.segments):
                mask = (corpus.roots == root) & (corpus.segments == segment)
                if mask.any():
                    yield f'root_segment/{root}/{segment}', mask


def cross_inference(models, corpora, predict):
    results, arrays, direct_checks = {}, {}, []
    for key, params in models.items():
        print(f'T0 cross inference {key}', flush=True)
        results[key] = {}
        for c in corpora:
            prediction = predict(params, c.observations)
            reconstructed = predict(params, c.observations, independent=True)
            direct_checks.append(dict(name=f'{key}/{c.name}/independent_vs_Brax',
                                      maxabs=float(np.max(np.abs(prediction-reconstructed))), n=len(prediction)))
            arrays[f'{key}__{c.name}'] = prediction
            results[key][c.name] = {group: error_statistics(prediction[ids], c.actions[ids]) for group, ids in corpus_groups(c)}
    return results, arrays, direct_checks


def execution_chains(models, rows, diagnostics, registry, predict, nodes=NODES):
    records, pi0_errors = [], []
    for stage in ('control', 'intervention'):
        for step in nodes:
            entries = [(0, e) for e in rows[stage][str(step)]['train']]
            if step == 1000:
                entries += [(i+1, e) for i, repetition in enumerate(diagnostics[stage]['TRAIN_confirmations']) for e in repetition]
            for repeat, entry in entries:
                require_train(entry)
                attempt = entry['combinations']['independent']['attempt']
                trace = saved_trace(registry, attempt)
                params = models[f'{stage}_{step}']
                if attempt['actor_sha256'] != tree_sha(params[1]) or attempt['normalizer_sha256'] != tree_sha(params[0]):
                    raise ValueError('saved executed policy/normalizer mismatch')
                predicted = predict(params, trace['actor_observation_before'], independent=True)
                native = predict(params, trace['actor_observation_before'])
                stats = reconstruction_statistics(predicted, native, trace['normalized_action_executed'])
                item = dict(name=f'{stage}/{step}/{entry["root_id"]}/repeat{repeat}',
                            stage=stage, update=step, root_id=entry['root_id'], repeat=repeat,
                            trace=attempt['trace'], trace_lane=attempt['trace_lane'],
                            actor_sha256=attempt['actor_sha256'], normalizer_sha256=attempt['normalizer_sha256'],
                            label=attempt['label'], chain='raw saved pre-action obs -> normalize once -> Swish Actor -> tanh mode -> clip[-1,1] -> saved executed action', **stats)
                records.append(item)
                if step == 0:
                    pi0_errors.append(stats['maxabs'])
    return records, pi0_errors


def train_gradients(models, corpora, production):
    import jax
    import jax.numpy as jp
    from jax.flatten_util import ravel_pytree
    @jax.jit
    def value_gradient(actor, norm, obs, target, weights):
        def objective(a):
            prediction = production(norm, a, obs)
            return jp.sum(jp.mean((prediction - target) ** 2, axis=1) * weights)
        value, grad = jax.value_and_grad(objective)(actor)
        return value, ravel_pytree(grad)[0]
    output = {}
    for key, params in models.items():
        print(f'T0 fixed TRAIN gradients {key}', flush=True)
        vectors, stats = {}, {}
        for c in corpora:
            split = c.segments if c.name == 'keep' else c.roots
            total_gradient = None
            total_value = 0.
            for group in dict.fromkeys(split):
                ids = np.flatnonzero(split == group)
                weights = c.weights[ids].astype(np.float32)
                weights /= weights.sum()
                gradient, value = None, 0.
                for start in range(0, len(ids), 256):
                    chosen = ids[start:start+256]; n = len(chosen)
                    obs = np.pad(c.observations[chosen], ((0, 256-n), (0, 0)))
                    target = np.pad(c.actions[chosen], ((0, 256-n), (0, 0)))
                    w = np.pad(weights[start:start+256], (0, 256-n))
                    loss, grad = value_gradient(params[1], params[0], jp.asarray(obs), jp.asarray(target), jp.asarray(w))
                    g = np.asarray(grad, dtype=np.float64)
                    gradient = g if gradient is None else gradient + g
                    value += float(loss)
                group_key = f'{c.name}/{group}'
                vectors[group_key] = gradient
                stats[group_key] = dict(n=len(ids), weighted_mse=value, gradient_norm=float(np.linalg.norm(gradient)))
                mass = float(c.weights[ids].sum() / c.weights.sum())
                total_gradient = mass * gradient if total_gradient is None else total_gradient + mass * gradient
                total_value += mass * value
            vectors[c.name] = total_gradient
            stats[c.name] = dict(n=len(c.observations), weighted_mse=total_value,
                                 gradient_norm=float(np.linalg.norm(total_gradient)))
        names = list(vectors)
        pairs = [{'left': a, 'right': b, 'cosine': gradient_cosine(vectors[a], vectors[b])}
                 for i, a in enumerate(names) for b in names[i+1:]]
        output[key] = dict(groups=stats, pairwise_cosines=pairs,
                           semantics='fixed full TRAIN distribution gradients at frozen checkpoint; not actual optimizer minibatches; no update/clip/Adam applied')
    return output


def first_difference(a, b, tolerance):
    n = min(len(a), len(b))
    error = np.max(np.abs(a[:n] - b[:n]).reshape(n, -1), axis=1)
    ids = np.flatnonzero(error > tolerance)
    return dict(first_tick=int(ids[0]) if len(ids) else None, initial_maxabs=float(error[0]),
                compared_ticks=n, maxabs=float(error.max()))


def drift_statistics(a, b, quantity):
    units = {'qpos':'per component: translation m, quaternion unitless, joint rad',
             'qvel':'per component: translation m/s, angular/joint rad/s',
             'actor_observation_before':'per raw feature: mixed physical units and flags; not a physical vector norm',
             'normalized_action_executed':'normalized action, unitless'}
    if quantity not in units:
        raise ValueError('unknown descriptive drift quantity')
    return dict(**first_difference(a, b, 1e-6), quantity=quantity,
                threshold_per_component=1e-6, units=units[quantity], gate_used=False,
                interpretation='first any-component numerical difference only; not a physical significance threshold')


def plot_focus(out, rows, diagnostics, teachers, registry, models, predict):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = {}
    fig_xy, axes_xy = plt.subplots(1, 3, figsize=(15, 4.5))
    for rid, xy_ax in zip(FOCUS, axes_xy):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        teacher = teachers[rid]
        t_teacher = np.arange(len(teacher['qpos'])) * .02
        axes[0, 0].plot(*teacher['qpos'][:, :2].T, color='purple', label='G H16 + pi0 teacher')
        xy_ax.plot(*teacher['qpos'][:, :2].T, color='purple', label='G H16 + pi0 teacher')
        records = {}
        for stage, color in [('control', 'tab:blue'), ('intervention', 'tab:orange')]:
            e = next(x for x in rows[stage]['1000']['train'] if x['root_id'] == rid)
            entries = [e] + [next(x for x in rep if x['root_id'] == rid) for rep in diagnostics[stage]['TRAIN_confirmations']]
            traces = []
            for repeat, entry in enumerate(entries):
                attempt = entry['combinations']['independent']['attempt']
                z = saved_trace(registry, attempt); traces.append(z)
                t = np.arange(len(z['qpos'])) * .02
                label = f'{stage} r{repeat}: {attempt["label"]}'
                style = '-' if repeat == 0 else '--'
                for ax in [axes[0, 0], xy_ax]:
                    ax.plot(*z['qpos'][:, :2].T, style, color=color, alpha=1. if repeat == 0 else .5, label=label)
                    ax.scatter(*z['qpos'][-1, :2], color=color, marker='o' if attempt['label'] else 'x', s=20)
                axes[0, 1].plot(t, z['roll'], style, color=color, label=label)
                contact = np.flatnonzero(z['first_valid_contact'])
                if len(contact): axes[0, 1].axvline(t[contact[0]], color=color, alpha=.15)
            own = traces[0]; n = min(len(own['qpos']), len(teacher['qpos']))
            own_error = np.sqrt(np.mean((own['normalized_action_executed'][:n] - teacher['normalized_action_executed'][:n]) ** 2, axis=1))
            axes[1, 0].plot(np.arange(n)*.02, own_error, color=color, label=stage)
            teacher_prediction = predict(models[f'{stage}_1000'], teacher['actor_observation_before'])
            same_state_error = np.sqrt(np.mean((teacher_prediction - teacher['normalized_action_executed']) ** 2, axis=1))
            axes[1, 1].plot(t_teacher, same_state_error, color=color, label=stage)
            records[stage] = dict(labels=[x['combinations']['independent']['label'] for x in entries],
                teacher_same_state=error_statistics(teacher_prediction, teacher['normalized_action_executed']),
                closed_loop_prefix16_rmse=float(np.sqrt(np.mean((own['normalized_action_executed'][:16] - teacher['normalized_action_executed'][:16]) ** 2))),
                teacher_first_observation_difference=drift_statistics(own['actor_observation_before'], teacher['actor_observation_before'], 'actor_observation_before'),
                repeat_divergence=[{k:drift_statistics(own[k], z[k], k) for k in ('qpos','qvel','actor_observation_before','normalized_action_executed')} for z in traces[1:]])
        axes[0, 0].set(xlabel='World x (m)', ylabel='World y (m)', title='Actual XY; o success, x failure')
        axes[0, 0].set_aspect('equal', adjustable='datalim')
        xy_ax.set(xlabel='World x (m)', ylabel='World y (m)', title=rid)
        xy_ax.set_aspect('equal', adjustable='datalim')
        axes[0, 1].set(xlabel='Suffix time (s)', ylabel='Roll (rad)', title='All fixed repeats; vertical lines = first contact')
        axes[1, 0].set(xlabel='Suffix time (s)', ylabel='Normalized action RMSE', title='Own-state trajectories vs teacher at same tick (not same state)')
        axes[1, 1].set(xlabel='Teacher suffix time (s)', ylabel='Normalized action RMSE', title='Student on exact teacher observations (same state)')
        for ax in axes.flat: ax.legend(fontsize=7); ax.grid(alpha=.2)
        xy_ax.legend(fontsize=6)
        fig.suptitle(f'{rid}: frozen BC1000; TRAIN diagnostic, zero new physics')
        fig.tight_layout(); fig.savefig(out/f'root_{rid.rsplit("-",1)[-1]}_divergence.png', dpi=140); plt.close(fig)
        summary[rid] = records
    fig_xy.tight_layout(); fig_xy.savefig(out/'xy_focus.png', dpi=140); plt.close(fig_xy)
    return summary


def run(control, intervention, output):
    """Execute only saved-state CPU forward/backward computations, with fail-closed gate."""
    out = create_output(output, [control, intervention])
    started = time.time()
    declaration = dict(schema=SCHEMA, control=str(Path(control).resolve()), intervention=str(Path(intervention).resolve()),
        nodes=list(NODES), declared_corpora=['old_demo525', 'added_demo228', 'keep6289', 'old_four_stable_TRAIN_roots'],
        old_stable_representative='fixed second confirmation of selected control BC1000',
        tolerance_rule='max(1e-6, 10*pi0_reconstruction_maxabs)', calibration_scope='saved TRAIN pi0 execution actions only',
        descriptive_state_drift_threshold=1e-6, descriptive_state_drift_scope='each raw component in its native units; distinct from engineering action gate',
        inference='all declared observations, all eight stage/checkpoint pairs', gradient='fixed TRAIN per ancestor and keep group; no optimizer step',
        simulation_steps=0, supervised_updates=0, TEST_opened=False, started_unix=started)
    write_json(out/'declaration.json', declaration)
    try:
        import jax
        if jax.default_backend() != 'cpu' or any(d.platform != 'cpu' for d in jax.devices()):
            raise ValueError('T0 requires JAX_PLATFORMS=cpu')
        registry = SourceRegistry()
        registry.register(__file__)
        registry.register(Path(__file__).resolve().parents[2] / 'cli/run_retention_t0.py')
        runs, plans, rows, diagnostics, corpora, teachers = load_inputs(control, intervention, registry)
        models, identities = load_models(runs, plans, registry)
        predict, production = make_predictors()
        corpora[2].actions = predict(models['control_0'], corpora[2].observations, independent=True)
        calibration_chains, pi0_errors = execution_chains(models, rows, diagnostics, registry, predict, nodes=(0,))
        tolerance = calibration_tolerance(pi0_errors)
        write_json(out/'calibration.json', dict(rule=declaration['tolerance_rule'], tolerance=tolerance,
                   pi0_reconstruction_maxabs=max(pi0_errors), calibration_cases=len(pi0_errors), frozen_before_candidate_gate=True))
        candidate_chains, _ = execution_chains(models, rows, diagnostics, registry, predict, nodes=(100, 500, 1000))
        chains = calibration_chains + candidate_chains
        cross, _, independent_checks = cross_inference(models, corpora, predict)
        write_json(out/'cross_inference.json', cross)
        with (out/'cross_inference_overview.csv').open('w') as stream:
            writer = csv.writer(stream)
            writer.writerow(['model', 'corpus', 'group', 'n', 'mse', 'rmse', 'maxabs', *ACTION_ORDER])
            for model, by_corpus in cross.items():
                for corpus, groups in by_corpus.items():
                    for group, values in groups.items():
                        writer.writerow([model, corpus, group, values['n'], values['mse'], values['rmse'], values['maxabs'], *values['mse_by_action']])
        write_json(out/'execution_chain.json', dict(records=chains, independent_network_checks=independent_checks,
            action_order=ACTION_ORDER, tolerance=tolerance, normalization='(raw_obs - frozen_mean)/frozen_std exactly once; no observation clipping',
            action='tanh(first4 logits), then [-1,1] legal clipping', actuator_mapping_scope='normalized action chain only; no physics or actuator target recomputation'))
        execution_native_checks = [dict(name=r['name']+'/independent_vs_native_CPU',
                                        maxabs=r['independent_cpu_vs_native_maxabs']) for r in chains]
        gate = chain_gate(chains + independent_checks + execution_native_checks, tolerance)
        gradients = train_gradients(models, corpora, production)
        write_json(out/'TRAIN_gradients.json', gradients)
        focus = plot_focus(out, rows, diagnostics, teachers, registry, models, predict)
        write_json(out/'focus_diagnostics.json', focus)
        registry.verify_unchanged()
        corpus_info = {c.name:dict(n=len(c.observations), roots=list(dict.fromkeys(c.roots)),
                          source_weight_sum=float(c.weights.sum()), role='TRAIN') for c in corpora}
        input_identities = dict(control=str(runs['control']), intervention=str(runs['intervention']), models=identities,
            candidate_actor_sha256=identities['intervention_1000']['actor_sha256'],
            source_pi0_actor_sha256=identities['control_0']['actor_sha256'],
            normalizer_sha256=identities['control_0']['normalizer_sha256'],
            execution_code_revision=plans['intervention']['code_revision'])
        gate.update(input_identities=input_identities, corpus_counts={k:v['n'] for k,v in corpus_info.items()},
            backend='cpu', source_files_unchanged=True, TEST_opened=False,
            checked_unix=time.time(), wall_seconds=time.time()-started,
            acceptance_scope='implementation chain and artifact integrity only; no recovery/retention/capacity claim')
        gate['numerical_reconstruction'] = dict(
            pi0_independent_cpu_vs_saved_maxabs=max(pi0_errors),
            all_independent_cpu_vs_saved_maxabs=max(r['maxabs'] for r in chains),
            all_native_cpu_vs_saved_maxabs=max(r['native_cpu_vs_saved_maxabs'] for r in chains),
            all_independent_cpu_vs_native_maxabs=max(r['independent_cpu_vs_native_maxabs'] for r in chains),
            interpretation='independent-vs-native CPU separates reconstruction code error from the shared CPU-to-saved-execution discrepancy; exact backend kernel cause is not established',
            state_drift_threshold=1e-6, state_drift_not_used_for_action_gate=True)
        write_json(out/'input_identities.json', dict(**input_identities, sources=registry.files, corpora=corpus_info))
        write_json(out/'summary.json', dict(gate=gate, corpora=corpus_info, focus=focus,
                   limitations=['fixed TRAIN diagnostics are not independent tests', 'three rollouts are not independent training seeds',
                                'gradient conflict is not proof of insufficient capacity', 'same-tick own-state action differences are not same-state imitation errors',
                                'no simulator, policy/optimizer update, DEV training data, or TEST access']))
        write_index(out, gate, cross, focus)
        write_json(out/'T0_gate.json', gate)
        print(json.dumps({'output':str(out), 'phase':gate['phase'], 't1_allowed':gate['t1_allowed'], 'tolerance':tolerance}), flush=True)
        return gate
    except Exception as error:
        write_json(out/'T0_gate.json', dict(schema=SCHEMA, phase='blocked', t1_allowed=False,
            simulation_steps=0, supervised_updates=0, TEST_opened=False, blocking_failures=[dict(name=type(error).__name__, detail=str(error))]))
        raise


def write_index(out, gate, cross, focus):
    lines = ['# T0：保存TRAIN数据的零物理诊断', '', '![原始世界坐标XY与固定重复](xy_focus.png)', '',
             f'工程门槛：**{gate["phase"]}**；T1 allowed=`{gate["t1_allowed"]}`。冻结容差 `{gate["tolerance"]:.10g}`。', '',
             '本报告只做CPU推理与固定TRAIN梯度计算：新增物理0、参数更新0。通过只表示输入身份与执行动作链一致，不表示恢复能力提升。', '',
             '|模型|旧525 MSE|新增228 MSE|keep6289 MSE|旧稳定学生自访MSE|', '|---|---:|---:|---:|---:|']
    for key, corpora in cross.items():
        lines.append('|' + key + '|' + '|'.join(f'{corpora[c]["all"]["mse"]:.8g}' for c in ('old_demo','added_demo','keep','old_stable_student')) + '|')
    lines += ['', '旧稳定学生自访观测固定选择旧BC1000四个三次成功根的第二次确认轨迹；只作为TRAIN诊断参考，未加入学习。新增示范与原示范分开，避免变动分母掩盖误差。', '',
              '|根|旧BC1000三次|新BC1000三次|旧/新各自闭环prefix16 RMSE|', '|---|---|---|---|']
    for rid, pair in focus.items():
        lines.append(f'|{rid}|{pair["control"]["labels"]}|{pair["intervention"]["labels"]}|{pair["control"]["closed_loop_prefix16_rmse"]:.6f}/{pair["intervention"]["closed_loop_prefix16_rmse"]:.6f}|')
    lines += ['', '闭环轨迹同tick的动作差不是同状态误差。各根分歧图同时展示精确教师观测上的学生动作误差；接触及失败均保留真实终点。', '',
              *[f'![{rid} 分歧](root_{rid.rsplit("-",1)[-1]}_divergence.png)' for rid in FOCUS], '',
              '`TRAIN_gradients.json`给出固定完整TRAIN分组梯度范数及夹角；没有执行clip、Adam或参数更新，不能冒充实际mini-batch梯度或容量证明。', '',
              '机器入口：`T0_gate.json`。详细来源/身份：`input_identities.json`。动作链：`execution_chain.json`。共同观测分组误差：`cross_inference.json`及`cross_inference_overview.csv`。不复制模型、全部观测或物理轨迹；由锁定原始来源可重算。', '',
              '状态分歧按各原始分量1e-6描述，保留qpos/qvel/观测的原单位；它不采用动作重建验收容差，也不是物理显著性判据。', '',
              f'独立CPU→保存动作最大差{gate["numerical_reconstruction"]["all_independent_cpu_vs_saved_maxabs"]:.9g}，原生Brax CPU→保存动作最大差{gate["numerical_reconstruction"]["all_native_cpu_vs_saved_maxabs"]:.9g}；独立CPU→原生CPU最大差{gate["numerical_reconstruction"]["all_independent_cpu_vs_native_maxabs"]:.9g}。共同CPU与保存执行之间存在差异，不能据此精确归因某个GPU内核。冻结门槛公式保持不变。', '',
              '门槛失败阻断T1；不得调大已冻结容差、重试物理到成功或打开TEST。原始模型、数据和轨迹均未覆盖。']
    (out/'INDEX.md').write_text('\n'.join(lines) + '\n')
