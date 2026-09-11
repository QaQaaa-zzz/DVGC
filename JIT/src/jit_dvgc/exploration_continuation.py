"""Complete reached-state capture and budgeted frozen-bank suffix witnesses.

No simulator interaction occurs until ``evaluate``. Labels describe existence of
an unconflicted first-landing witness from the captured context under the locked
bank, with the bank's declared fresh-continuation counter semantics.
"""
from pathlib import Path
from types import SimpleNamespace
import json

import jax
import numpy as np

from .evidence_integrity import canonical_sha256
from .probe_bank import _file_sha
from .unified_envelope_snapshot import (
    UP_EVENT_FIELDS, DOWN_EVENT_FIELDS, capture_unified_envelope_snapshot,
    save_unified_envelope_snapshot, snapshot_context_sha256, physical_state_sha256,
)

INFO_FIELDS = (
    'last_action', 'rng', 'active_phase', 'start_phase', 'phase_transitioned',
    'episode_step', 'phase_episode_step', 'episode_return', 'reset_from_soft_tube',
    'source_tick', 'parent_group_index', 'tube_entry_index', 'tube_global_index',
    'expert_switching_used',
)


def snapshot_arrays(state):
    """JAX-pytree-compatible flat leaves for scan output; no invented history."""
    result = {'data/' + k: getattr(state.data, k) for k in ('qpos', 'qvel', 'ctrl')}
    result.update({'obs/state': state.obs['state'], 'info/done': state.done,
                   'history/frames': state.info['history'].frames,
                   'history/valid_count': state.info['history'].valid_count})
    result.update({'info/' + k: state.info[k] for k in INFO_FIELDS})
    result['info/rng'] = jax.random.key_data(state.info['rng'])
    for prefix, fields in (('up', UP_EVENT_FIELDS), ('down', DOWN_EVENT_FIELDS)):
        result.update({prefix + '/' + k: getattr(state.info[prefix + '_events'], k) for k in fields})
    return result


def snapshot_from_arrays(arrays, *, env, record, parent_trajectory, parent_state_sha256):
    """Capture a host-side single candidate through the canonical snapshot API."""
    info = {k: arrays['info/' + k] for k in INFO_FIELDS}
    info['history'] = SimpleNamespace(frames=arrays['history/frames'], valid_count=arrays['history/valid_count'])
    for prefix, fields in (('up', UP_EVENT_FIELDS), ('down', DOWN_EVENT_FIELDS)):
        info[prefix + '_events'] = SimpleNamespace(**{k: arrays[prefix + '/' + k] for k in fields})
    state = SimpleNamespace(data=SimpleNamespace(**{k: arrays['data/' + k] for k in ('qpos', 'qvel', 'ctrl')}),
                            obs={'state': arrays['obs/state']}, done=arrays['info/done'], info=info)
    return capture_unified_envelope_snapshot(state, env=env, parent_trajectory=parent_trajectory,
        parent_state_sha256=parent_state_sha256, config_sha256=record['formal_config_sha256'],
        policy_actor_sha256=record['actor_sha256'], policy_payload_sha256=record['payload_sha256'],
        policy_iteration=record['iteration'])


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


class FrozenSuffixEvaluator:
    """Serial lazy GPU evaluator, one attempt per candidate and bank member.

    Reserve a whole horizon before starting each attempt; charge actual dispatched
    steps, including a step that raises. A partial bank or engineering failure is
    unknown, never a negative. Existing output directories cannot be resumed.
    """
    def __init__(self, bank, order, horizon, output, budget):
        from .probe_bank import load_probe_bank
        self.bank_path = Path(bank)
        self.bank = load_probe_bank(self.bank_path)
        self.order = tuple(order)
        self.horizon, self.budget = horizon, budget
        if type(horizon) is not int or horizon <= 0 or type(budget) is not int or budget < 0:
            raise ValueError('invalid suffix horizon/budget')
        self.members = {m['name']: m for m in self.bank['members'] if 'evaluator' in m['roles']}
        if not self.order or len(set(self.order)) != len(self.order) or set(self.order) != set(self.members):
            raise ValueError('order must contain each declared evaluator exactly once')
        if horizon != self.bank['max_ticks']:
            raise ValueError('suffix horizon differs from locked bank')
        task = self.bank['task']
        if task['success_criterion'] != 'first_valid_landing' or task['continuation_start_semantics'] != 'fresh_continuation_v1':
            raise ValueError('unsupported bank endpoint/context semantics')
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.charged_interactions = 0
        self._runtimes, self._seen = {}, set()
        _write(self.output / 'declaration.json', dict(bank_sha256=self.bank['bank_sha256'],
            order=self.order, horizon=horizon, budget=budget, maximum_attempts_per_member=1,
            endpoint='unconflicted_first_valid_landing', continuation_start_semantics='fresh_continuation_v1',
            role='train', replay_gate=False))

    def _runtime(self, name):
        if name in self._runtimes:
            return self._runtimes[name]
        from .checkpoint import load_checkpoint
        from .handoff_bank import pytree_sha256
        from .ppo import make_checkpoint_policy
        from .unified_formal import build_unified_formal_environment
        from .unified_policy_freeze import load_frozen_unified_manifest
        from .unified_training import checkpoint_identity
        from .probe_bank import load_probe_bank
        if jax.default_backend() != 'gpu':
            raise RuntimeError('frozen suffix runtime requires genuine GPU execution')
        if load_probe_bank(self.bank_path) != self.bank:
            raise ValueError('locked bank drift')
        member = self.members[name]
        record = load_frozen_unified_manifest(Path(member['frozen_policy']))['policy']
        if record != member['policy']:
            raise ValueError('frozen evaluator identity drift')
        config, _, env = build_unified_formal_environment(Path(record['formal_config']))
        payload = load_checkpoint(Path(record['checkpoint']), expected=checkpoint_identity(config, env))
        for field, value in (('actor_sha256', payload.actor_params), ('normalizer_sha256', payload.observation_normalizer),
                             ('critic_sha256', payload.critic_params)):
            if pytree_sha256(value) != record[field]:
                raise ValueError('frozen evaluator payload drift: ' + field)
        if env._bundle.xml_sha256 != record['xml_sha256']:
            raise ValueError('evaluator model drift')
        if 'warp' in str(env._require_runtime_model().impl.value).lower():
            from .frontier_label_shard_runner import _build_memory_stable_step
            step = _build_memory_stable_step(env)
        else:
            step = jax.jit(env.step)
        runtime = env, jax.jit(make_checkpoint_policy(env, payload, deterministic=True)), step
        self._runtimes[name] = runtime
        return runtime

    def _attempt(self, snapshot, name, directory):
        from .unified_continuation_labels import fresh_unified_continuation_start, classify_first_valid_landing_outcome
        from .jump_evidence_runtime import view, action_for
        env, policy, step = self._runtime(name)
        state = fresh_unified_continuation_start(snapshot, env)
        frames, actions = [view(state)], []
        try:
            for tick in range(self.horizon):
                final = frames[-1]
                if final['done'] or final['down/valid_contact_seen']:
                    break
                action = action_for(policy, state, jax.random.fold_in(jax.random.PRNGKey(0), tick))
                actions.append(np.asarray(jax.device_get(action)).tolist())
                if self.charged_interactions >= self.budget:
                    raise RuntimeError('suffix interaction budget exhausted')
                self.charged_interactions += 1
                state = step(state, action)
                jax.block_until_ready(state)
                frames.append(view(state))
            final = frames[-1]
            valid, failure = bool(final['down/valid_contact_seen']), bool(final['physical_failure'])
            if valid and failure:
                return dict(label=None, outcome='simultaneous_landing_failure_unresolved')
            positive, outcome = classify_first_valid_landing_outcome(valid_contact_seen=valid,
                physical_failure_before_landing=failure, timeout=bool(final['timeout']), done=bool(final['done']),
                reached_rollout_horizon=len(actions) >= self.horizon)
            return dict(label=int(positive), outcome=outcome, end_flags={k: final[k] for k in
                ('done', 'physical_failure', 'timeout', 'end_code', 'success', 'down/valid_contact_seen')})
        finally:
            _write(directory / 'trace.json', {'frames': frames, 'actions': actions,
                'action_count_includes_dispatched_failure': True})

    def evaluate(self, snapshot):
        context = snapshot_context_sha256(snapshot)
        if context in self._seen:
            raise ValueError('candidate already attempted; retries are not allowed')
        if bool(snapshot.down_events['valid_contact_seen']):
            raise ValueError('candidate already landed before suffix evaluation')
        if snapshot.xml_sha256 != self.bank['task']['xml_sha256']:
            raise ValueError('candidate XML differs from locked bank')
        self._seen.add(context)
        directory = self.output / context
        directory.mkdir()
        save_unified_envelope_snapshot(directory / 'snapshot', snapshot)
        start_charge = self.charged_interactions
        labels = {name: None for name in self.order}
        receipts, witness = [], None
        for name in self.order:
            if self.budget - self.charged_interactions < self.horizon:
                break
            attempt_dir = directory / name
            attempt_dir.mkdir()
            before = self.charged_interactions
            receipt = dict(evaluator=name, actor_sha256=self.members[name]['policy']['actor_sha256'],
                payload_sha256=self.members[name]['policy']['payload_sha256'], snapshot_context_sha256=context,
                reserved_interactions=self.horizon, status='running')
            _write(attempt_dir / 'receipt.json', receipt)
            try:
                receipt.update(self._attempt(snapshot, name, attempt_dir), status='completed')
            except Exception as exc:
                receipt.update(label=None, status='engineering_error', error=f'{type(exc).__name__}: {exc}')
            receipt['charged_interactions'] = self.charged_interactions - before
            receipt['files'] = {p.name: _file_sha(p) for p in attempt_dir.iterdir() if p.name != 'receipt.json'}
            receipt['receipt_sha256'] = canonical_sha256(receipt)
            _write(attempt_dir / 'receipt.json', receipt)
            receipts.append(receipt)
            labels[name] = receipt['label']
            if receipt['label'] == 1:
                witness = name
                break
            if receipt['status'] == 'engineering_error':
                break
        result = dict(label=1 if witness else (0 if all(v == 0 for v in labels.values()) else None),
            witness=witness, labels=labels, attempts=receipts, snapshot_context_sha256=context,
            state_sha256=physical_state_sha256(snapshot), bank_sha256=self.bank['bank_sha256'],
            charged_interactions=self.charged_interactions - start_charge,
            total_charged_interactions=self.charged_interactions,
            snapshot_identity_sha256=_file_sha(directory / 'snapshot' / 'identity.json'))
        result['receipt_sha256'] = canonical_sha256(result)
        _write(directory / 'result.json', result)
        return result
