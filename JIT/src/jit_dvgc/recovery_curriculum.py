"""Real TRAIN teacher-prefix states and root-balanced recovery resets.

Preparation is read-only. Construction requires an explicit live environment and
replays actual normalized executed actions; observations/history are never built
from qpos or copied frames. Stage k is the state AFTER k transitions, hence the
next action sees teacher observation_before[k].
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

STAGES = (16, 12, 8, 4, 0)
SCHEMA = 'jit_recovery_curriculum_v1'


def _read(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def remaining_horizon(horizon, original_elapsed, prefix_steps=0):
    remaining = int(horizon) - int(original_elapsed) - int(prefix_steps)
    if min(int(original_elapsed), int(prefix_steps)) < 0 or remaining <= 0:
        raise ValueError('curriculum snapshot has exhausted original horizon')
    return remaining


def validate_teacher_entries(manifest):
    rows = manifest['entries']
    if not rows or len({r['root_id'] for r in rows}) != len(rows):
        raise ValueError('empty or duplicate teacher root')
    for row in rows:
        if (row.get('role', '').lower() != 'train' or row.get('inherited_split') != 'generator_train'
                or not row.get('complete') or not row.get('full_success')
                or row.get('teacher_status') != 'verified_solution'
                or not row.get('teacher_repeat_labels') or any(x != 1 for x in row['teacher_repeat_labels'])):
            raise ValueError('curriculum requires complete verified TRAIN teachers')
    return rows


def curriculum_probabilities(entries, weights):
    """Stage mass, then equal roots within stage, then equal states per root."""
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(STAGES),) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError('invalid curriculum weights')
    available = np.asarray([any(e['stage'] == s for e in entries) for s in STAGES])
    mass = weights * available
    if mass.sum() <= 0:
        raise ValueError('weighted stages have no real snapshots')
    mass /= mass.sum()
    p = np.zeros(len(entries))
    for i, stage in enumerate(STAGES):
        roots = {e['root_id'] for e in entries if e['stage'] == stage}
        for root in roots:
            indices = [j for j, e in enumerate(entries) if e['stage'] == stage and e['root_id'] == root]
            p[indices] = mass[i] / len(roots) / len(indices)
    return p


def resolve_teacher_sources(teacher_manifest, source_rows=None):
    """Resolve original TRAIN roots via verification receipts, never root suffix guesses."""
    manifest = _read(teacher_manifest)
    entries = validate_teacher_entries(manifest)
    result = []
    cache = {}
    for entry in entries:
        for field in ('source_trace', 'verification_receipt'):
            lock = entry[field]
            if _sha(lock['path']) != lock['sha256']:
                raise ValueError('teacher source identity changed: ' + field)
        adapter = _read(entry['verification_receipt']['path'])
        origin = adapter['original_source']
        if source_rows is None:
            if origin['path'] not in cache:
                if _sha(origin['path']) != origin['sha256']:
                    raise ValueError('original teacher results identity changed')
                cache[origin['path']] = _read(origin['path'])
            candidates = cache[origin['path']]
        else:
            candidates = _read(source_rows) if isinstance(source_rows, (str, Path)) else source_rows
        matches = [r for r in candidates if r['root_id'] == entry['root_id']]
        if len(matches) != 1 or matches[0].get('role', '').upper() != 'TRAIN':
            raise ValueError('missing/ambiguous/nonTRAIN original root')
        root = dict(matches[0])
        if root['snapshot_context_sha256'] != entry['root_context_sha256']:
            raise ValueError('root context identity mismatch')
        attempt = adapter['attempts'][-1]  # canonical adapter declares final original confirmation
        if attempt['label'] != 1 or _sha(attempt['trace']) != attempt['trace_sha256']:
            raise ValueError('invalid original confirmation trace')
        with np.load(attempt['trace']) as trace:
            lane = int(attempt['trace_lane'])
            snapshot_time = float(trace['time_before'][0, lane])
        if 'snapshot_time' in root and not np.isclose(root['snapshot_time'], snapshot_time, atol=1e-6):
            raise ValueError('original root recorded time mismatch')
        root['snapshot_time'] = snapshot_time
        result.append((entry, root))
    return result


def build_recovery_curriculum(teacher_manifest, out, *, env, source_rows=None,
                              horizon=400, observation_atol=1e-4, stages=STAGES,
                              replay_capacity=1, canonical_replay_required=True):
    """Explicit physical replay; returns immutable inventory with actual cost.

    No fake snapshot is produced if replay diverges from the saved teacher.
    Tolerance is declared in the inventory; exact solver replay is not claimed.
    """
    import jax
    import jax.numpy as jp
    from .continuation.device_rollout import stack_worlds,prepare_parallel_worlds,take_world
    from .unified_envelope_snapshot import (load_unified_envelope_snapshot,
        restore_unified_envelope_snapshot, snapshot_context_sha256,
        capture_unified_envelope_snapshot, save_unified_envelope_snapshot)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    entries, exclusions, replay_deviations = [], [], []
    transitions = 0
    if type(replay_capacity) is not int or replay_capacity<1:raise ValueError('positive replay capacity required')
    step_fn = jax.jit(env.step) if replay_capacity==1 else jax.jit(jax.vmap(env.step))
    for entry, root in resolve_teacher_sources(teacher_manifest, source_rows):
        snapshot = load_unified_envelope_snapshot(Path(root['snapshot']))
        if snapshot_context_sha256(snapshot) != root['snapshot_context_sha256']:
            raise ValueError('original root snapshot changed')
        state = restore_unified_envelope_snapshot(snapshot, env)
        state = state.replace(data=state.data.replace(time=jp.asarray(root['snapshot_time'], jp.float32)))
        if replay_capacity>1:
            batched=prepare_parallel_worlds(stack_worlds([state]*replay_capacity),env,replay_capacity)
            state=take_world(batched,0)
        elapsed = snapshot.episode_step
        with np.load(entry['source_trace']['path']) as trace:
            obs = trace['actor_observation_before'].copy()
            actions = trace['normalized_action_executed'].copy()
            valid = trace['valid_mask'].astype(bool)
        n = len(actions)
        if not valid.all() or obs.shape != (n, 76) or actions.shape != (n, 4):
            raise ValueError('teacher trace contains padding or wrong actor/action space')
        if not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1.000001:
            raise ValueError('teacher executed normalized action is invalid')
        for k in range(min(max(stages), n - 1) + 1):
            # This check proves capture-after-tick alignment against saved preobs.
            actual=np.asarray(state.obs['state'])
            if not np.isfinite(actual).all():raise FloatingPointError('nonfinite real teacher replay')
            error=float(np.max(np.abs(actual-obs[k])))
            replay_deviations.append(dict(root_id=entry['root_id'],step=k,observation_max_abs=error))
            # Initial context/history must match exactly. Subsequent contexts
            # are genuine new simulator states under the saved executed G
            # actions, not copied historical observations. A caller may record
            # numerical replay differences instead of claiming equivalence.
            if k==0 or canonical_replay_required:
                np.testing.assert_allclose(actual, obs[k], atol=observation_atol, rtol=0,
                                           err_msg=f'teacher replay diverged: {entry["root_id"]} step {k}')
            terminal = bool(np.asarray(state.done)) or bool(np.asarray(state.info['success']))
            if k in stages:
                if terminal:
                    exclusions.append(dict(root_id=entry['root_id'], stage=k, reason='terminal_or_success'))
                else:
                    remaining = remaining_horizon(horizon, elapsed, k)
                    tail = entry['source_tail_identity']
                    captured = capture_unified_envelope_snapshot(state, env=env,
                        parent_trajectory=entry['source_trace']['path'],
                        parent_state_sha256=entry['source_trace']['sha256'],
                        config_sha256=snapshot.config_sha256,
                        policy_actor_sha256=tail['actor_sha256'], policy_payload_sha256=tail['payload_sha256'],
                        policy_iteration=tail['iteration'], capture_recovery_context=True)
                    dest = out / 'snapshots' / entry['root_id'] / f'stage_{k:02d}'
                    save_unified_envelope_snapshot(dest, captured)
                    entries.append(dict(root_id=entry['root_id'], ancestor_root_id=entry['root_episode_id'],
                        role='TRAIN', stage=k, snapshot=str(dest.resolve()),
                        snapshot_context_sha256=snapshot_context_sha256(captured),
                        snapshot_time=captured.snapshot_time, original_elapsed=elapsed,
                        control_step=elapsed+k, remaining_horizon=remaining,
                        trajectory_sha256=entry['trajectory_sha256'], source_trace=entry['source_trace'],
                        model_sha256=entry['model_sha256'], protocol_sha256=entry['protocol_sha256'],
                        expert_switching_used=captured.expert_switching_used,
                        control_source='original_root' if k == 0 else 'G_bridge_prefix',
                        observation_alignment='next_pre_action', terminal=False, success=False))
            if terminal or k == min(max(stages), n - 1):
                break
            if replay_capacity==1:
                state = step_fn(state, jp.asarray(actions[k]))
            else:
                batched=step_fn(batched,jp.broadcast_to(jp.asarray(actions[k]),(replay_capacity,4)))
                state=take_world(batched,0)
            state = state.replace(info={**state.info, 'expert_switching_used': jp.asarray(True)})
            jax.block_until_ready(state)
            transitions += 1
            (out / 'capture_status.json').write_text(json.dumps(dict(phase='capturing', active_interactions=transitions, root_id=entry['root_id'], control_step=k+1))+'\n')
    inventory = dict(schema=SCHEMA, role='TRAIN', stages=list(STAGES), entries=entries,
                     excluded=exclusions, teacher_manifest=str(Path(teacher_manifest).resolve()),
                     teacher_manifest_sha256=_sha(teacher_manifest), horizon=horizon,
                     observation_atol=observation_atol, active_interactions=transitions,
                     replay_capacity=replay_capacity,charged_interactions=transitions*replay_capacity,
                     canonical_replay_required=canonical_replay_required,
                     canonical_observation_deviations=replay_deviations,
                     snapshot_semantics='actual_new_physical_replay_of_saved_executed_G_actions',
                     root_ids=[e['root_id'] for e, _ in resolve_teacher_sources(teacher_manifest, source_rows)])
    (out / 'curriculum.json').write_text(json.dumps(inventory, indent=2, allow_nan=False)+'\n')
    (out / 'capture_status.json').write_text(json.dumps(dict(phase='completed', active_interactions=transitions, snapshot_count=len(entries)))+'\n')
    return inventory


class RecoveryCurriculumReset:
    """JIT-safe stage weights live in state.info; roots are equal within stage.

    Owns autoreset: on terminal preserves terminal reward/done/flags but installs
    the next complete physical context. Install with Brax auto_reset=False.
    """
    def __init__(self, env, inventory, weights=(1, 0, 0, 0, 0)):
        import jax
        import jax.numpy as jp
        from .continuation.device_rollout import stack_worlds
        from .unified_envelope_snapshot import load_unified_envelope_snapshot, restore_unified_envelope_snapshot, snapshot_context_sha256
        if inventory.get('schema') != SCHEMA or inventory.get('role') != 'TRAIN':
            raise ValueError('invalid recovery curriculum inventory')
        self.env, self.original_step = env, env.step
        self.entries = inventory['entries']
        if not self.entries or any(e.get('role') != 'TRAIN' or e.get('terminal') or e.get('success') for e in self.entries):
            raise ValueError('empty, terminal, or nonTRAIN curriculum')
        self.weights = jp.asarray(weights, jp.float32)
        curriculum_probabilities(self.entries, weights)
        states = []
        self.stage_basis = jp.asarray(np.stack([curriculum_probabilities(self.entries, np.eye(len(STAGES))[i])
            if any(e['stage'] == STAGES[i] for e in self.entries) else np.zeros(len(self.entries))
            for i in range(len(STAGES))]), jp.float32)
        for index, entry in enumerate(self.entries):
            snap = load_unified_envelope_snapshot(Path(entry['snapshot']))
            if snapshot_context_sha256(snap) != entry['snapshot_context_sha256']:
                raise ValueError('curriculum context changed')
            if getattr(snap, 'snapshot_time', None) != entry['snapshot_time']:
                raise ValueError('curriculum requires V2 explicit time')
            state = restore_unified_envelope_snapshot(snap, env)
            np.testing.assert_array_equal(state.obs['state'], snap.observation)
            np.testing.assert_array_equal(state.info['history'].frames, snap.observation_fifo)
            info = {**state.info, 'recovery_curriculum_weights': self.weights,
                    'recovery_curriculum_index': jp.asarray(index, jp.int32),
                    'recovery_reset_rng': jax.random.key(0),
                    'recovery_curriculum_stage': jp.asarray(entry['stage'], jp.int32),
                    'recovery_remaining_horizon': jp.asarray(entry['remaining_horizon'], jp.int32)}
            info['rng'] = jax.random.wrap_key_data(jax.random.key_data(info['rng']))
            states.append(state.replace(info=info))
        self.pool = stack_worlds(states)

    @staticmethod
    def set_state_weights(state, weights):
        import jax.numpy as jp
        return state.replace(info={**state.info, 'recovery_curriculum_weights': jp.asarray(weights, jp.float32)})

    def reset_with_weights(self, key, weights):
        import jax
        import jax.numpy as jp
        from .continuation.device_rollout import take_world
        mass = jp.asarray(weights) @ self.stage_basis
        key = jax.random.wrap_key_data(jax.random.key_data(key))
        selection_key, next_key = jax.random.split(key)
        index = jax.random.categorical(selection_key, jp.where(mass > 0, jp.log(mass), -jp.inf))
        state = take_world(self.pool, index)
        state = state.replace(info={**state.info, 'recovery_reset_rng':next_key})
        return self.set_state_weights(state, weights)

    def reset(self, key):
        return self.reset_with_weights(key, self.weights)

    def reset_from_state(self, state, key):
        return self.reset_with_weights(key, state.info['recovery_curriculum_weights'])

    def step(self, state, action):
        import jax
        import jax.numpy as jp
        advanced = self.original_step(state, action)
        left = state.info['recovery_remaining_horizon'] - 1
        expired = left <= 0
        advanced = advanced.replace(done=jp.maximum(advanced.done, expired.astype(advanced.done.dtype)),
            info={**advanced.info, 'expert_switching_used': state.info.get('expert_switching_used', jp.asarray(False)),
                  'recovery_curriculum_weights': state.info['recovery_curriculum_weights'],
                  'recovery_curriculum_index': state.info['recovery_curriculum_index'],
                  'recovery_curriculum_stage': state.info['recovery_curriculum_stage'],
                  'recovery_remaining_horizon': left,
                  'truncated': advanced.info['truncated'] | expired,
                  'timeout': advanced.info['timeout'] | expired,
                  'time_out': jp.maximum(advanced.info['time_out'], expired.astype(jp.float32))})
        def terminal_reset(current):
            # Selection RNG is separate from preserved snapshot simulator RNG.
            key = current.info['recovery_reset_rng']
            fresh = self.reset_from_state(current, key)
            terminal_fields = ('terminated','truncated','time_out','end_code','success','physical_failure','roll_limit',
                               'pitch_limit','jump_zone_missed','stuck','yaw_limit','timeout')
            info = {**current.info, **fresh.info, **{k:current.info[k] for k in terminal_fields}}
            return fresh.replace(reward=current.reward, done=current.done, metrics=current.metrics, info=info)
        # DataWarp contact arrays are shared scratch, not per-world leaves.
        # A cond over the complete state under vmap broadcasts those arrays and
        # leaks BatchTracers through DataWarp's custom non-vmap registration.
        # Select only logical state; retain the physics step's shared scratch.
        from .continuation.device_rollout import _shared_warp
        fresh = terminal_reset(advanced)
        return jax.tree_util.tree_map_with_path(
            lambda path, new, old: old if _shared_warp(path) else
                jp.where(advanced.done.astype(bool), new, old), fresh, advanced)

    def install(self):
        self.env.reset, self.env.step = self.reset, self.step
        return self.env


def wrap_recovery_for_training(env, episode_length=400, action_repeat=1, randomization_fn=None):
    """Vectorization + PPO metrics, using curriculum's full-context autoreset.

    Never gives a late saved state a fresh episode_length. The native environment
    and recovery_remaining_horizon enforce the original absolute end. Repeated
    actions/domain randomization would change the pinned replay contract and are
    deliberately rejected here.
    """
    import jax.numpy as jp
    from brax.envs.wrappers.training import VmapWrapper
    from brax.envs.base import Wrapper
    if action_repeat != 1 or randomization_fn is not None:
        raise ValueError('recovery wrapper preserves action_repeat=1 and original physics')
    reset_owner = getattr(env.reset, '__self__', None)
    if not isinstance(reset_owner, RecoveryCurriculumReset):
        raise ValueError('install RecoveryCurriculumReset before training wrapper')
    if any(e['control_step'] + e['remaining_horizon'] != episode_length for e in reset_owner.entries):
        raise ValueError('wrapper horizon differs from curriculum absolute horizon')

    class RecoveryEpisodeMetrics(Wrapper):
        def reset(self, rng):
            state = self.env.reset(rng)
            zero = jp.zeros_like(state.done)
            metrics = {k:jp.zeros_like(v) for k,v in state.metrics.items()}
            metrics.update(sum_reward=zero, length=zero)
            return state.replace(info={**state.info, 'steps':state.info['episode_step'],
                'truncation':zero, 'episode_done':zero, 'episode_metrics':metrics})

        def step(self, state, action):
            previous_done = state.info['episode_done']
            clean = state.replace(done=jp.zeros_like(state.done))
            advanced = self.env.step(clean, action)
            metrics = {k:state.info['episode_metrics'][k]*(1-previous_done)+v
                       for k,v in advanced.metrics.items()}
            metrics['sum_reward'] = state.info['episode_metrics']['sum_reward']*(1-previous_done)+advanced.reward
            metrics['length'] = state.info['episode_metrics']['length']*(1-previous_done)+1
            steps = jp.where(previous_done.astype(bool), state.info['episode_step'], state.info['steps'])+1
            return advanced.replace(info={**advanced.info, 'steps':steps,
                'truncation':advanced.info['truncated'].astype(advanced.done.dtype),
                'episode_done':advanced.done, 'episode_metrics':metrics})

    return RecoveryEpisodeMetrics(VmapWrapper(env))
