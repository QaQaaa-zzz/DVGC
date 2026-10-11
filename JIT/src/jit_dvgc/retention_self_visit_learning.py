"""File-only preparation of verified learner-state BC demonstrations.

No policy, optimizer, simulator, generator, or service is created here. Original
demonstrations stay first and unchanged; only their declared sampling mass is
redistributed within an ancestor that gains a verified trajectory.
"""
from pathlib import Path
import numpy as np

from .retention_repair import read, write, sha
from .generative_bridge.data import validate_trace


class _Sources:
    def __init__(self):
        self.locks = {}

    def lock(self, path, expected=None):
        path = str(Path(path).resolve())
        actual = self.locks.get(path)
        if actual is None:
            actual = sha(path)
            self.locks[path] = actual
        if expected is not None and actual != expected:
            raise ValueError('source hash mismatch: ' + path)
        return path

    def json(self, path):
        return read(self.lock(path))

    def finish(self):
        for path, expected in self.locks.items():
            if sha(path) != expected:
                raise ValueError('source hash changed during preparation: ' + path)


def _stable_failures(parent, ancestors, actor, normalizer, sources):
    completed = sources.json(parent / 'B_completed.json')
    repeated = sources.json(parent / 'selected_diagnostics.json')
    nodes = sources.json(parent / 'B_checkpoint_results.json')
    if (completed.get('phase') != 'completed' or completed.get('selected_update') != 1000
            or repeated.get('selected_update') != 1000
            or repeated.get('TRAIN_confirmation_update') != 1000
            or len(repeated.get('TRAIN_confirmations', [])) != 2
            or nodes['1000']['actor_sha256'] != actor):
        raise ValueError('complete fixed BC1000 TRAIN repetitions required')
    labels = {rid: [] for rid in ancestors}
    for rows in [nodes['1000']['train'], *repeated['TRAIN_confirmations']]:
        if len(rows) != 8 or {r['root_id'] for r in rows} != set(ancestors):
            raise ValueError('historical TRAIN coverage differs')
        for row in rows:
            independent = row['combinations']['independent']
            attempt = independent['attempt']
            if (row['role'] != 'TRAIN' or independent['label'] not in (0, 1)
                    or attempt['label'] != independent['label']
                    or attempt['actor_sha256'] != actor
                    or attempt['normalizer_sha256'] != normalizer):
                raise ValueError('historical TRAIN learner/label identity mismatch')
            labels[row['root_id']].append(independent['label'])
    return {rid for rid, values in labels.items() if values == [0, 0, 0]}, labels


def _old_demo(path, ancestors, sources):
    manifest = sources.json(path)
    if manifest.get('role') != 'TRAIN' or manifest.get('schema') != 'jit_bridge_demo_v1_2':
        raise ValueError('original TRAIN demo manifest required')
    data_path = sources.lock(manifest['path'], manifest['sha256'])
    with np.load(data_path, allow_pickle=False) as data:
        obs = data['observations'].copy()
        actions = data['actions'].copy()
        weights = data['weights'].copy()
    roots = np.asarray(manifest['sample_roots'])
    origins = np.asarray(manifest['sample_origins'])
    n = manifest['count']
    if (obs.shape != (n, 76) or actions.shape != (n, 4) or weights.shape != (n,)
            or roots.shape != (n,) or origins.shape != (n,) or set(roots) != set(ancestors)
            or not np.isfinite(obs).all() or not np.isfinite(actions).all()
            or np.any(np.abs(actions) > 1) or not np.isfinite(weights).all()):
        raise ValueError('invalid original demo dimensions/values/provenance')
    expected = np.zeros(n, np.float64)
    for rid in ancestors:
        indices = np.flatnonzero(roots == rid)
        local = origins[indices]
        if (len(indices) <= 16 or not np.all(local[:16] == 'bridge_prefix')
                or not np.all(local[16:] == 'source_tail')):
            raise ValueError('original H16 prefix/tail provenance changed')
        for segment in ('bridge_prefix', 'source_tail'):
            mask = (roots == rid) & (origins == segment)
            expected[mask] = .5 / 8 / mask.sum()
    if not np.allclose(weights, expected, rtol=1e-7, atol=1e-12):
        raise ValueError('original ancestor/prefix-tail weights differ')
    return manifest, obs, actions, roots.tolist(), origins.tolist()


def _winner_trace(query, folder, policy, actor, sources):
    step = query['step']
    window = 16 - step
    capture = query['capture']
    lock = query['winner_lock']
    candidate = lock['selected_candidate_id']
    if (query['source_labels'] != [0, 0, 0] or query['winner_labels'] != [1, 1, 1]
            or query['remaining_horizon'] != window
            or type(candidate) is not int or not 1 <= candidate < 17
            or lock.get('fixed_before_repeats') is not True or lock.get('layout') != 17
            or lock['remaining_horizon'] != window
            or capture.get('capture_status') != 'captured' or capture['capture_step'] != step
            or capture.get('role') != 'TRAIN' or capture.get('data_role') != 'train'
            or capture['parent_root_id'] != query['root_id']
            or capture['root_episode_id'] != query['root_id'] or capture['actor_sha256'] != actor
            or lock['snapshot_context_sha256'] != capture['snapshot_context_sha256']):
        raise ValueError('verified TRAIN query identity/remaining window mismatch')
    if sources.json(folder / f'j{step:02d}_winner_lock.json') != lock:
        raise ValueError('fixed winner lock differs from query')
    sources.lock(query['proposals'], lock['proposals_sha256'])
    with np.load(query['proposals'], allow_pickle=False) as data:
        proposals = data['actions'].copy()
        if (proposals.shape != (17, 16, 4) or not np.isfinite(proposals).all()
                or np.any(np.abs(proposals) > 1)
                or not np.array_equal(data['source_only'], np.arange(17) == 0)
                or data['root_contexts'].tolist() != [capture['snapshot_context_sha256']] * 17):
            raise ValueError('proposal layout/context drift')
    matrix = query['all_candidate_labels']
    if (np.asarray(matrix).shape != (3, 17) or not np.isin(matrix, [0, 1]).all()
            or [row[0] for row in matrix] != [0, 0, 0]
            or [row[candidate] for row in matrix] != [1, 1, 1]
            or len(query['winner_attempts']) != 3 or len(query['source_attempts']) != 3):
        raise ValueError('complete same-layout fixed-winner repetitions required')
    for repeat, suffix in enumerate(('search', 'repeat1', 'repeat2')):
        rows = sources.json(folder / f'j{step:02d}_{suffix}/results.json')
        winner = query['winner_attempts'][repeat]
        source = query['source_attempts'][repeat]
        if (len(rows) != 17 or [r['candidate_id'] for r in rows] != list(range(17))
                or [r['label'] for r in rows] != matrix[repeat]
                or rows[candidate]['attempts'] != [winner] or rows[0]['attempts'] != [source]):
            raise ValueError('raw repetition results disagree with fixed winner')
        for attempt, lane, label, source_only in ((winner, candidate, 1, False), (source, 0, 0, True)):
            if (attempt['label'] != label or attempt['trace_lane'] != lane
                    or attempt['actor_sha256'] != policy['actor_sha256']
                    or attempt['normalizer_sha256'] != policy['normalizer_sha256']
                    or attempt['model_sha256'] != policy['xml_sha256']
                    or attempt['snapshot_context_sha256'] != capture['snapshot_context_sha256']
                    or attempt.get('source_only') is not source_only
                    or attempt.get('handoff_tick') != window or attempt.get('reset_at_handoff') is not False
                    or attempt.get('recording_schema') != 'jit_bridge_trace_v1'
                    or attempt.get('controller_kind') != 'composite_teacher'):
                raise ValueError('teacher tail/normalizer/handoff/lane identity mismatch')
            sources.lock(attempt['trace'], attempt['trace_sha256'])
        if winner['outcome'] != 'stable_forward_recovery':
            raise ValueError('full task success is required')
    # Fixed representative: second confirmation, never search-best among replays.
    attempt = query['winner_attempts'][2]
    with np.load(attempt['trace'], allow_pickle=False) as raw:
        mask = np.asarray(raw['mask'][:, candidate])
        if not np.isin(mask, [0, 1]).all():
            raise ValueError('nonbinary real-step mask')
        indices = np.flatnonzero(mask)
        n = len(indices)
        if not n or not np.array_equal(indices, np.arange(n)) or n != attempt['steps']:
            raise ValueError('noncontiguous or incomplete real-step mask')
        keys = ('actor_observation_before', 'actor_observation_after', 'normalized_action_executed',
                'done', 'success', 'physical_failure', 'timeout', 'phase_before', 'action_origin_code', 'time_before')
        arrays = {key: raw[key][indices, candidate].copy() for key in keys}
    codes = arrays.pop('action_origin_code')
    if not np.array_equal(codes, np.where(np.arange(n) < window, 1, 2)):
        raise ValueError('actual bridge/source origin differs from remaining window')
    arrays['failure'] = arrays.pop('physical_failure')
    arrays['valid_mask'] = np.ones(n, bool)
    arrays['action_origin'] = np.where(codes == 1, 'bridge_prefix', 'source_tail')
    validate_trace({'arrays': arrays})
    if not arrays['success'][-1] or np.any(arrays['failure']) or np.any(arrays['timeout']):
        raise ValueError('teacher trajectory lacks unambiguous complete success')
    if not np.array_equal(arrays['normalized_action_executed'][:min(window, n)], proposals[candidate, :min(window, n)]):
        raise ValueError('executed bridge actions differ from locked proposal')
    sources.lock(capture['trace'], capture['trace_sha256'])
    with np.load(capture['trace'], allow_pickle=False) as tape:
        lane = capture['capture_lane']
        if (not bool(tape['mask'][step - 1, lane])
                or not np.array_equal(arrays['actor_observation_before'][0], tape['actor_observation_after'][step - 1, lane])):
            raise ValueError('teacher did not start at real learner post-step observation')
    if not np.isfinite(arrays['time_before']).all() or not np.isclose(arrays['time_before'][0], capture['snapshot_time'], rtol=0, atol=1e-7):
        raise ValueError('teacher start time differs from captured state')
    return arrays, attempt


def prepare_demo(parent_coverage_root, diagnostic_root, output):
    """Validate complete evidence, then write one immutable offline BC dataset."""
    parent = Path(parent_coverage_root).resolve()
    diagnostic = Path(diagnostic_root).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('new output directory required; no overwrite')
    sources = _Sources()
    parent_plan = sources.json(parent / 'plan.json')
    plan = sources.json(diagnostic / 'plan.json')
    status = sources.json(diagnostic / 'status.json')
    if (sources.json(parent / 'status.json').get('phase') != 'completed'
            or status.get('phase') != 'completed' or status.get('completed_roots') != 8
            or status.get('evaluation_status') != 'complete'
            or plan.get('schema') != 'jit_retention_self_visit_v1'
            or Path(plan['parent']).resolve() != parent or plan.get('training_updates') != 0
            or plan.get('data_role') != 'TRAIN' or plan['query_steps'] != [4, 8, 12]):
        raise ValueError('complete TRAIN-only diagnostic required before preparing data')
    original = sources.json(parent_plan['train_roots'])
    declared = sources.json(plan['train_roots'])
    ancestors = [entry['root']['root_id'] for entry in original]
    if (len(ancestors) != 8 or len(set(ancestors)) != 8 or original != declared
            or any(e['root'].get('role') != 'TRAIN' or e['root'].get('data_role') != 'train'
                   or e['root']['root_episode_id'] != e['root']['root_id'] for e in original)):
        raise ValueError('eight original TRAIN ancestors and roles required')
    policy = parent_plan['models']['pi0']['policy']
    receipt = sources.json(parent / 'B/learner_update_1000.json')
    if (plan['model']['policy'] != policy or plan['student_actor_sha256'] != receipt['actor_sha256']
            or receipt['normalizer_sha256'] != policy['normalizer_sha256']
            or receipt['critic_sha256'] != policy['critic_sha256']
            or plan['student']['normalizer_sha256'] != policy['normalizer_sha256']
            or plan['student']['source_actor_sha256'] != policy['actor_sha256']
            or Path(plan['student']['path']).resolve() != parent / 'B/update_1000.pkl'):
        raise ValueError('fixed source/student identity mismatch')
    sources.lock(plan['student']['path'], plan['student']['sha256'])
    stable, historical_labels = _stable_failures(parent, ancestors, receipt['actor_sha256'], policy['normalizer_sha256'], sources)
    queries = []
    for ordinal, rid in enumerate(ancestors):
        local = sources.json(diagnostic / f'root_{ordinal:02d}/queries.json')
        if len(local) != 3 or {(q['root_id'], q['step']) for q in local} != {(rid, j) for j in (4, 8, 12)}:
            raise ValueError('complete unique 24-slot diagnostic coverage required')
        if any(q.get('role') != 'TRAIN' for q in local):
            raise ValueError('DEV cannot enter TRAIN data')
        queries.extend(local)
    summary = sources.json(diagnostic / 'report/summary.json')
    if (summary.get('query_count') != 24 or summary.get('coverage', {}).get('complete') is not True
            or summary.get('evaluation_status') != 'complete'
            or sources.json(diagnostic / 'report/query_matrix.json') != queries):
        raise ValueError('completed report and raw query matrix disagree')
    old, obs, actions, roots, origins = _old_demo(parent_plan['demo_manifest'], ancestors, sources)
    old_count = len(obs)
    provenance = []
    for rid in ancestors:
        for local, index in enumerate(np.flatnonzero(np.asarray(roots) == rid)):
            provenance.append((int(index), dict(ancestor=rid, trajectory_id='original:' + rid, source_kind='original',
                local_transition=local, capture_step=None, segment=origins[index], source_data=old['path'])))
    provenance = [row for _, row in sorted(provenance)]
    observations = [obs]
    targets = [actions]
    new_entries = []
    admission = []
    for query in queries:
        rid = query['root_id']
        reason = ('ancestor_not_stably_failed' if rid not in stable else
                  'current_student_succeeded' if query['student_full_label'] == 1 else
                  'query_not_verified_incremental' if query['classification'] != 'G_incremental_verified' else None)
        admitted = dict(root_id=rid, step=query['step'], admitted=reason is None, reason=reason)
        admission.append(admitted)
        if reason is not None:
            continue
        if query['student_full_label'] != 0:
            raise ValueError('unknown student outcome cannot be a failed TRAIN root')
        folder = diagnostic / f'root_{ancestors.index(rid):02d}'
        arrays, attempt = _winner_trace(query, folder, policy, receipt['actor_sha256'], sources)
        n = len(arrays['normalized_action_executed'])
        tid = rid + f':self_visit_{query["step"]}'
        entry = dict(ancestor=rid, trajectory_id=tid, capture_step=query['step'], remaining_horizon=16-query['step'],
                     count=n, representative='fixed_second_confirmation', **attempt)
        new_entries.append(entry)
        observations.append(arrays['actor_observation_before'])
        targets.append(arrays['normalized_action_executed'])
        roots.extend([rid] * n)
        origins.extend(arrays['action_origin'].tolist())
        provenance.extend(dict(ancestor=rid, trajectory_id=tid, source_kind='self_visit', capture_step=query['step'],
            local_transition=t, segment=str(arrays['action_origin'][t]), trace=attempt['trace'], trace_lane=attempt['trace_lane']) for t in range(n))
    if not new_entries:
        raise ValueError('no verified new demonstration for a stable failed ancestor')
    weights = np.zeros(len(roots), np.float64)
    trajectory_ids = np.asarray([row['trajectory_id'] for row in provenance])
    root_ids = np.asarray(roots)
    segment_ids = np.asarray(origins)
    for rid in ancestors:
        trajectories = np.unique(trajectory_ids[root_ids == rid])
        for tid in trajectories:
            segments = np.unique(segment_ids[trajectory_ids == tid])
            for segment in segments:
                mask = (trajectory_ids == tid) & (segment_ids == segment)
                weights[mask] = 1 / (8 * len(trajectories) * len(segments) * mask.sum())
    sources.finish()
    output.mkdir(parents=True, exist_ok=False)
    path = output / 'data.npz'
    np.savez_compressed(path, observations=np.concatenate(observations), actions=np.concatenate(targets), weights=weights)
    write(output / 'frame_provenance.json', provenance)
    write(output / 'admission.json', dict(historical_TRAIN_labels=historical_labels, stable_failed_ancestors=sorted(stable), queries=admission))
    write(output / 'source_lock.json', dict(schema='jit_self_visit_demo_sources_v1', locks=sources.locks, training_updates=0, physical_interactions=0))
    manifest = dict(schema='jit_bridge_demo_v1_2', role='TRAIN', count=len(roots), path=str(path), sha256=sha(path),
        sample_roots=roots, sample_origins=origins, old_count=old_count, new_count=len(roots)-old_count,
        eligible_query_count=len(new_entries), new_entries=new_entries, original_manifest=str(Path(parent_plan['demo_manifest']).resolve()),
        sampling='eight original ancestors equally; trajectories equally within ancestor; real prefix/tail segments equally; real frames equally',
        source_normalizer_sha256=policy['normalizer_sha256'], source_actor_sha256=policy['actor_sha256'],
        collection_student_actor_sha256=receipt['actor_sha256'], parent_coverage_root=str(parent), diagnostic_root=str(diagnostic),
        frame_provenance=dict(path=str(output/'frame_provenance.json'),sha256=sha(output/'frame_provenance.json')),
        source_lock=dict(path=str(output/'source_lock.json'),sha256=sha(output/'source_lock.json')),
        training_updates=0, physical_interactions=0, keep_unchanged=True, loss_unchanged=True)
    write(output / 'manifest.json', manifest)
    return output / 'manifest.json'
