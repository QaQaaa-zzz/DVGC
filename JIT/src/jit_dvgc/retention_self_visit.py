"""Finite TRAIN-only learner-state queries; no optimization or policy adoption."""
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from .retention_repair import read, write, sha
from .retention_repair_execution import PY

SCHEMA = 'jit_retention_self_visit_v1'
GRANT_SCHEMA = 'jit_retention_self_visit_execution_v1'
STEPS = (4, 8, 12)


def budget(prior):
    result = dict(capture=8 * 17 * 400, queries=24 * 3 * 17 * 400,
                  prior_charged=int(prior), wall_seconds=10800, hard_cap=2000000)
    result['total'] = result['capture'] + result['queries']
    result['cumulative_maximum'] = prior + result['total']
    if prior < 0 or result['cumulative_maximum'] > result['hard_cap']:
        raise ValueError('cumulative charged budget exceeded')
    return result


def remaining_window(step):
    if type(step) is not int or step not in STEPS:
        raise ValueError('query must be at completed TRAIN transition 4, 8 or 12')
    return 16 - step


def validate_roots(rows):
    if len(rows) != 8 or len({r['root']['root_id'] for r in rows}) != 8:
        raise ValueError('exactly eight distinct original TRAIN roots required')
    for entry in rows:
        if entry['root']['role'] != 'TRAIN' or entry['root']['data_role'] != 'train':
            raise ValueError('DEV/TEST cannot enter learner-state queries')
        if type(entry['lane']) is not int or not 1 <= entry['lane'] < 17:
            raise ValueError('retain original selected teacher lane in 17-world capture')


def classify(source, winner):
    if len(source) != 3 or any(x not in (0, 1) for x in source):
        raise ValueError('three complete source labels required')
    if winner is not None and (len(winner) != 3 or any(x not in (0, 1) for x in winner)):
        raise ValueError('three fixed-winner labels required')
    if source == [1, 1, 1]:
        return 'source_recoverable'
    if any(source):
        return 'source_ambiguous'
    if winner is None:
        return 'searched_no_solution'
    return 'G_incremental_verified' if winner == [1, 1, 1] else 'winner_ambiguous'


def classify_query(source, winner, all_labels):
    if len(all_labels) != 3 or any(len(row) != 17 or any(x not in (0, 1) for x in row) for row in all_labels):
        raise ValueError('all original-layout search/replay labels must be retained')
    result = classify(source, winner)
    if result == 'searched_no_solution' and any(any(row[1:]) for row in all_labels[1:]):
        return 'search_no_winner_replay_positive'
    return result


def coverage(rows, root_ids):
    expected = {(rid, step) for rid in root_ids for step in STEPS}
    actual = [(r['root_id'], r['step']) for r in rows]
    if len(set(actual)) != len(actual) or not set(actual) <= expected:
        raise ValueError('duplicate or unexpected query identity')
    return dict(complete=set(actual) == expected, observed=len(actual), expected=len(expected),
                missing=[dict(root_id=rid, step=step) for rid, step in sorted(expected - set(actual))])


def require_grant(plan, grant):
    g = read(grant)
    if (g.get('schema') != GRANT_SCHEMA or g.get('execute') is not True
            or g.get('plan_sha256') != sha(plan) or not g.get('source')):
        raise ValueError('explicit hash-bound new-stage execution grant required')


def prepare(parent, cost_parent, output, repository, port=6030):
    parent = Path(parent).resolve()
    cost_parent = Path(cost_parent).resolve()
    p = read(parent / 'plan.json')
    cp = read(cost_parent / 'plan.json')
    for folder in (parent, cost_parent):
        if read(folder / 'status.json')['phase'] != 'completed':
            raise ValueError('both historical stages must be complete')
    rows = read(p['train_roots'])
    validate_roots(rows)
    prior = cp['budget']['prior_charged'] + sum(c['charged'] for c in read(cost_parent / 'costs.json'))
    root = Path(output).resolve()
    code = Path(repository).resolve()
    root.mkdir(parents=True, exist_ok=False)
    locks = {}
    # Old executable files are historical identities, not mutable execution inputs.
    for f, expected in p['locks'].items():
        if Path(f).suffix == '.py':
            continue
        if sha(f) != expected:
            raise ValueError('historical input changed: ' + f)
        locks[f] = expected
    paths = [parent / 'plan.json', cost_parent / 'plan.json', cost_parent / 'costs.json',
             Path(p['train_roots']), parent / 'B/update_1000.pkl',
             parent / 'B/learner_update_1000.json']
    g = rows[0]['teacher']['G']
    if any(r['teacher']['G'] != g for r in rows):
        raise ValueError('all TRAIN roots must use the same fixed generator')
    if (sha(g['manifest']) != g['manifest_sha256']
            or sha(Path(g['manifest']).parent / 'state.msgpack') != read(g['manifest'])['state_sha256']):
        raise ValueError('historical generator identity drift')
    paths += [Path(g['manifest']), Path(g['manifest']).parent / 'state.msgpack']
    for path in paths:
        locks[str(path)] = sha(path)
    student_receipt = read(parent / 'B/learner_update_1000.json')
    source = p['models']['pi0']['policy']
    if any(student_receipt[k] != source[k] for k in ('normalizer_sha256', 'critic_sha256')):
        raise ValueError('student frozen normalizer/critic identity differs')
    student = dict(path=str(parent / 'B/update_1000.pkl'),
                   sha256=sha(parent / 'B/update_1000.pkl'),
                   source_actor_sha256=source['actor_sha256'],
                   normalizer_sha256=source['normalizer_sha256'])
    write(root / 'train_roots.json', rows)
    locks[str(root / 'train_roots.json')] = sha(root / 'train_roots.json')
    code_locks = {str(f): sha(f) for folder in ('JIT/src/jit_dvgc', 'JIT/cli')
                  for f in (code / folder).rglob('*.py')}
    locks.update(code_locks)
    plan = dict(schema=SCHEMA, stage='TRAIN_self_visit', output=str(root), code=str(code),
                code_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=code, text=True).strip(),
                parent=str(parent), cost_parent=str(cost_parent),
                train_roots=str(root / 'train_roots.json'), model=p['models']['pi0'],
                student=student, student_actor_sha256=student_receipt['actor_sha256'],
                G=g, budget=budget(prior), locks=locks, query_steps=list(STEPS),
                proposal_seed=1011273101, tensorboard_port=port, training_updates=0,
                E_updates=0, G_updates=0, PPO_updates=0, automatic_next_stage=False,
                published_policy=None, data_role='TRAIN', initialization='frozen_inference_only',
                snapshot_semantics='canonical full controller/event context; not bitwise simulator restoration',
                acceptance='classify all 24 states; no policy adoption or student-learning claim')
    write(root / 'plan.json', plan)
    write(root / 'code_lock.json', code_locks)
    write(root / 'budget_dry_run.json', plan['budget'])
    write(root / 'status.json', dict(phase='prepared', training_updates=0, evaluation_status='not_started'))
    return root / 'plan.json'


def audit(path):
    p = read(path)
    if (p['schema'] != SCHEMA or p['query_steps'] != list(STEPS)
            or any(p[k] != 0 for k in ('training_updates', 'E_updates', 'G_updates', 'PPO_updates'))
            or p['automatic_next_stage'] or p['data_role'] != 'TRAIN'):
        raise ValueError('self-visit method contract drift')
    if p['budget'] != budget(p['budget']['prior_charged']):
        raise ValueError('budget drift')
    validate_roots(read(p['train_roots']))
    for file, expected in p['locks'].items():
        if sha(file) != expected:
            raise ValueError('locked input/code drift: ' + file)
    return dict(phase='passed', locked_files=len(p['locks']), budget=p['budget'])


def evaluation_spec(p, row, rowsfile):
    return dict(bank=p['model']['bank'], proposer=p['model']['proposer'],
                order=[p['model']['proposer']], horizon=400, budget=6800,
                candidates=str(rowsfile), full_matrix=True, record_actor_preobservations=True,
                record_retention_diagnostics=True, reward_mode='original_all_phases',
                success_criterion='stable_forward_recovery', preserve_snapshot_episode_context=True,
                suffix_rng_count=17, suffix_rng_indices=list(range(17)))


def scalar_writer(folder):
    from tensorboard.summary.writer.event_file_writer import EventFileWriter
    return EventFileWriter(str(folder))


def scalar(writer, tag, value, step):
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    writer.add_event(Event(wall_time=time.time(), step=step,
                           summary=Summary(value=[Summary.Value(tag=tag, simple_value=float(value))])))
    writer.flush()


def worker(request_path):
    request = read(request_path)
    if sha(request['plan']) != request['plan_sha256'] or request.get('execute') is not True:
        raise ValueError('worker requires frozen parent request')
    p = read(request['plan'])
    for key in ('student',):
        if sha(p[key]['path']) != p[key]['sha256']:
            raise ValueError('student payload drift')
    ordinal = request['ordinal']
    entry = read(p['train_roots'])[ordinal]
    row = entry['root']
    root = Path(request['output'])
    root.mkdir(exist_ok=False)
    from .pulse_exploration_runtime import evaluate
    costs = []
    writer = scalar_writer(Path(p['output']) / 'tensorboard')

    def status(phase, **extra):
        write(root / 'status.json', dict(phase=phase, root_id=row['root_id'], pid=os.getpid(),
              charged_interactions=sum(c['charged_interactions'] for c in costs),
              active_interactions=sum(c.get('active_interactions', 0) for c in costs),
              training_updates=0, updated_unix=time.time(), **extra))

    def evaluate_one(name, spec, session=None):
        cost = dict(name=name, charged_interactions=6800, reservation=6800, phase='running')
        costs.append(cost)
        write(root / 'costs.json', costs)
        status('running', stage=name)
        write(root / (name + '_spec.json'), spec)
        out = root / name
        evaluate(spec, out, session=session)
        receipt = read(out / 'status.json')
        results = read(out / 'results.json')
        if (receipt['phase'] != 'completed' or receipt['charged_interactions'] > 6800
                or len(results) != 17 or any(r['label'] not in (0, 1) or len(r['attempts']) != 1 for r in results)):
            raise ValueError('incomplete evaluation cannot become a negative label')
        cost.update({k: receipt[k] for k in ('phase', 'charged_interactions', 'active_interactions', 'padding_interactions')})
        write(root / 'costs.json', costs)
        return results

    try:
        capture_rows = [{**row, 'index': i, 'candidate_id': i} for i in range(17)]
        rowfile = root / 'capture_rows.json'
        write(rowfile, capture_rows)
        spec = evaluation_spec(p, row, rowfile)
        spec.update(warmup_initializer=p['student'], capture_snapshot_steps=list(STEPS),
                    capture_snapshot_lanes=[entry['lane']])
        student_results = evaluate_one('student_capture', spec)
        attempt = student_results[entry['lane']]['attempts'][0]
        if attempt['actor_sha256'] != p['student_actor_sha256']:
            raise ValueError('capture did not run the declared BC1000 Actor')
        scalar(writer, 'diagnostic/student_reward_per_transition', attempt['task_return'] / attempt['steps'], ordinal)
        scalar(writer, 'diagnostic/student_full_success', student_results[entry['lane']]['label'], ordinal)
        captures = read(root / 'student_capture/snapshots.json')
        from .generative_bridge.worker import generator_template
        from .generative_bridge.diffusion import restore_state, ddim_sample
        from .generative_bridge.proposals import stable_seed
        from .generative_bridge.teacher import select_teacher
        from .generative_bridge.production import teacher_candidate
        from .generative_bridge.teacher_runtime import TeacherEvaluationSession
        from .unified_envelope_snapshot import load_unified_envelope_snapshot
        import jax
        import jax.numpy as jp
        net, template, identity = generator_template(p['model']['frozen_policy'], p['proposal_seed'])
        if identity != read(p['G']['manifest'])['identity']:
            raise ValueError('generator reference identity differs')
        state = restore_state(Path(p['G']['manifest']).parent, template, identity)
        generation = jax.jit(lambda obs, noise: ddim_sample(lambda x, o, k: net.apply(state['ema'], x, o, k), obs, noise))
        session = TeacherEvaluationSession()
        queries = []
        for j in STEPS:
            capture = next(c for c in captures if c['capture_step'] == j and c['capture_lane'] == entry['lane'])
            result = dict(root_id=row['root_id'], role='TRAIN', step=j, remaining_horizon=remaining_window(j),
                          student_full_label=student_results[entry['lane']]['label'],
                          student_attempt=attempt, capture=capture)
            if not capture.get('snapshot'):
                result.update(classification='unavailable_terminal_or_horizon', exclusion=capture)
                queries.append(result)
                write(root / 'queries.json', queries)
                continue
            query_row = {**row, **{k: capture[k] for k in ('snapshot', 'snapshot_context_sha256', 'state_sha256', 'snapshot_time', 'snapshot_control_step')},
                         'root_id': row['root_id'] + f'/student_j{j:02d}', 'original_root_id': row['root_id'],
                         'prefix_terminal': False}
            snap = load_unified_envelope_snapshot(Path(query_row['snapshot']))
            noise = np.stack([np.random.default_rng(stable_seed(p['proposal_seed'], query_row['root_id'], cid)).normal(size=(16, 4)) for cid in range(1, 17)]).astype('f4')
            obs = jp.asarray(np.repeat(np.asarray(snap.observation)[None], 16, axis=0))
            generated = np.asarray(generation((obs - state['normalizer']['mean']) / state['normalizer']['std'], jp.asarray(noise)), np.float32)
            actions = np.concatenate((np.zeros((1, 16, 4), np.float32), generated))
            if actions.shape != (17, 16, 4) or not np.isfinite(actions).all():
                raise ValueError('invalid generator proposals')
            ap = root / f'j{j:02d}_proposals.npz'
            np.savez_compressed(ap, actions=actions, source_only=np.arange(17) == 0,
                               root_contexts=np.array([query_row['snapshot_context_sha256']] * 17), G_noise=noise)
            rf = root / f'j{j:02d}_rows.json'
            write(rf, [{**query_row, 'index': cid, 'candidate_id': cid} for cid in range(17)])
            spec = evaluation_spec(p, query_row, rf)
            spec.update(bridge_action_plan=dict(path=str(ap), sha256=sha(ap)), bridge_execution_horizon=remaining_window(j))
            search = evaluate_one(f'j{j:02d}_search', spec, session)
            candidates = [teacher_candidate(r, 'diffusion', r['candidate_id'], snap.last_action) for r in search[1:]]
            winner = select_teacher(candidates)
            cid = winner['candidate_id'] if winner else None
            lock = dict(selected_candidate_id=cid, proposals_sha256=sha(ap), layout=17,
                        fixed_before_repeats=True, remaining_horizon=remaining_window(j),
                        snapshot_context_sha256=query_row['snapshot_context_sha256'])
            write(root / f'j{j:02d}_winner_lock.json', lock)
            source_labels = [search[0]['label']]
            winner_labels = [search[cid]['label']] if cid is not None else None
            winner_attempts = [search[cid]['attempts'][0]] if cid is not None else []
            source_attempts = [search[0]['attempts'][0]]
            all_labels = [[r['label'] for r in search]]
            for repeat in (1, 2):
                repeated = evaluate_one(f'j{j:02d}_repeat{repeat}', spec, session)
                all_labels.append([r['label'] for r in repeated])
                source_labels.append(repeated[0]['label'])
                source_attempts.append(repeated[0]['attempts'][0])
                if cid is not None:
                    winner_labels.append(repeated[cid]['label'])
                    winner_attempts.append(repeated[cid]['attempts'][0])
            result.update(classification=classify_query(source_labels, winner_labels, all_labels), source_labels=source_labels,
                          winner_labels=winner_labels, winner_lock=lock, source_attempts=source_attempts,
                          winner_attempts=winner_attempts, proposals=str(ap),
                          all_candidate_labels=all_labels,
                          search_successful_candidates=[r['candidate_id'] for r in search[1:] if r['label'] == 1])
            queries.append(result)
            write(root / 'queries.json', queries)
            scalar(writer, 'diagnostic/G_incremental_verified', result['classification'] == 'G_incremental_verified', ordinal * 3 + STEPS.index(j))
            scalar(writer, 'diagnostic/source_recoverable', result['classification'] == 'source_recoverable', ordinal * 3 + STEPS.index(j))
        session.close()
        status('completed', completed_queries=len(queries), full_student_label=student_results[entry['lane']]['label'])
    except BaseException as exc:
        status('failed', error=repr(exc))
        raise
    finally:
        writer.close()


def run(path, authorization):
    require_grant(path, authorization)
    audit(path)
    p = read(path)
    root = Path(p['output'])
    if read(root / 'status.json')['phase'] != 'prepared':
        raise ValueError('no implicit retry/resume')
    start = time.time()
    env = dict(os.environ, PYTHONPATH=str(Path(p['code']) / 'JIT/src'), JAX_PLATFORMS='cuda,cpu',
               XLA_PYTHON_CLIENT_PREALLOCATE='false', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               JIT_AUTO_PUBLISH='0', JAX_COMPILATION_CACHE_DIR=str(root / 'jax_cache'))
    costs = []
    completed = 0
    child = None

    def status(phase, **extra):
        write(root / 'status.json', dict(phase=phase, stage='TRAIN_self_visit', pid=os.getpid(),
              started_unix=start, updated_unix=time.time(), completed_roots=completed, declared_roots=8,
              charged_interactions=sum(c['charged'] for c in costs), maximum_planned_interactions=p['budget']['total'],
              prior_charged=p['budget']['prior_charged'], training_updates=0, E_updates=0, G_updates=0,
              evaluation_status='complete' if phase == 'completed' else 'partial', adoption_status='not_applicable',
              wall_seconds=time.time() - start, **extra))
    status('running', active='launch')
    write(root / 'ACTIVE_RUN.json', dict(execution=str(root / 'status.json'), lineage=str(root / 'status.json')))
    with (root / 'watcher.log').open('x') as log:
        watcher = subprocess.Popen([PY, str(Path(p['code']) / 'JIT/cli/watch_run_errors.py'), '--active-run',
            str(root / 'ACTIVE_RUN.json'), '--state-dir', str(root / 'notifications')], env=env, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
    with (root / 'tensorboard.log').open('x') as log:
        tb = subprocess.Popen([PY, '-m', 'tensorboard.main', '--logdir', str(root / 'tensorboard'),
            '--port', str(p['tensorboard_port']), '--host', '127.0.0.1', '--reload_interval', '1'],
            env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(root / 'launch.json', dict(pid=os.getpid(), watcher_pid=watcher.pid, tensorboard_pid=tb.pid,
          tensorboard_url=f"http://localhost:{p['tensorboard_port']}", started_unix=start,
          plan=str(path), plan_sha256=sha(path), authorization=str(authorization), command='run-self-visit'))
    try:
        for ordinal in range(8):
            remaining = p['budget']['wall_seconds'] - (time.time() - start)
            if remaining <= 0:
                raise TimeoutError('independent three-hour stage expired')
            maximum = 17 * 400 + 3 * 3 * 17 * 400
            if sum(c['charged'] for c in costs) + maximum > p['budget']['total']:
                raise ValueError('reservation exceeds declared finite budget')
            name = f'root_{ordinal:02d}'
            cost = dict(name=name, charged=maximum, reservation=maximum, phase='running')
            costs.append(cost)
            write(root / 'costs.json', costs)
            req = root / (name + '_request.json')
            write(req, dict(plan=str(path), plan_sha256=sha(path), ordinal=ordinal,
                           output=str(root / name), execute=True))
            with (root / (name + '.log')).open('x') as log:
                child = subprocess.Popen([PY, str(Path(p['code']) / 'JIT/cli/run_retention_next.py'),
                    'worker-self-visit', '--request', str(req)], env=env, cwd=p['code'], stdout=log, stderr=subprocess.STDOUT)
                status('running', active=name, child_pid=child.pid)
                try:
                    rc = child.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    raise TimeoutError('worker exceeded remaining stage wall budget')
            if rc:
                raise RuntimeError(f'{name} exit {rc}; reservation retained; no automatic retry')
            receipt = read(root / name / 'status.json')
            if (receipt['phase'] != 'completed' or receipt['charged_interactions'] > maximum
                    or receipt.get('completed_queries') != 3):
                raise ValueError('incomplete root receipt')
            cost.update(phase='completed', charged=receipt['charged_interactions'], active=receipt['active_interactions'])
            write(root / 'costs.json', costs)
            completed += 1
            status('running', active=name)
        audit(path)
        report(path)
        if not read(root / 'report/summary.json')['coverage']['complete']:
            raise ValueError('cannot complete a partial query matrix')
        status('completed', active='reported', result='diagnostic only; no automatic training')
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate()
        status('failed', error=repr(exc))
        raise


def report(path):
    p = read(path)
    root = Path(p['output'])
    rows = [r for folder in sorted(root.glob('root_[0-9][0-9]'))
            if (folder / 'queries.json').exists() for r in read(folder / 'queries.json')]
    out = root / 'report'
    out.mkdir(exist_ok=True)
    from collections import Counter
    matrix_coverage = coverage(rows, [r['root']['root_id'] for r in read(p['train_roots'])])
    summary = dict(query_count=len(rows), declared_queries=24, counts=dict(Counter(r['classification'] for r in rows)),
                   coverage=matrix_coverage, evaluation_status='complete' if matrix_coverage['complete'] else 'partial',
                   completed_roots=len({r['root_id'] for r in rows}), training_updates=0,
                   identity=dict(student=p['student_actor_sha256'], pi0=p['model']['policy']['actor_sha256'], G=p['G']),
                   scope='TRAIN diagnosis; not student improvement, final TEST, diffusion superiority or sim2sim evidence',
                   costs=read(root / 'costs.json'))
    write(out / 'summary.json', summary)
    write(out / 'query_matrix.json', rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lines = ['# Student self-visited TRAIN state diagnosis', '',
             'Actual XY trajectories below stop at recorded endpoints. These are branched continuations, not one deployable Actor.', '']
    for ordinal, rid in enumerate(dict.fromkeys(r['root_id'] for r in rows)):
        selected = [r for r in rows if r['root_id'] == rid]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), squeeze=False)
        for ax, q in zip(axes[0], selected):
            attempts = [('BC1000 full', q['student_attempt'])]
            if q.get('source_attempts'):
                attempts.append(('pi0 from student state', q['source_attempts'][0]))
            if q.get('winner_attempts'):
                attempts.append(('G remaining + pi0', q['winner_attempts'][0]))
            for label, attempt in attempts:
                with np.load(attempt['trace']) as data:
                    lane = attempt['trace_lane']
                    mask = data['mask'][:, lane]
                    xy = data['qpos'][mask, lane, :2]
                ax.plot(xy[:, 0], xy[:, 1], label=label)
                ax.scatter(*xy[-1], marker='o' if attempt['label'] else 'x', s=25)
            ax.set(title=f"j={q['step']}, H={q['remaining_horizon']}\n{q['classification']}", xlabel='world X (m)', ylabel='world Y (m)')
            ax.set_aspect('equal', adjustable='datalim')
            ax.legend(fontsize=7)
            ax.grid(alpha=.2)
        fig.suptitle(rid + ' / fixed BC1000 / fixed pi0 and G73 / TRAIN')
        fig.tight_layout()
        filename = f'root_{ordinal:02d}_xy.png'
        fig.savefig(out / filename, dpi=140)
        plt.close(fig)
        lines += [f'![{rid}]({filename})', '']
    lines += ['| TRAIN root | step | remaining H | student full | source labels | fixed G winner labels | result |',
              '|---|---:|---:|---:|---|---|---|']
    for r in rows:
        lines.append(f"| {r['root_id']} | {r['step']} | {r['remaining_horizon']} | {r['student_full_label']} | {r.get('source_labels', 'NA')} | {r.get('winner_labels', 'NA')} | {r['classification']} |")
    lines += ['', ('All 8 original TRAIN roots are retained, including successful students and unavailable terminal states.'
                   if matrix_coverage['complete'] else f"PARTIAL: {matrix_coverage['observed']}/24 queries available; missing queries are not failures."),
              'A verified query is a composite teacher witness. It does not establish that a trained student absorbed it.',
              'Canonical state restoration retains recorded controller/event context and time; bitwise simulator restoration is not claimed.',
              'No retuning, additional search after failed repeats, DEV-to-TRAIN transfer, optimization, or automatic continuation.',
              '', '[Machine-readable summary](summary.json) / [all query evidence](query_matrix.json)', '']
    (out / 'INDEX.md').write_text('\n'.join(lines))
    return out
