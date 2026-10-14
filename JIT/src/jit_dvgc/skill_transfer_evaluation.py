"""Independent recovery Actor evaluation on immutable TRAIN roots.

Preparation reads/locks identities only; evaluation dispatches existing bounded
Evaluator workers. Three repeats measure numerical label stability, never three
training seeds. No G actions or source-tail/prefix switching enter execution.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from .recovery_curriculum import resolve_teacher_sources, remaining_horizon
from .retention_consolidation_evaluation import stable_repeat_label

# Canonical bank explicitly supplied by the task owner, not discovered by naming.
REFERENCE_BANK = '/home/qy/DVGC/JIT/runs/monitoring/bridge_pi0_hard_initial1000_20261010/bank.json'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def _descriptor(value):
    if isinstance(value, (str, Path)):
        value = dict(path=str(Path(value).resolve()), sha256=_sha(value))
    if not isinstance(value, dict) or not value.get('path') or not value.get('sha256'):
        raise ValueError('explicit frozen parent file descriptor required')
    if _sha(value['path']) != value['sha256']:
        raise ValueError('frozen parent descriptor identity changed')
    return value


def prepare_recovery_panel(output, parent_descriptor, teacher_manifest_path,
                           source_rows_path, *, roots=None, repeats=3, bank_path=REFERENCE_BANK):
    """Source audit and panel construction; zero simulator/reset/rollout calls."""
    from .probe_bank import load_probe_bank, lock_probe_bank
    from .unified_envelope_snapshot import load_unified_envelope_snapshot, snapshot_context_sha256
    if type(repeats) is not int or repeats < 1:
        raise ValueError('positive predeclared repeat count required')
    descriptor = _descriptor(parent_descriptor)
    policy = _read(descriptor['path'])['policy']
    bank = load_probe_bank(Path(bank_path))
    members = [m for m in bank['members'] if m['name'] == policy['name']]
    if len(members) != 1 or members[0]['policy'] != policy:
        raise ValueError('parent does not match source frozen bank identity')
    if bank['task']['success_criterion'] != 'stable_forward_recovery':
        raise ValueError('original recovery success protocol required')
    sources = resolve_teacher_sources(teacher_manifest_path, source_rows_path)
    full_roots = [e['root_id'] for e, _ in sources]
    selected = full_roots if roots is None else list(roots)
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(full_roots):
        raise ValueError('explicit existing nonempty TRAIN root subset required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    groups = {}
    for entry, row in sources:
        if entry['root_id'] not in selected:
            continue
        if entry['actor_sha256'] != policy['actor_sha256'] or entry['normalizer_sha256'] != policy['normalizer_sha256'] or entry['model_sha256'] != policy['xml_sha256']:
            raise ValueError('teacher/parent actor, normalizer or model identity mismatch')
        snap = load_unified_envelope_snapshot(Path(row['snapshot']))
        if snapshot_context_sha256(snap) != row['snapshot_context_sha256']:
            raise ValueError('TRAIN root full context identity changed')
        remaining = remaining_horizon(400, snap.episode_step)
        clean = {key:row[key] for key in ('root_id','root_episode_id','role','snapshot','state_sha256','snapshot_context_sha256','snapshot_time')}
        clean.update(original_elapsed=snap.episode_step, remaining_horizon=remaining)
        groups.setdefault(remaining, []).append(clean)
    cases = []
    for horizon, rows in groups.items():
        rows_path = output / f'roots_remaining_{horizon}.json'
        _write(rows_path, rows)
        # The evaluator binds its horizon to the bank. Create an explicit new
        # remaining-horizon bank; never alter the immutable source bank or add
        # an extra original episode length after the restored elapsed clock.
        remaining_bank = output / f'bank_remaining_{horizon}.json'
        lock_probe_bank(dict(version=f'recovery_remaining_{horizon}',task=bank['task'],
            max_ticks=horizon,label_interaction_budget=len(rows)*horizon,
            max_candidates_per_process=bank['max_candidates_per_process'],
            members=[dict(frozen_policy=descriptor['path'],roles=['proposer','evaluator'])]),remaining_bank)
        spec = dict(bank=str(remaining_bank.resolve()), proposer=policy['name'], order=[policy['name']],
            candidates=str(rows_path.resolve()), horizon=horizon, budget=len(rows)*horizon,
            full_matrix=True, record_actor_preobservations=True, record_retention_diagnostics=True,
            reward_mode='original_all_phases', success_criterion=bank['task']['success_criterion'],
            preserve_snapshot_episode_context=True)
        for lane, row in enumerate(rows):
            cases.append(dict(case_id=row['root_id'], root_id=row['root_id'], mode='evaluate',
                role='TRAIN', group='recovery_original_root', lane=lane, capacity=len(rows),
                horizon=horizon, original_elapsed=row['original_elapsed'], repeat_seeds=list(range(repeats)),spec=spec))
    panel = dict(schema='jit_skill_transfer_recovery_panel_v1', full_train_root_ids=full_roots,
        selected_root_ids=selected, original_absolute_horizon=400, cases=cases,
        parent_descriptor=descriptor, teacher_manifest=dict(path=str(Path(teacher_manifest_path).resolve()),sha256=_sha(teacher_manifest_path)),
        source_rows=dict(path=str(Path(source_rows_path).resolve()),sha256=_sha(source_rows_path)),
        bank=dict(path=str(Path(bank_path).resolve()),sha256=_sha(bank_path)),
        execution='independent_complete_actor', forbidden_teacher_switching=True,
        repeats=repeats, repeat_semantics='numerical_label_stability_not_independent_training_seeds',
        preparation_environment_transitions=0)
    path = output / 'panel.json'
    _write(path, panel)
    return dict(panel=panel, descriptor=dict(path=str(path.resolve()),sha256=_sha(path)),policy=policy)


def summarize_recovery(results, full_root_ids, repeats=3):
    if not set(results) <= set(full_root_ids):
        raise ValueError('evaluation includes roots outside declared TRAIN set')
    statuses = {root:stable_repeat_label(row['labels'], repeats) for root,row in results.items()}
    counts = {str(label):sum(v == label for v in statuses.values()) for label in (1,0,'mixed','unknown','execution_error','timeout')}
    return dict(full_train_root_denominator=len(full_root_ids), selected_root_denominator=len(results),
        full_train_root_ids=list(full_root_ids), unselected_roots=[r for r in full_root_ids if r not in results],
        stable_successes=counts['1'], stable_failures=counts['0'],
        mixed=counts['mixed'], unknown=counts['unknown'], execution_error=counts['execution_error'],timeout=counts['timeout'],
        stable_success_rate=counts['1']/len(results) if results else None,
        per_root={root:dict(**row,stable_label=statuses[root]) for root,row in results.items()},
        execution='independent_complete_actor', teacher_executed_during_evaluation=False,
        split='TRAIN_seen_root_diagnostic', repeats=repeats,
        repeat_semantics='numerical_label_stability_not_independent_training_seeds',
        final_TEST_opened=False, qualified_policy_adoption=False)


def evaluate_recovery(params_path, output, parent_descriptor, teacher_manifest_path,
                      source_rows_path, repository, roots=None, repeats=3):
    """Run genuine standalone R (or S) Actor from every selected original root."""
    from .retention_consolidation_runtime import Evaluator, evaluation_reservation
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    descriptor = _descriptor(parent_descriptor)
    prepared = prepare_recovery_panel(output/'inputs', descriptor, teacher_manifest_path,
                                      source_rows_path, roots=roots,repeats=repeats)
    reservation = evaluation_reservation(prepared['panel']['cases'])
    plan = dict(output=str(output.resolve()), repository=str(Path(repository).resolve()),
        config=dict(evaluation_panel=prepared['descriptor'],cost_limits=dict(
            wall_seconds=86400,environment_transitions=reservation)))
    _write(output/'plan.json',plan)
    _write(output/'status.json',dict(phase='running',reserved_interactions=reservation,
        params_path=str(Path(params_path).resolve()), params_sha256=_sha(params_path)))
    evaluator = Evaluator(plan)
    try:
        results = evaluator.evaluate('recovery', params_path, prepared['policy'], descriptor)
        _write(output/'results.json',results)
        summary = summarize_recovery(results, prepared['panel']['full_train_root_ids'],repeats)
        summary.update(params_path=str(Path(params_path).resolve()),params_sha256=_sha(params_path),
                       reserved_interactions=reservation,cost_ledger=str(output/'cost_ledger.json'))
        _write(output/'summary.json',summary)
        _write(output/'status.json',dict(phase='completed',reserved_interactions=reservation,
            charged_interactions=sum(e['charged_transitions'] for e in evaluator.ledger['entries']),
            independent_stable_successes=summary['stable_successes']))
        return summary
    except BaseException as exc:
        _write(output/'status.json',dict(phase='error',error=repr(exc),reserved_interactions=reservation,
            charged_interactions=sum(e.get('charged_transitions',e['reserved_transitions']) for e in evaluator.ledger['entries'])))
        raise
