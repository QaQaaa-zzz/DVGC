"""Predeclared local action-window refinement of the observed TRAIN gap."""
from collections import Counter
import csv
import fcntl
from pathlib import Path
import traceback

from .jump_evidence_validation import read, write, file_sha, verify_hash
from .evidence_integrity import canonical_sha256
from .dense_tube import run as run_dense
from .result_bundle import bundle

DEFAULT_PREVIOUS = 'JIT/runs/discovery/landing_frontier_v1'
DEFAULT_OUTPUT = 'JIT/runs/discovery/knee_boundary_v1_budgetfix'
SETTINGS = dict(version='knee_boundary_v1', targets=[2.85, 2.9, 2.95],
    strengths=[.15, .175, .2], action_names=['knee'], signs=[1],
    max_trajectories=9, max_candidates=128, sampling_max_x_m=8.,
    acquisition_ceiling=6000, acquisition_seed=9843101, label_seed=9843201,
    serial_only=True)
CEILING = 6000 + 9 * 128 * 4 * 400


def validate_refinement(profile):
    if set(profile) != set(SETTINGS) | {'source_files'} or any(profile[k] != v for k,v in SETTINGS.items()):
        raise ValueError('boundary profile drift')
    if not isinstance(profile['source_files'], dict) or len(profile['source_files']) != 2:
        raise ValueError('boundary source identities required')
    for source, sha in profile['source_files'].items():
        if file_sha(source) != sha:
            raise ValueError('boundary source evidence changed')


def select_profile(previous):
    previous = Path(previous).resolve()
    report = read(previous/'summary.json')
    verify_hash(report, 'report_sha256')
    if (report.get('status') != 'completed' or report.get('scope') != 'TRAIN-informed landing/frontier discovery'
        or report.get('final_test_used') is not False):
        raise ValueError('completed TRAIN frontier evidence required')
    with (previous/'figures/boundary_candidates.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    if Counter(r['status'] for r in rows) != Counter(report['frontier']['candidate_status_counts']):
        raise ValueError('source boundary counts disagree')
    gaps = [r for r in rows if r['status'] == 'no_success_witness']
    if not gaps or any(r['proposer'] != 'pi_1' or r['action'] != 'knee' or float(r['strength']) != .2
                       or int(r['sign']) != 1 for r in gaps) or len({r['trajectory_id'] for r in gaps}) != 1:
        raise ValueError('source gap differs from declared pi_1 positive-knee target')
    return {**SETTINGS, 'source_files': {str(p):file_sha(p) for p in
        (previous/'summary.json', previous/'figures/boundary_candidates.csv')}}


def analyze(output):
    from .policy_comparison import verify_plan
    from .policy_comparison_runtime import complete_output
    from .analysis.frontier_evidence import classify_labels
    from .analysis.policy_envelopes import write_csv
    (output/'figures').mkdir(parents=True, exist_ok=True)
    child = output/'pi_1'
    plan = verify_plan(child/'plan.json')
    manifest = read(child/'analysis_inputs.json')
    points = read(manifest['projected'])
    catalog = read(manifest['catalog'])
    members = plan['members']
    proposer = next(m for m in members if m['policy']['name'] == 'pi_1')
    labels = {}
    for member in members:
        name = member['policy']['name']
        if not points:
            labels[name] = []
            continue
        completed = complete_output(manifest['merged'][name], manifest['catalog'], proposer,
                                    member, 400, plan['label_seed'])
        if completed is None: raise ValueError('incomplete boundary labels')
        labels[name] = completed[1]
    trails = catalog['trajectory_receipts']
    trail_by_id = {t['trajectory_id']:t for t in trails}
    rows = []
    for i, (point, entry) in enumerate(zip(points, catalog['entries'], strict=True)):
        if any(point[k] != entry[k] for k in ('candidate_id','state_sha256')):
            raise ValueError('boundary candidate identity drift')
        values = []
        for name in plan['names']:
            row = labels[name][i]
            if any(row[k] != point[k] for k in ('candidate_id','state_sha256','phase')):
                raise ValueError('boundary label order drift')
            values.append(row['label'])
        status = classify_labels(values)
        perturb = entry['perturbation']
        rows.append(dict(candidate_id=point['candidate_id'], state_sha256=point['state_sha256'],
            trajectory_id=point['trajectory_id'], phase=point['phase'],
            strength=perturb['strength'], status=status,
            anchor_x_m=trail_by_id[point['trajectory_id']]['anchor_x_m'],
            training_admitted=False, **{name: value for name,value in zip(plan['names'],values)},
            **point['coordinates']))
    if len(trails) != SETTINGS['max_trajectories'] or sum(t['environment_interactions'] for t in trails) != catalog['environment_interactions']:
        raise ValueError('incomplete acquisition receipts')
    groups = []
    for trail in trails:
        subset = [r for r in rows if r['trajectory_id'] == trail['trajectory_id']]
        groups.append({k:trail[k] for k in ('trajectory_id','anchor_x_m','strength','stop_reason',
            'physical_failure','timeout','truncated','environment_interactions')} | {
            'candidates':len(subset), **{s:sum(r['status']==s for r in subset) for s in
                ('all_succeed','policy_disagreement','no_success_witness')}})
    write_csv(output/'figures/boundary_candidates.csv',rows)
    write_csv(output/'figures/trajectory_comparison.csv',groups)
    write_csv(output/'figures/training_review_candidates.csv',
              [r for r in rows if r['status'] != 'all_succeed'])
    render(rows, output/'figures')
    report = dict(status='completed', scope='TRAIN local boundary refinement; no fair policy ranking',
        candidate_status_counts=dict(Counter(r['status'] for r in rows)),
        trajectory_stop_counts=dict(Counter(t['stop_reason'] for t in trails)),
        truncated_trajectories=sum(t['truncated'] for t in trails),
        trajectories=len(trails), training_transitions=0, training_admission_authorized=False,
        final_test_used=False, independent_repetitions=1,
        charged_interactions=read(child/'summary.json')['charged_interactions'])
    report['report_sha256'] = canonical_sha256(report)
    return report


def render(rows, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    for ax, strength in zip(axes, SETTINGS['strengths']):
        for status, color in [('all_succeed','#999999'), ('policy_disagreement','#E69F00'),
                               ('no_success_witness','#D55E00')]:
            subset=[r for r in rows if r['strength']==strength and r['status']==status]
            ax.scatter([r['root_x_m'] for r in subset], [r['root_z_m'] for r in subset],
                       s=12, c=color, label=status)
        ax.set(title=f'Positive knee offset {strength}', xlabel='x (m)', ylabel='z (m)')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle('Real pi_1 arrivals; three action-window endpoints per panel')
    fig.tight_layout()
    for ext in ('png','pdf','svg'):
        fig.savefig(destination/f'boundary_refinement.{ext}',dpi=200)
    plt.close(fig)


def run(repo, output, *, previous=None, baseline=None, gpu='0', budget=2_000_000):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output/'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = {'status':'engineering_error','training_transitions':0,'final_test_used':False}
        try:
            profile = select_profile(previous or repo/DEFAULT_PREVIOUS)
            if budget < CEILING: raise ValueError('budget below declared first-attempt ceiling')
            request = dict(profile=profile, baseline=str(Path(baseline).resolve()) if baseline else None, budget=budget)
            if (output/'request.json').exists() and read(output/'request.json') != request:
                raise ValueError('request changed; use a new output directory')
            write(output/'request.json',request)
            child = run_dense(repo, output/'pi_1', baseline=baseline, gpu=gpu, budget=budget,
                              proposer='pi_1', profile=profile)
            if child['status'] not in ('completed','completed_empty'):
                raise RuntimeError('boundary child failed; see pi_1 diagnostics')
            result = analyze(output)
        except Exception as exc:
            result.update(error=str(exc), traceback=traceback.format_exc())
        finally:
            write(output/'summary.json',result)
            print(f"[boundary] {result['status']}\nReturn this file: {bundle(output)}",flush=True)
        return result
