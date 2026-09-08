"""Four frozen proposers, fixed action schedule, observed coverage versus cost.

This is a development pilot. Cost curves replay a predeclared catalog-order
schedule; all acquisition, engineering and padding costs are charged up front.
No interpolated state or fractional evaluation receives a witness.
"""
from pathlib import Path
import fcntl
import traceback

from .dense_tube import run as run_dense, DEFAULT_BASELINE
from .jump_evidence_validation import read, write, file_sha, verify_hash
from .evidence_integrity import canonical_sha256
from .policy_comparison import bundle, verify_plan

NAMES = ['pi_0', 'pi_1', 'pi_2', 'pi_3']
DEFAULT_OUTPUT = 'JIT/runs/discovery/four_proposers_5cm_v1'
DEFAULT_PREVIOUS = 'JIT/runs/dense_tube/pi0_5cm_pilot_v1'


def lock_previous(pilot):
    pilot=Path(pilot)
    if read(pilot/'summary.json')['status']!='completed':
        raise ValueError('completed previous dense pilot required')
    verify_hash(read(pilot/'plan.json'),'plan_sha256')
    verify_hash(read(pilot/'figures/summary.json'),'report_sha256')
    manifest=read(pilot/'analysis_inputs.json')
    paths=[pilot/'plan.json',pilot/'summary.json',pilot/'analysis_inputs.json',
           pilot/'figures/summary.json',Path(manifest['projected'])]
    for directory in manifest['merged'].values():
        paths += [Path(directory)/n for n in ('summary.json','protocol.json','labels.json')]
    return {str(p.resolve()):file_sha(p) for p in paths}


def previous_cells(pilot, resolution):
    from .policy_comparison_runtime import complete_output
    pilot=Path(pilot);plan=read(pilot/'plan.json');verify_hash(plan,'plan_sha256')
    if read(pilot/'summary.json')['status']!='completed':
        raise ValueError('previous pilot is not complete')
    manifest=read(pilot/'analysis_inputs.json');points=read(manifest['projected'])
    report=read(pilot/'figures/summary.json');verify_hash(report,'report_sha256')
    if report['resolution']!=resolution or report.get('plan_sha256')!=plan['plan_sha256']:
        raise ValueError('previous pilot grid/plan mismatch')
    labels={}
    for member in plan['members']:
        name=member['policy']['name']
        result=complete_output(manifest['merged'][name],manifest['catalog'],next(m for m in plan['members'] if m['policy']['name']==plan.get('proposer','pi_0')),member,400,plan.get('label_seed',9841201))
        if result is None:raise ValueError('previous pilot labels incomplete')
        labels[name]=result[1]
    _,events=coverage_events(points,labels,sum(sum(r['environment_interactions'] for r in rows) for rows in labels.values()))
    return {e['root_cell'] for e in events if e['root_cell'] is not None}



def coverage_events(points, labels, charged):
    """Complete four-evaluator rows are atomic; cost includes unsuccessful rows."""
    if set(labels) != set(NAMES) or any(len(rows) != len(points) for rows in labels.values()):
        raise ValueError('incomplete evaluator panel')
    events = []
    for i, point in enumerate(points):
        rows = [labels[n][i] for n in NAMES]
        for row in rows:
            if any(row[k] != point[k] for k in ('candidate_id', 'state_sha256', 'phase')):
                raise ValueError('candidate identity/order mismatch')
            if row.get('success_criterion') != 'first_valid_landing' or type(row['label']) is not int or row['label'] not in (0, 1):
                raise ValueError('endpoint or label mismatch')
            if type(row['environment_interactions']) is not int or row['environment_interactions'] < 0:
                raise ValueError('invalid interaction count')
        events.append({'cost': sum(r['environment_interactions'] for r in rows),
                       'root_cell': point['root_cell'] if any(r['label'] for r in rows) else None,
                       'full_cell': point['full_cell'] if any(r['label'] for r in rows) else None})
    overhead = charged - sum(e['cost'] for e in events)
    if overhead < 0:
        raise ValueError('ledger undercounts completed suffix work')
    return overhead, events


def curve(events, overhead, old_root):
    cost = overhead
    root, full = set(), set()
    result = [{'interactions': cost, 'root_cells': 0, 'full_cells': 0, 'novel_root_cells': 0}]
    for event in events:
        cost += event['cost']
        if event['root_cell'] is not None:
            root.add(event['root_cell']); full.add(event['full_cell'])
        result.append({'interactions': cost, 'root_cells': len(root), 'full_cells': len(full),
                       'novel_root_cells': len(root-old_root)})
    return result


def at_budget(rows, budget):
    allowed = [r for r in rows if r['interactions'] <= budget]
    return allowed[-1] if allowed else {'interactions': budget, 'root_cells': 0, 'full_cells': 0, 'novel_root_cells': 0}


def analyze(output, previous_pilot=None, previous_discovery=None):
    from .policy_comparison_runtime import complete_output
    from .analysis.policy_envelopes import write_csv
    panels, metrics, matrix, all_labels = {}, [], [], {}
    overheads, events = {}, {}
    old_root = None
    for name in NAMES:
        child = output/name
        plan = verify_plan(child/'plan.json')
        if plan.get('proposer') != name:
            raise ValueError('proposer plan mismatch')
        run = read(child/'summary.json')
        if run['status'] not in {'completed','completed_empty'}:
            raise ValueError('incomplete proposer')
        manifest = read(child/'analysis_inputs.json')
        points = read(manifest['projected'])
        acquisition = next(m for m in plan['members'] if m['policy']['name'] == name)
        labels = {}
        for member in plan['members']:
            evaluator = member['policy']['name']
            if not points:
                labels[evaluator] = []
                continue
            completed = complete_output(manifest['merged'][evaluator], manifest['catalog'],
                                        acquisition, member, 400, plan.get('label_seed',9841201))
            if completed is None:
                raise ValueError('invalid completed evaluator labels')
            labels[evaluator] = completed[1]
        report = read(child/'figures/summary.json'); verify_hash(report, 'report_sha256')
        if panels and report['resolution'] != resolution:
            raise ValueError('physical grid changed')
        resolution = report['resolution']
        panels[name] = points
        all_labels[name] = labels
        overheads[name], events[name] = coverage_events(points, labels, run['charged_interactions'])
        own = [p for p, r in zip(points, labels[name]) if r['label']]
        supported = [p for i, p in enumerate(points) if any(labels[n][i]['label'] for n in NAMES)]
        metrics.append({'proposer': name, 'trajectories': len({p['trajectory_id'] for p in points}),
                        'candidates': len(points), 'own_success_states': len(own),
                        'own_root_cells': len({p['root_cell'] for p in own}),
                        'bank_success_states': len(supported),
                        'bank_root_cells': len({p['root_cell'] for p in supported}),
                        'charged_interactions': run['charged_interactions']})
        for evaluator in NAMES:
            selected = [p for p, r in zip(points, labels[evaluator]) if r['label']]
            matrix.append({'proposer': name, 'evaluator': evaluator, 'candidates': len(points),
                           'successful_states': len(selected), 'root_cells': len({p['root_cell'] for p in selected})})
        old = read(Path(plan['baseline'])/'plan.json')
        old_points = read(old['panels']['train']['projected'])
        old_report = read(Path(plan['baseline'])/'figures/train/summary.json')
        baseline_cells = {p['root_cell'] for p, ok in zip(old_points, old_report['masks']['union'], strict=True) if ok}
        if old_root is not None and old_root != baseline_cells:
            raise ValueError('baseline mismatch')
        old_root = baseline_cells
    if previous_pilot is not None:
        old_root |= previous_cells(previous_pilot, resolution)
    if previous_discovery is not None:
        for name in NAMES:
            old_root |= previous_cells(Path(previous_discovery)/name, resolution)
    curves = {n: curve(events[n], overheads[n], old_root) for n in NAMES}
    # Locked round-robin by candidate index; never sort by success or novelty.
    pooled = [events[n][i] for i in range(max(map(len, events.values()))) for n in NAMES if i < len(events[n])]
    curves['bank_round_robin'] = curve(pooled, sum(overheads.values()), old_root)
    budget = min(rows[-1]['interactions'] for rows in curves.values())
    matched = [{'schedule': n, 'budget': budget, **{k:v for k,v in at_budget(rows,budget).items() if k!='interactions'}}
               for n, rows in curves.items()]
    sets = {n: {e['root_cell'] for e in events[n] if e['root_cell'] is not None} for n in NAMES}
    for row in metrics:
        name = row['proposer']
        row['unique_proposer_root_cells'] = len(sets[name] - set().union(*(sets[n] for n in NAMES if n != name)))
        row['novel_vs_previous_train_union'] = len(sets[name]-old_root)
    union = set().union(*sets.values())
    report = {'status':'completed', 'scope':'TRAIN four-proposer development discovery pilot',
              'metrics':metrics, 'matrix':matrix, 'matched_budget':matched,
              'pooled_root_cells':len(union), 'novel_root_cells':len(union-old_root),
              'baseline_scope':'previous shared-panel TRAIN union plus completed 5cm pi0 pilot' if previous_pilot else 'previous shared-panel TRAIN union',
              'baseline_root_cells':len(old_root), 'cumulative_root_cells':len(union|old_root),
              'charged_interactions':sum(r['charged_interactions'] for r in metrics),
              'cost_curve_semantics':'predeclared catalog-order replay; round-robin across proposers; all acquisition, benchmark, failed-attempt and padding costs upfront; complete four-label candidate events only',
              'comparison_scope':'each proposer uses the same four suffix evaluators; own-Actor support is a separate geometric diagnostic',
              'independent_repetitions':1, 'training_transitions':0, 'final_test_used':False,
              'training_admission_authorized':False, 'physical_resolution':resolution,
              'continuous_volume_claim':False,
              'own_policy_semantics':'same frozen Actor for perturbed prefix and fresh-continuation suffix; no claim of uninterrupted exact replay',
              'root_overhead_by_proposer':overheads}
    figures = output/'figures';figures.mkdir(exist_ok=True)
    write_csv(figures/'proposer_metrics.csv', metrics)
    write_csv(figures/'proposer_evaluator_matrix.csv', matrix)
    write_csv(figures/('descriptive_budget.csv' if previous_discovery is not None else 'matched_budget.csv'), matched)
    write_csv(figures/'coverage_cost.csv', [{'schedule':n,**r} for n,rs in curves.items() for r in rs])
    render(panels, all_labels, curves, figures)
    if previous_discovery is not None:
        from .analysis.frontier_evidence import summarize_frontier
        frontier = summarize_frontier(output, panels, all_labels, old_root, figures)
        report['descriptive_budget']=report.pop('matched_budget')
        report.update(frontier=frontier,scope='TRAIN-informed landing/frontier discovery',
                      baseline_scope='previous shared panel + pi0 dense pilot + four-proposer discovery',
                      comparison_scope='unequal TRAIN-informed perturbation profiles; descriptive costs, not a fair policy ranking',
                      matched_budget_is_formal_comparison=False)
    report['report_sha256'] = canonical_sha256(report)
    write(figures/'summary.json', report)
    return report


def render(panels, labels, curves, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    all_points = [p for ps in panels.values() for p in ps]
    colors = ['#0072B2','#E69F00','#009E73','#CC79A7']
    for mode in ('own_policy','bank_continuation'):
        fig,axes=plt.subplots(5,2,figsize=(10,12),squeeze=False)
        for j,name in enumerate(NAMES):
            ps=panels[name]
            selected=[p for i,p in enumerate(ps) if (labels[name][name][i]['label'] if mode=='own_policy'
                      else any(labels[name][n][i]['label'] for n in NAMES))]
            for col,key in enumerate(('root_z_m','root_vz_mps')):
                for row in (j,4):
                    ax=axes[row,col]
                    ax.scatter([p['coordinates']['root_x_m'] for p in selected],
                               [p['coordinates'][key] for p in selected],s=7,color=colors[j],label=name,rasterized=True)
        for row,axs in enumerate(axes):
            for col,ax in enumerate(axs):
                key=('root_z_m','root_vz_mps')[col]
                for setter,field in ((ax.set_xlim,'root_x_m'),(ax.set_ylim,key)):
                    values=[p['coordinates'][field] for p in all_points] or [0.,1.];pad=max(.01,(max(values)-min(values))*.05)
                    setter(min(values)-pad,max(values)+pad)
                ax.set(xlabel='x (m)',ylabel=('z (m)','vz (m/s)')[col],
                       title=NAMES[row] if row<4 else 'Union by proposer');ax.grid(alpha=.15)
        axes[4,0].legend(fontsize=8)
        fig.suptitle('Own proposer + own suffix' if mode=='own_policy' else 'Each proposer + any frozen suffix')
        fig.tight_layout(rect=(0,0,1,.97))
        for ext in ('png','pdf','svg'):fig.savefig(output/f'{mode}_tubes.{ext}',dpi=300)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    for name,rows in curves.items():
        ax.step([0]+[r['interactions'] for r in rows],[0]+[r['novel_root_cells'] for r in rows],where='post',label=name)
    ax.set(xlabel='Accounted environment interactions',ylabel='New root-state cells vs previous TRAIN union',
           title='Fixed-order discovery replay; all setup costs included')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout()
    for ext in ('png','pdf','svg'):fig.savefig(output/f'coverage_cost.{ext}',dpi=300)
    plt.close(fig)


def run(repo, output, *, baseline=None, gpu='0', budget_per_proposer=2_000_000, previous_pilot=None, previous_discovery=None):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output/'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        request={'repo':str(repo),'baseline':str(Path(baseline or repo/DEFAULT_BASELINE).resolve()),
                 'previous_pilot':str(Path(previous_pilot or repo/DEFAULT_PREVIOUS).resolve()),
                 'budget_per_proposer':budget_per_proposer,'proposers':NAMES,
                 'schedule':'catalog-order round-robin v1',
                 'sources':{str(p.relative_to(repo)):file_sha(p) for p in (repo/'JIT').rglob('*.py') if 'runs' not in p.parts}}
        if previous_discovery is not None:
            request['previous_discovery']=str(Path(previous_discovery).resolve())
        baseline_plan=Path(request['baseline'])/'plan.json'
        request['baseline_plan_file_sha256']=file_sha(baseline_plan) if baseline_plan.exists() else None
        if (output/'request.json').exists() and read(output/'request.json')!=request:
            raise ValueError('discovery request changed; choose a new directory')
        write(output/'request.json',request)
        result={'status':'running'}
        try:
            locked=lock_previous(request['previous_pilot'])
            frozen=output/'previous_pilot_files.json'
            if frozen.exists() and read(frozen)!=locked:raise ValueError('previous pilot input drift')
            write(frozen,locked)
            profiles={}
            discovery_locked={}
            if previous_discovery is not None:
                from .frontier_exploration import allocate
                prior=Path(request['previous_discovery'])
                allocation=allocate(read(prior/'summary.json'))
                if (output/'allocation.json').exists() and read(output/'allocation.json')!=allocation:
                    raise ValueError('allocation changed after locking')
                write(output/'allocation.json',allocation)
                profiles=allocation['profiles']
                if any(p['acquisition_ceiling']+p['max_trajectories']*p['max_candidates']*4*400 > budget_per_proposer for p in profiles.values()):
                    raise ValueError('budget below predeclared frontier ceiling; no GPU work started')
                discovery_locked={str(prior/'summary.json'):file_sha(prior/'summary.json')}
                for name in NAMES: discovery_locked.update(lock_previous(prior/name))
                path=output/'previous_discovery_files.json'
                if path.exists() and read(path)!=discovery_locked:raise ValueError('previous discovery drift')
                write(path,discovery_locked)
            for name in NAMES:
                if any(file_sha(p)!=sha for p,sha in discovery_locked.items()):raise ValueError('previous discovery changed during run')
                if any(file_sha(repo/p)!=sha for p,sha in request['sources'].items()):
                    raise ValueError('source changed during multi-proposer run')
                if lock_previous(request['previous_pilot'])!=locked:raise ValueError('previous pilot changed during run')
                child=run_dense(repo,output/name,baseline=request['baseline'],gpu=gpu,
                                budget=budget_per_proposer,proposer=name,**({'profile':profiles[name]} if profiles else {}))
                if child['status'] not in {'completed','completed_empty'}:
                    raise RuntimeError(f'{name} incomplete; inspect preserved child logs')
            result=analyze(output,request['previous_pilot'],request.get('previous_discovery'))
        except BaseException as exc:
            result={'status':'engineering_error','error':str(exc),'traceback':traceback.format_exc()}
        finally:
            write(output/'summary.json',result)
            print(f"[discovery] {result['status']}\nReturn this file: {bundle(output)}",flush=True)
        return result
