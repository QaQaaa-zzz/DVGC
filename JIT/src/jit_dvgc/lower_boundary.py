"""Search observed lower successful support using only real action rollouts."""
from collections import defaultdict
from pathlib import Path
import math
import fcntl
import traceback
from .jump_evidence_validation import read,write,file_sha,verify_hash
from .evidence_integrity import canonical_sha256
from .dense_tube import run as run_dense
from .result_bundle import bundle

DEFAULT_SOURCE='JIT/runs/training_support/complementary_boundary_v1'
DEFAULT_OUTPUT='JIT/runs/discovery/lower_boundary_v1'
SETTINGS=dict(version='lower_boundary_v1',targets=[2.85,2.95],strengths=[.1,.2],
    action_names=['hip','knee'],signs=[-1,1],max_trajectories=16,max_candidates=128,
    sampling_max_x_m=8.,acquisition_ceiling=8000,acquisition_seed=9844101,
    label_seed=9844201,serial_only=True)
CEILING=8000+16*128*4*400


def validate_profile(profile):
    if set(profile)!=set(SETTINGS)|{'source_files'} or any(profile[k]!=v for k,v in SETTINGS.items()):
        raise ValueError('lower-boundary profile drift')
    if len(profile['source_files'])!=2:raise ValueError('source identities required')
    for path,sha in profile['source_files'].items():
        if file_sha(path)!=sha:raise ValueError('lower-boundary source changed')


def select_profile(source):
    source=Path(source).resolve()
    summary=read(source/'summary.json');support=read(source/'support.json')
    verify_hash(support,'support_sha256')
    if (summary.get('status')!='completed' or support.get('role')!='train' or support.get('final_test_used') is not False
        or summary.get('support_sha256')!=support['support_sha256']
        or support['initializer']['policy']['name']!='pi_2'):
        raise ValueError('verified TRAIN pi_2 support source required')
    return {**SETTINGS,'source_files':{str(p):file_sha(p) for p in (source/'support.json',source/'summary.json')}}


def lower_slices(rows):
    """Observed minima only; independent slices never imply one executable path."""
    groups=defaultdict(list)
    for row in rows:
        if row['bank_success']:
            # Same declared 5 cm display bins, separately for ascent/descent.
            key=(row['phase'],int(math.floor(row['x_m']/.05+.5)))
            groups[key].append(row)
    result=[]
    for (phase,index),points in sorted(groups.items()):
        low=min(points,key=lambda r:(r['z_m'],r['candidate_id']))
        result.append(dict(phase=phase,x_slice_center_m=index*.05,observed_min_root_z_m=low['z_m'],
            actual_x_m=low['x_m'],candidate_id=low['candidate_id'],trajectory_id=low['trajectory_id'],
            state_sha256=low['state_sha256'],samples=len(points),continuous_path_claim=False))
    return result


def analyze(output):
    from .policy_comparison import verify_plan
    from .policy_comparison_runtime import complete_output
    from .analysis.policy_envelopes import write_csv
    from .analysis.frontier_evidence import classify_labels
    child=output/'pi_2';plan=verify_plan(child/'plan.json')
    manifest=read(child/'analysis_inputs.json');points=read(manifest['projected'])
    catalog=read(manifest['catalog']);members=plan['members']
    proposer=next(m for m in members if m['policy']['name']=='pi_2')
    labels={}
    for m in members:
        name=m['policy']['name']
        if not points:labels[name]=[];continue
        completed=complete_output(manifest['merged'][name],manifest['catalog'],proposer,m,400,plan['label_seed'])
        if completed is None:raise ValueError('missing evaluator results')
        labels[name]=completed[1]
    trails=catalog['trajectory_receipts']
    if len(trails)!=16 or sum(t['environment_interactions'] for t in trails)!=catalog['environment_interactions']:
        raise ValueError('incomplete lower-search acquisition receipts')
    by_id={t['trajectory_id']:t for t in trails}
    rows=[]
    for i,(point,entry) in enumerate(zip(points,catalog['entries'],strict=True)):
        if any(point[k]!=entry[k] for k in ('candidate_id','state_sha256')):raise ValueError('source identity drift')
        values=[]
        for name in plan['names']:
            label=labels[name][i]
            if any(label[k]!=point[k] for k in ('candidate_id','state_sha256','phase')):raise ValueError('label identity drift')
            values.append(label['label'])
        status=classify_labels(values)
        trail=by_id[point['trajectory_id']]
        rows.append(dict(candidate_id=point['candidate_id'],state_sha256=point['state_sha256'],
            trajectory_id=point['trajectory_id'],phase=point['phase'],x_m=point['coordinates']['root_x_m'],
            z_m=point['coordinates']['root_z_m'],bank_success=any(values),status=status,
            forward_trajectory_landed=trail['valid_landing'],**dict(zip(plan['names'],values))))
    landed=[t for t in trails if t['valid_landing'] and not t['truncated']]
    if any(not math.isfinite(t['peak_root_z_m']) for t in trails):raise ValueError('invalid peak measurement')
    best=min(landed,key=lambda t:(t['peak_root_z_m'],t['trajectory_id'])) if landed else None
    dest=output/'figures';dest.mkdir(parents=True,exist_ok=True)
    write_csv(dest/'all_arrivals.csv',rows)
    slices=lower_slices(rows);write_csv(dest/'observed_lower_slices.csv',slices)
    write_csv(dest/'trajectory_outcomes.csv',[{k:v for k,v in t.items() if k!='direction'}|
        {'action':t['direction']['action_name'],'sign':t['direction']['sign']} for t in trails])
    if best:write_csv(dest/'lowest_peak_successful_trajectory.csv',[r for r in rows if r['trajectory_id']==best['trajectory_id']])
    render(rows,slices,dest)
    result=dict(status='completed',scope='TRAIN observed lower-boundary search',
        candidate_count=len(rows),bank_witnesses=sum(r['bank_success'] for r in rows),
        trajectories=len(trails),forward_landings=len(landed),
        truncated_trajectories=sum(t['truncated'] for t in trails),
        lowest_observed_successful_peak_root_z_m=best['peak_root_z_m'] if best else None,
        lowest_peak_trajectory_id=best['trajectory_id'] if best else None,
        peak_sampling='control-step states; not continuous-time maximum',
        global_minimum_claim=False,slice_minima_form_one_trajectory=False,
        training_transitions=0,final_test_used=False,
        charged_interactions=read(child/'summary.json')['charged_interactions'])
    result['report_sha256']=canonical_sha256(result)
    return result


def render(rows,slices,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharex=True,sharey=True)
    for phase,ax in zip(('upstream','downstream'),axes):
        for valid,color,label in [(False,'#bbbbbb','No current bank witness'),(True,'#0072B2','Landing witness')]:
            part=[r for r in rows if r['phase']==phase and r['bank_success']==valid]
            ax.scatter([r['x_m'] for r in part],[r['z_m'] for r in part],s=9,c=color,label=label)
        part=[r for r in slices if r['phase']==phase]
        ax.scatter([r['actual_x_m'] for r in part],[r['observed_min_root_z_m'] for r in part],
                   marker='x',s=25,c='#D55E00',label='Observed slice minimum')
        ax.set(title=phase,xlabel='root x (m)',ylabel='root z (m)');ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.tight_layout()
    for ext in ('png','pdf','svg'):fig.savefig(destination/f'lower_boundary.{ext}',dpi=200)
    plt.close(fig)


def run(repo,output,*,source=None,baseline=None,gpu='0',budget=3_500_000):
    repo,output=Path(repo).resolve(),Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    with (output/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result={'status':'engineering_error','training_transitions':0}
        try:
            profile=select_profile(source or repo/DEFAULT_SOURCE)
            if budget<CEILING:raise ValueError('budget below first-attempt ceiling')
            child=run_dense(repo,output/'pi_2',baseline=baseline,gpu=gpu,budget=budget,proposer='pi_2',profile=profile)
            if child['status'] not in ('completed','completed_empty'):raise RuntimeError('lower-search child failed')
            result=analyze(output)
        except Exception as exc:result.update(error=str(exc),traceback=traceback.format_exc())
        finally:
            write(output/'summary.json',result)
            print(f"[lower] {result['status']}\nReturn this file: {bundle(output)}",flush=True)
        return result
