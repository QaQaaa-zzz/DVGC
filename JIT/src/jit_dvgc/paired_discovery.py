"""Locked pi_2/pi_4 proposer comparison, reusing completed pi_4 evidence."""
from pathlib import Path
import fcntl
import traceback
from .jump_evidence_validation import read,write,file_sha,verify_hash
from .envelope_campaign import read_child
from .dense_tube import run as run_dense
from .discovery_comparison import curve,at_budget
from .analysis.policy_envelopes import write_csv
from .result_bundle import bundle

DEFAULT_SOURCE='JIT/runs/campaign/empirical_envelope_smoke_callbackfix_v1'
DEFAULT_OUTPUT='JIT/runs/discovery/pi2_pi4_paired_v1'
NAMES=('pi_2','pi_4')
PHYSICS_SOURCES=tuple('JIT/src/jit_dvgc/'+name for name in (
    'acquisition/causal_jump.py','unified_env.py','env.py','unified_envelope_snapshot.py',
    'iterative_probe_training.py','unified_continuation_shards.py','frontier_label_shard_runner.py',
    'policy_family_landing.py','snapshot_pool.py'))


def panel(child,recovery=False):
    rows,inputs,plan=read_child(child,recover_figures_failure=recovery)
    manifest=read(Path(child)/'analysis_inputs.json')
    points=read(manifest['projected']);catalog=read(manifest['catalog'])
    labels={m['policy']['name']:read(Path(manifest['merged'][m['policy']['name']])/'labels.json')
            for m in plan['members']} if points else {m['policy']['name']:[] for m in plan['members']}
    ledger=Path(child)/'cost_ledger.json';inputs[str(ledger.resolve())]=file_sha(ledger)
    charged=read(ledger)['charged_interactions']
    return dict(rows=rows,points=points,labels=labels,plan=plan,inputs=inputs,
                catalog=catalog,charged=charged)


def events(data):
    result=[];points=data['points'];labels=data['labels']
    if set(labels)!={f'pi_{i}' for i in range(5)}:raise ValueError('five common evaluators required')
    if any(len(v)!=len(points) for v in labels.values()):raise ValueError('incomplete panel')
    for i,point in enumerate(points):
        evidence=[v[i] for v in labels.values()]
        for row in evidence:
            if any(row[k]!=point[k] for k in ('candidate_id','state_sha256','phase')):raise ValueError('row identity drift')
            if row['success_criterion']!='first_valid_landing' or type(row['label']) is not int or row['label'] not in (0,1):
                raise ValueError('invalid outcome')
            if type(row['environment_interactions']) is not int or row['environment_interactions']<0:raise ValueError('invalid cost')
        ok=any(r['label'] for r in evidence)
        result.append(dict(cost=sum(r['environment_interactions'] for r in evidence),
                           root_cell=point['root_cell'] if ok else None,full_cell=point['full_cell'] if ok else None))
    overhead=data['charged']-sum(r['cost'] for r in result)
    if overhead<0:raise ValueError('ledger undercounts labels')
    return overhead,result


def verify_pair(left,right):
    a,b=left['plan'],right['plan']
    for key in ('members','bank_sha256','physical_resolution','frontier_profile','horizon','label_seed','centerline','baseline'):
        if a[key]!=b[key]:raise ValueError(f'paired contract differs: {key}')
    if any(a['sources'][p]!=b['sources'][p] for p in PHYSICS_SOURCES):raise ValueError('rollout implementation changed')
    if (a['proposer'],b['proposer'])!=NAMES:raise ValueError('wrong proposers')
    def schedule(data):
        trails=data['catalog']['trajectory_receipts']
        if len(trails)!=32 or sum(t['environment_interactions'] for t in trails)!=data['catalog']['environment_interactions']:
            raise ValueError('incomplete 32-trajectory schedule')
        return [{k:t[k] for k in ('trajectory_id','anchor_x_m','strength','direction')} for t in trails]
    if schedule(left)!=schedule(right):raise ValueError('perturbation schedule differs')


def summarize(panels,old_root,training_cost,failed_smoke_cost):
    curves={};metrics=[];sets={}
    for name in NAMES:
        data=panels[name];overhead,ev=events(data);curves[name]=curve(ev,overhead,old_root)
        cells={e['root_cell'] for e in ev if e['root_cell'] is not None}
        sets[name]=cells
        trails=data['catalog']['trajectory_receipts']
        landed=[t for t in trails if t['valid_landing'] and not t['truncated']]
        own=[p for p,r in zip(data['points'],data['labels'][name],strict=True) if r['label']]
        metrics.append(dict(proposer=name,candidates=len(ev),bank_witnesses=sum(e['root_cell'] is not None for e in ev),
            root_cells=len(cells),novel_root_cells=len(cells-old_root),own_root_cells=len({p['root_cell'] for p in own}),
            discovery_interactions=data['charged'],trajectories=len(trails),forward_landings=len(landed),
            truncated_trajectories=sum(t['truncated'] for t in trails),
            lowest_successful_peak_root_z_m=min((t['peak_root_z_m'] for t in landed),default=None)))
    for row in metrics:
        other='pi_4' if row['proposer']=='pi_2' else 'pi_2'
        exclusive=sets[row['proposer']]-sets[other]
        row.update(exclusive_root_cells=len(exclusive),exclusive_novel_root_cells=len(exclusive-old_root))
    comparisons=[]
    scenarios={'exploration_only':0,'plus_completed_probe_training':training_cost}
    if failed_smoke_cost is not None:scenarios['plus_recorded_failed_smoke']=training_cost+failed_smoke_cost
    for mode,surcharge in scenarios.items():
        adjusted={n:[{**r,'interactions':r['interactions']+(surcharge if n=='pi_4' else 0)} for r in curves[n]] for n in NAMES}
        budget=min(rs[-1]['interactions'] for rs in adjusted.values())
        for n in NAMES:
            comparisons.append(dict(scenario=mode,proposer=n,common_budget=budget,training_surcharge=surcharge if n=='pi_4' else 0,
                **{k:v for k,v in at_budget(adjusted[n],budget).items() if k!='interactions'}))
    return metrics,curves,comparisons


def render(panels,curves,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for name,color in zip(NAMES,('#0072B2','#D55E00')):
        data=panels[name];selected=[p for i,p in enumerate(data['points']) if any(v[i]['label'] for v in data['labels'].values())]
        for ax,y in zip(axes[:2],('root_z_m','root_vz_mps')):
            ax.scatter([p['coordinates']['root_x_m'] for p in selected],[p['coordinates'][y] for p in selected],s=6,alpha=.55,label=name,color=color)
            ax.set(xlabel='x (m)',ylabel='Root z (m)' if y=='root_z_m' else 'Root vz (m/s)')
        rs=curves[name];axes[2].step([r['interactions'] for r in rs],[r['novel_root_cells'] for r in rs],where='post',label=name,color=color)
    axes[2].set(xlabel='Exploration interactions',ylabel='Novel witnessed root cells',title='Catalog-order budget replay')
    for ax in axes:ax.grid(alpha=.2);ax.legend()
    for ext in ('png','pdf','svg'):fig.savefig(output/f'paired_discovery.{ext}',dpi=180)
    plt.close(fig)


def run(repo,output,source=None,gpu='0'):
    repo=Path(repo).resolve();source=Path(source or repo/DEFAULT_SOURCE).resolve();output=Path(output).resolve()
    if source==output or output.is_relative_to(source) or source.is_relative_to(output):raise ValueError('use a separate comparison directory')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        report={'status':'engineering_error','training_transitions':0,'final_test_used':False,'physical_boundary_proven':False}
        try:
            pi4=panel(source/'round_000/discovery',recovery=True)
            plan=pi4['plan'];profile=plan['frontier_profile']
            for path in PHYSICS_SOURCES:
                if file_sha(repo/path)!=plan['sources'][path]:raise ValueError('source rollout implementation changed')
            if plan['proposer']!='pi_4' or profile['round_index']!=0:raise ValueError('requires first pi_4 smoke round')
            done=source/'round_000/training_attempt_000/completion.json';completion=read(done)
            for path,sha in completion['artifacts'].items():
                if file_sha(path)!=sha:raise ValueError('completed training changed')
            if completion['policy']!=next(m['path'] for m in plan['members'] if m['policy']['name']=='pi_4'):
                raise ValueError('pi_4 identity mismatch')
            inputs={**pi4['inputs'],**completion['artifacts'],str(done):file_sha(done)}
            old_root=set();seed_map=source/'seed_inputs.json';inputs[str(seed_map)]=file_sha(seed_map)
            seed_inputs=read(seed_map)
            for path,sha in seed_inputs.items():
                if file_sha(path)!=sha:raise ValueError('seed evidence drift')
            for path in sorted({Path(p).parent for p in seed_inputs if Path(p).name=='analysis_inputs.json'}):
                rows,locked,_=read_child(path);inputs.update(locked)
                old_root|={r['root_cell'] for r in rows if r['witnessed']}
            if not old_root:raise ValueError('no seed baseline')
            old_request=source/'round_000/discovery/request.json';inputs[str(old_request)]=file_sha(old_request)
            budget=read(old_request)['budget']
            failed=source.parent/'empirical_envelope_smoke_v1/summary.json'
            failed_cost=None
            if failed.exists():
                previous=read(failed)
                if previous['status']!='engineering_error' or previous.get('rounds'):raise ValueError('unexpected prior failed smoke')
                failed_cost=previous['charged_interactions']
                if type(failed_cost) is not int or failed_cost<0:raise ValueError('invalid prior attempt cost')
                inputs[str(failed)]=file_sha(failed)
            request=dict(source=str(source),reused_proposer='pi_4',new_proposer='pi_2',budget_per_proposer=budget,
                frozen_common_evaluators=[m['policy']['name'] for m in plan['members']],profile=profile,
                baseline_root_cells=len(old_root),completed_training_charge=completion['charged_interactions'],
                known_failed_smoke_charge=failed_cost,inputs=inputs,
                sources={str(p.relative_to(repo)):file_sha(p) for p in (repo/'JIT').rglob('*.py') if 'runs' not in p.parts})
            if (output/'request.json').exists() and read(output/'request.json')!=request:raise ValueError('comparison inputs/code changed; use new output')
            write(output/'request.json',request)
            print('[paired] reuse verified pi_4; run pi_2 only; no PPO',flush=True)
            result=run_dense(repo,output/'pi_2',baseline=Path(plan['baseline']),gpu=gpu,budget=budget,proposer='pi_2',profile=profile)
            if result['status'] not in ('completed','completed_empty'):raise ValueError('pi_2 incomplete; preserve attempts and rerun the same command')
            pi2=panel(output/'pi_2');verify_pair(pi2,pi4)
            panels={'pi_2':pi2,'pi_4':pi4}
            metrics,curves,matched=summarize(panels,old_root,completion['charged_interactions'],failed_cost)
            figures=output/'figures';figures.mkdir(exist_ok=True)
            write_csv(figures/'proposer_metrics.csv',metrics);write_csv(figures/'matched_budget.csv',matched)
            write_csv(figures/'coverage_cost.csv',[{'proposer':n,**row} for n,rs in curves.items() for row in rs])
            for n,data in panels.items():
                write_csv(figures/f'{n}_trajectories.csv',[{k:v for k,v in t.items() if k!='direction'}|
                    {'action':t['direction']['action_name'],'sign':t['direction']['sign']} for t in data['catalog']['trajectory_receipts']])
            render(panels,curves,figures)
            report.update(status='completed',metrics=metrics,matched_budget=matched,new_interactions=pi2['charged'],
                reused_pi4_discovery_interactions=pi4['charged'],baseline_root_cells=len(old_root),
                source_completed_training_charge=completion['charged_interactions'],known_failed_smoke_charge=failed_cost,
                proposer_union_root_cells=metrics[0]['root_cells']+metrics[1]['exclusive_root_cells'],
                proposer_union_novel_root_cells=metrics[0]['novel_root_cells']+metrics[1]['exclusive_novel_root_cells'],
                scope='retrospective matched-schedule TRAIN proposer control; not independent repeated training',
                baseline_scope='two pre-training campaign seed panels; not all historical Tube',
                cost_semantics='catalog-order replay, all acquisition and retries upfront, complete five-evaluator candidate events',
                historical_bootstrap_cost_included=False,global_physical_boundary_claim=False)
        except Exception as exc:
            report.update(error=str(exc),traceback=traceback.format_exc())
            ledger=output/'pi_2/cost_ledger.json'
            if ledger.exists():report['new_interactions']=read(ledger)['charged_interactions']
        finally:
            write(output/'summary.json',report)
            print(f"[paired] {report['status']}\nReturn this file: {bundle(output)}",flush=True)
        return report
