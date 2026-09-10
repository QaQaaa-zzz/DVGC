"""All-proposer campaign evidence reuse and reproducible per-round exports."""
from pathlib import Path
import math
from .jump_evidence_validation import read,write,file_sha
from .analysis.policy_envelopes import write_csv


def inherited_bank(repo):
    from .paired_discovery import DEFAULT_SOURCE,DEFAULT_OUTPUT,panel,verify_pair,PHYSICS_SOURCES
    source=Path(repo)/DEFAULT_SOURCE;paired=Path(repo)/DEFAULT_OUTPUT
    report=read(paired/'summary.json')
    if report['status']!='completed':raise ValueError('completed pi_2/pi_4 control required')
    request=read(paired/'request.json')
    for path,sha in request['inputs'].items():
        if file_sha(path)!=sha:raise ValueError('paired source changed')
    pi4=panel(source/'round_000/discovery',recovery=True);pi2=panel(paired/'pi_2')
    verify_pair(pi2,pi4)
    for path in PHYSICS_SOURCES:
        if file_sha(Path(repo)/path)!=pi4['plan']['sources'][path]:raise ValueError('inherited physics implementation changed')
    done=source/'round_000/training_attempt_000/completion.json';completion=read(done)
    for path,sha in completion['artifacts'].items():
        if file_sha(path)!=sha:raise ValueError('inherited training changed')
    member=next(m for m in pi4['plan']['members'] if m['policy']['name']=='pi_4')
    if completion['policy']!=member['path']:raise ValueError('inherited pi_4 identity mismatch')
    locked={**pi2['inputs'],**pi4['inputs'],**completion['artifacts']}
    for path in (done,paired/'summary.json',paired/'request.json'):locked[str(path)]=file_sha(path)
    failed=report.get('known_failed_smoke_charge') or 0
    charge=pi2['charged']+pi4['charged']+completion['charged_interactions']+failed
    return pi2['rows']+pi4['rows'],locked,[completion['policy']],charge


def explore_bank(repo,round_dir,names,extra,inputs,baseline,gpu,budget,cost):
    from .envelope_campaign import profile_for,read_child
    from .dense_tube import run
    # Equal opportunity, fixed 32 legal action trajectories per proposer. Adaptive
    # allocation is not claimed by this first version.
    evidence=round_dir/'bank_inputs.json'
    if evidence.exists() and read(evidence)!=inputs:raise ValueError('bank round inputs changed')
    write(evidence,inputs)
    profile=profile_for(len(extra)-1,list(extra),{str(evidence):file_sha(evidence)})
    results=[]
    for name in names:
        child=round_dir/'proposers'/name
        used=read(child/'cost_ledger.json')['charged_interactions'] if (child/'cost_ledger.json').exists() else 0
        available=budget-cost()+used
        ceiling=16000+32*128*(4+len(extra))*400
        if not (child/'request.json').exists() and available<ceiling:return None
        local=read(child/'request.json')['budget'] if (child/'request.json').exists() else available
        print(f'[campaign] explore {name}; {child}',flush=True)
        outcome=run(repo,child,baseline=baseline,gpu=gpu,budget=local,proposer=name,profile=profile)
        if outcome['status'] not in ('completed','completed_empty'):raise RuntimeError(f'{name} discovery incomplete; preserve attempts')
        rows,locked,_=read_child(child);results.append((rows,locked,str(child.resolve())))
    return results


def export_round(directory,rows,baseline_cells,charged,inherited):
    """All exact reached points and outcomes, plus observed 5 cm slice summaries.

    Blank evaluator cells mean untested, never failure. All-data figures represent
    proposer-conditioned witnessed arrivals, not a common-panel Actor ranking.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=Path(directory);figdir=directory/'figures';figdir.mkdir(parents=True,exist_ok=True)
    sources=sorted({r['source'] for r in rows})
    plans={s:read(Path(s)/'plan.json') for s in sources}
    source_identity={s:dict(plan_sha256=plans[s]['plan_sha256'],bank_sha256=plans[s]['bank_sha256']) for s in sources}
    manifest_path=figdir/'figure_manifest.json'
    if manifest_path.exists():
        if read(manifest_path)['sources']!=source_identity:raise ValueError('completed figure sources changed')
        return  # Preserve the originally recorded stage cost on campaign resume.
    groups={}
    for row in rows:groups.setdefault(plans[row['source']]['proposer'],[]).append(row)
    evaluator_names=sorted({n for r in rows for n in r['labels']})
    coord_names=sorted({k for r in rows for k in r['coordinates']})
    fields=['key','proposer','phase','trajectory_id','source','state_sha256','snapshot_context_sha256','root_cell','witnessed']
    allpoints=[];metrics=[];slices=[]
    allsets={n:{r['root_cell'] for r in rs if r['witnessed']} for n,rs in groups.items()}
    coords=[r['coordinates'] for r in rows if r['coordinates']]
    limits={k:(min(c[k] for c in coords)-.01,max(c[k] for c in coords)+.01) for k in ('root_x_m','root_z_m','root_vz_mps') if coords and all(k in c for c in coords)}
    def axes_setup(axes):
        for ax,key in zip(axes,('root_z_m','root_vz_mps')):
            ax.set(xlabel='x (m)',ylabel=key);ax.grid(alpha=.2)
            if 'root_x_m' in limits:ax.set_xlim(limits['root_x_m'])
            if key in limits:ax.set_ylim(limits[key])
    def save(fig,name):
        for ext in ('png','pdf','svg'):fig.savefig(figdir/f'{name}.{ext}',dpi=180)
        plt.close(fig)
    combined,combined_axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for name,rs in sorted(groups.items()):
        table=[]
        for r in rs:
            record={k:r.get(k) for k in fields};record['proposer']=name
            record.update({k:r['coordinates'].get(k) for k in coord_names})
            record.update({f'evaluator_{n}':r['labels'].get(n) for n in evaluator_names})
            table.append(record)
        allpoints+=table;write_csv(figdir/f'{name}_points.csv',table)
        witnessed=[r for r in rs if r['witnessed']]
        other=set().union(*(s for n,s in allsets.items() if n!=name))
        metrics.append(dict(proposer=name,candidates=len(rs),witnessed=len(witnessed),
            root_cells=len(allsets[name]),novel_vs_round_start=len(allsets[name]-baseline_cells),
            exclusive_observed_cells=len(allsets[name]-other)))
        buckets={}
        for r in witnessed:
            c=r['coordinates'];bucket=(r['phase'],math.floor(c['root_x_m']/.05))
            buckets.setdefault(bucket,[]).append(c)
        for (phase,b),cs in sorted(buckets.items()):
            item=dict(proposer=name,phase=phase,x_left_m=b*.05,x_right_m=(b+1)*.05,count=len(cs))
            for key in coord_names:
                vals=[c[key] for c in cs if key in c]
                item[key+'_min']=min(vals) if vals else None;item[key+'_max']=max(vals) if vals else None
            slices.append(item)
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for ax,combined_ax,key in zip(axes,combined_axes,('root_z_m','root_vz_mps')):
            for ok,color,label in ((False,'#999999','No bank witness'),(True,'#16876b','Bank witness')):
                pts=[r['coordinates'] for r in rs if r['witnessed']==ok and key in r['coordinates']]
                if pts:ax.scatter([c['root_x_m'] for c in pts],[c[key] for c in pts],s=5,c=color,label=label,alpha=.65)
            pts=[r['coordinates'] for r in witnessed if key in r['coordinates']]
            if pts:combined_ax.scatter([c['root_x_m'] for c in pts],[c[key] for c in pts],s=4,label=name,alpha=.55)
        axes_setup(axes);axes[0].legend(fontsize=7);fig.suptitle(f'{name}: cumulative observed arrivals | TRAIN')
        save(fig,f'{name}_envelope')
    axes_setup(combined_axes);combined_axes[0].legend(fontsize=7);save(combined,'all_proposers')
    write_csv(figdir/'all_points.csv',allpoints);write_csv(figdir/'proposer_metrics.csv',metrics);write_csv(figdir/'x_slices_005m.csv',slices)
    trajectories=[]
    for source in sources:
        manifest=read(Path(source)/'analysis_inputs.json');catalog=read(manifest['catalog'])
        for receipt in catalog.get('trajectory_receipts',[]):
            trajectories.append(dict(source=source,proposer=plans[source]['proposer'],receipt=receipt))
    write(figdir/'trajectory_receipts.json',trajectories)
    write(figdir/'figure_manifest.json',dict(schema='jit_campaign_figures_v1',scope='cumulative TRAIN arrivals by proposer; differing panels',
        sources=source_identity,
        slice_width_m=.05,physical_resolution=plans[sources[-1]]['physical_resolution'],
        blank_labels='untested',hulls_filled=False,physical_boundary_proven=False,
        charged_new_interactions=charged,inherited_recorded_charge=inherited,
        baseline_scope='all inherited and prior round witnessed cells',historical_bootstrap_cost_included=False))
