"""Read-only BC retention report; never dispatches simulation or training."""
from pathlib import Path
import numpy as np
from .retention_repair import read,write


def build_report(path):
    p=read(path);root=Path(p['output']);status=read(root/'status.json')
    out=root/'report_001';out.mkdir(exist_ok=False)
    if status['phase']!='completed':
        write(out/'summary.json',dict(status=status,claim='incomplete B; no ability conclusion'))
        return out
    rows=read(root/'B_checkpoint_results.json');finish=read(root/'B_completed.json');costs=read(root/'costs.json')
    from .retention_b import full_results
    baselines={name:full_results(p,Path(p['reused_zero']['DEV'][name]) if p['reused_zero'] else root/('DEV_'+name)) for name in ('pi0','R5')}
    summary=dict(source_revision=p['code_revision'],source_lock=str(root/'source_lock.json'),budget=p['budget'],actual_charged=sum(x['charged'] for x in costs),D2_prior_charged=p['budget']['prior_charged'],baselines=baselines,checkpoints=rows,finish=finish)
    diagnostics=read(root/'selected_diagnostics.json')
    derived=analyze(p,rows,diagnostics,costs,out)
    summary.update(derived);write(out/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .retention_repair_report import CachedArchive
    for group in 'ABCD':
        fig,axes=plt.subplots(4,4,figsize=(12,12))
        lanes=[i for i,c in enumerate(p['batches'][0]['cases']) if c['group']==group]
        sources=[(name,Path(p['reused_zero']['DEV'][name]) if p['reused_zero'] else root/('DEV_'+name)) for name in ('pi0','R5')]+[(f"BC{k}",Path(v['full']['output'])) for k,v in rows.items() if k!='0']
        for label,folder in sources:
            with CachedArchive(folder/'prefixes.npz') as z:
                for ax,lane in zip(axes.flat,lanes):
                    mask=z['prefix_mask'][:,lane];xy=z['qpos'][:,lane,:2][mask]
                    ax.plot(xy[:,0],xy[:,1],label=label,lw=.8);ax.scatter(xy[-1,0],xy[-1,1],s=6);ax.set_aspect('equal',adjustable='datalim')
                    ax.set_title(p['batches'][0]['cases'][lane]['case'],fontsize=7);ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
        axes.flat[0].legend(fontsize=6);fig.suptitle(f'New DEV {group}: complete task, actual endpoints; 16 paired cases')
        fig.tight_layout();fig.savefig(out/f'xy_DEV_{group}.png',dpi=130);plt.close(fig)
    text=['# D2 B-only results','',*[f'![New DEV {g}](xy_DEV_{g}.png)' for g in 'ABCD'],'',f"Code `{p['code_revision']}`; BC updates {finish['completed_updates']}/2000. PPO/E/G updates 0.",'','|update|TRAIN absorption /8|SOLVER_DEV /7|A /16|B /16|C /16|D /16|eligible|','|---|---|---|---|---|---|---|---|']
    for k,v in rows.items():text.append('|'+ '|'.join(map(str,[k,v['train_success'],v['solver_success'],*[v['cells'][g]['success'] for g in 'ABCD'],v['eligible']]))+'|')
    text+=['',*[f"Baseline {name}: "+str({g:sum(v) for g,v in record['labels'].items()}) for name,record in baselines.items()], '',f"Selected update {finish['selected_update']} by all-seven independent SOLVER_DEV, earliest tie, excluding confirmed B/D regressions.",'','TRAIN is seen-root absorption. SOLVER_DEV is small-sample development transfer. Full DEV measures paired old losses and new gains; no TEST or policy publication. R5 remains the global best_dev_candidate.','',f"Charged new physics {sum(x['charged'] for x in costs)}; prior D2 {p['budget']['prior_charged']}; maximum {p['budget']['total']}. Full immutable state/labels/inputs in summary.json and parent run."]
    text+=['',derived['decision'],'','[Physical failure phases and first action/state differences](trajectory_diagnostics.json). [Same-root teacher/student matrix and confirmations](conversion_matrix.json). Full Actor gradient audit remains in ../B/metrics.jsonl.','',f"Active {derived['active_physics']}; padding {derived['padding_physics']}; measured wall {derived['wall_seconds']:.1f}s. Saved normalizer and critic were verified unchanged; Actor/Adam/RNG/absolute supervised clock are preserved, without an unimplemented resume claim."]
    (out/'INDEX.md').write_text('\n'.join(text)+'\n');return out


def analyze(p,rows,diagnostics,costs,out):
    from .retention_repair_report import CachedArchive
    from .retention_next_report import trace_anomalies
    root=Path(p['output']);selected=read(root/'B_completed.json')['selected_update'];latest=max(map(int,rows));chosen=rows[str(selected)];last=rows[str(latest)]
    matrix=[];physical=[]
    metadata={x['root']['root_id']:x for x in read(p['train_roots'])+read(p['solver_roots'])}
    for step,v in rows.items():
        for role in ('train','solver'):
            for entry in v[role]:
                attempt=entry['combinations']['independent']['attempt'];context=metadata[entry['root_id']]['root']
                physical.append(dict(update=int(step),root_id=entry['root_id'],role=entry['role'],**trace_anomalies(attempt,context)))
    for role in ('train','solver'):
        for entry in chosen[role]:
            rid=entry['root_id'];t=metadata[rid];combinations=entry['combinations'].copy()
            if role=='train':
                extra=next((r for r in diagnostics['combinations'] if r['root_id']==rid),None)
                if extra and diagnostics['diagnostic_update']==selected:combinations.update(extra['combinations'])
            matrix.append(dict(root_id=rid,role=entry['role'],G_solution=1 if entry['G_verified'] else None,student_independent=combinations['independent']['label'],R5_original=entry['R5_label'],combinations=combinations,
                TRAIN_repeat_update=diagnostics['TRAIN_confirmation_update'],diagnostic_combinations=extra['combinations'] if role=='train' and extra else None,diagnostic_update=diagnostics['diagnostic_update'],TRAIN_repeat_labels=[next(x for x in r if x['root_id']==rid)['combinations']['independent']['label'] for r in diagnostics['TRAIN_confirmations']] if role=='train' else None))
    difference=[]
    for entry in last['train']:
        rid=entry['root_id'];meta=metadata[rid];student=entry['combinations']['independent']['attempt'];teacher=meta['teacher']['methods']['G']['verified_attempts'][-1]
        with CachedArchive(student['trace']) as a,CachedArchive(teacher['trace']) as b:
            al=student['trace_lane'];bl=teacher['trace_lane'];n=min(int(a['mask'][:,al].sum()),int(b['mask'][:,bl].sum()))
            obs=np.max(np.abs(a['actor_observation_before'][:n,al]-b['actor_observation_before'][:n,bl]),axis=-1)
            act=np.max(np.abs(a['normalized_action_executed'][:n,al]-b['normalized_action_executed'][:n,bl]),axis=-1)
            first=lambda x: int(np.flatnonzero(x>1e-6)[0]) if np.any(x>1e-6) else None
            difference.append(dict(root_id=rid,update=latest,initial_observation_maxabs=float(obs[0]),first_action_difference=first(act),first_preaction_observation_difference=first(obs),prefix_action_RMSE=float(np.sqrt(np.mean((a['normalized_action_executed'][:min(16,n),al]-b['normalized_action_executed'][:min(16,n),bl])**2))),student_anomaly=trace_anomalies(student,meta['root']),teacher_anomaly=trace_anomalies(teacher,meta['root']),threshold=1e-6,causal_claim='temporal diagnostic only'))
    write(out/'trajectory_diagnostics.json',dict(physical=physical,latest_student_teacher_difference=difference));write(out/'conversion_matrix.json',matrix)
    active=padding=0
    for c in costs:
        receipt=read(root/c['stage']/'status.json');active+=receipt.get('active_interactions',0);padding+=receipt.get('padding_interactions',0)
    if chosen['train_success'] and chosen['solver_success']>rows['0']['solver_success']:decision='TRAIN absorption and small DEV transfer observed; review retention before proposing a separately bounded PPO comparison. No A/C launched.'
    elif last['train_success']:decision='Some seen TRAIN lessons were absorbed; unseen-root DEV transfer is insufficient. Prioritize limited new TRAIN coverage or student-state diagnosis; do not automatically start PPO.'
    else:decision='Offline imitation has not produced closed-loop TRAIN success. Prioritize prefix imitation / tail retention / closed-loop distribution-shift diagnosis; do not start PPO128k or enlarge this budget.'
    return dict(decision=decision,active_physics=active,padding_physics=padding,wall_seconds=read(root/'status.json')['updated_unix']-read(root/'launch.json')['started_unix'],conversion_matrix=matrix)
