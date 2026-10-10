"""Read-only BC retention report; never dispatches simulation or training."""
from pathlib import Path
import numpy as np
from .retention_repair import read,write


def build_report(path):
    from .retention_b import COVERAGE_SCHEMA
    if read(path).get('schema')==COVERAGE_SCHEMA:return coverage_report(path)
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
    compact_xy(p,rows,finish,out)
    text=['# D2 B-only results','',*[f'![New DEV {g}](xy_DEV_{g}.png)' for g in 'ABCD'],'',f"Code `{p['code_revision']}`; BC updates {finish['completed_updates']}/2000. PPO/E/G updates 0.",'','|update|TRAIN absorption /8|SOLVER_DEV /7|A /16|B /16|C /16|D /16|eligible|','|---|---|---|---|---|---|---|---|']
    for k,v in rows.items():text.append('|'+ '|'.join(map(str,[k,v['train_success'],v['solver_success'],*[v['cells'][g]['success'] for g in 'ABCD'],v['eligible']]))+'|')
    text+=['',*[f"Baseline {name}: "+str({g:sum(v) for g,v in record['labels'].items()}) for name,record in baselines.items()], '',f"Selected update {finish['selected_update']} by all-seven independent SOLVER_DEV, earliest tie, excluding confirmed B/D regressions.",'','TRAIN is seen-root absorption. SOLVER_DEV is small-sample development transfer. Full DEV measures paired old losses and new gains; no TEST or policy publication. R5 remains the global best_dev_candidate.','',f"Charged new physics {sum(x['charged'] for x in costs)}; prior D2 {p['budget']['prior_charged']}; maximum {p['budget']['total']}. Full immutable state/labels/inputs in summary.json and parent run."]
    text+=['',f"Fixed TRAIN repeats: {derived['TRAIN_repeat_counts']}.",f"Selected full DEV first old losses {derived['paired_retention']['first_lost']}, new gains {derived['paired_retention']['first_gain']}; repeat old losses {derived['paired_retention'].get('repeat_lost','NA')}, losses in both paired runs {derived['paired_retention'].get('both_pair_lost','NA')}.",'','Root55: studentH16+pi0=success, teacherH16+student=success, first independent=fail; its independent repeat is ambiguous, so this is a conditional own-state tail diagnostic. Root63: studentH16+pi0=fail, teacherH16+student=success, independent=fail: prefix/closed-loop state drift persists. Prefixes match exactly on roots10/55/63; root68 state-first numerical divergence prevents attributing all differences only to handoff. All bridges remain before first valid landing; never move them merely because failure occurs after contact.','',derived['decision'],'','[Physical failure phases and first action/state differences](trajectory_diagnostics.json). [Same-root teacher/student matrix and confirmations](conversion_matrix.json). Full Actor gradient audit remains in ../B/metrics.jsonl.','',f"Active {derived['active_physics']}; padding {derived['padding_physics']}; measured wall {derived['wall_seconds']:.1f}s. Saved normalizer and critic were verified unchanged; Actor/Adam/RNG/absolute supervised clock are preserved, without an unimplemented resume claim."]
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
    repeat_counts={name:0 for name in ('stable_success','stable_failure','ambiguous','unknown')}
    repeat_records=[]
    for entry in chosen['train']:
        rid=entry['root_id'];labels=[entry['combinations']['independent']['label']]+[next(x for x in repeat if x['root_id']==rid)['combinations']['independent']['label'] for repeat in diagnostics['TRAIN_confirmations']]
        kind=classify_repeats(labels);repeat_counts[kind]+=1;repeat_records.append(dict(root_id=rid,labels=labels,classification=kind))
    write(out/'TRAIN_repeat_labels.json',repeat_records)
    retention=dict(first_lost=sum(c['N10'] for c in chosen['cells'].values()),first_gain=sum(c['N01'] for c in chosen['cells'].values()))
    repeat_file=root/f'B{selected:04d}_warning_repeat.json'
    if repeat_file.exists():
        paired=read(repeat_file);detail={}
        for group in 'ABCD':
            a,b,ar,br=[np.asarray(paired[name]['labels'][group],bool) for name in ('baseline','original','repeated_pi0','repeated')]
            cases=[c['case'] for c in p['batches'][0]['cases'] if c['group']==group]
            detail[group]=dict(first_lost=int((a&~b).sum()),repeat_lost=int((ar&~br).sum()),both_pair_lost_cases=[cases[i] for i in np.flatnonzero(a&ar&~b&~br)],source_label_flips=int((a!=ar).sum()),student_label_flips=int((b!=br).sum()))
        retention.update(repeat_lost=sum(v['repeat_lost'] for v in detail.values()),both_pair_lost=sum(len(v['both_pair_lost_cases']) for v in detail.values()),by_cell=detail,warning_repeat_confirmed=paired['confirmed'])
    write(out/'paired_retention.json',retention)
    from .retention_b import COVERAGE_SCHEMA
    prefix_equivalence=read(root/'H16_prefix_equivalence.json') if p.get('schema')!=COVERAGE_SCHEMA else dict(status='not_repeated',reason='new stage changes keep only; original four-combination receipts retained in source B')
    write(out/'H16_prefix_equivalence.json',prefix_equivalence)
    write(out/'BC_full_state_verification.json',read(root/'BC_full_state_verification.json'))
    decision='BC distillation works on some seen roots and transfers to the three G-solved unseen DEV roots, but is incomplete and retains paired old losses. Prioritize closed-loop prefix/tail diagnosis and independently sampled TRAIN/keep coverage before PPO; propose any A/C only after separate four-pool GPU validation and actual shared-clipping audit. No automatic next stage.'
    return dict(decision=decision,TRAIN_repeat_counts=repeat_counts,paired_retention=retention,active_physics=active,padding_physics=padding,wall_seconds=read(root/'status.json')['updated_unix']-read(root/'launch.json')['started_unix'],conversion_matrix=matrix,R5_focus_confirmations=diagnostics['R5_focus_confirmations'])


def classify_repeats(labels):
    if not labels or any(x is None for x in labels):return 'unknown'
    if all(x==1 for x in labels):return 'stable_success'
    if all(x==0 for x in labels):return 'stable_failure'
    return 'ambiguous'


def compact_xy(p,rows,finish,out):
    import matplotlib.pyplot as plt
    from .retention_repair_report import CachedArchive
    root=Path(p['output']);selected=finish['selected_update'];chosen=rows[str(selected)]
    sources=[(name,Path(p['reused_zero']['DEV'][name]) if p['reused_zero'] else root/('DEV_'+name),color) for name,color in [('pi0','black'),('R5','tab:blue')]]+[(f'BC{selected}',Path(chosen['full']['output']),'tab:orange')]
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for name,path,color in sources:
        with CachedArchive(path/'prefixes.npz') as z:
            for ax,group in zip(axes.flat,'ABCD'):
                lanes=[i for i,c in enumerate(p['batches'][0]['cases']) if c['group']==group]
                for j,lane in enumerate(lanes):
                    mask=z['prefix_mask'][:,lane];xy=z['qpos'][mask,lane,:2];success=bool(np.any(z['success'][:,lane]&mask))
                    ax.plot(xy[:,0],xy[:,1],color=color,lw=.7,alpha=.7,label=name if j==0 else None);ax.scatter(*xy[-1],s=10,c=color,marker='o' if success else 'x')
                ax.set_title(f'{group}:16 paired DEV cases');ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');ax.set_aspect('equal',adjustable='datalim')
    axes.flat[0].legend(fontsize=8);fig.suptitle(f'Complete tasks: actual XY and true endpoints; pi0/R5/BC{selected}')
    fig.tight_layout();fig.savefig(out/'xy_DEV_overview.png',dpi=140);plt.close(fig)
    meta={e['root']['root_id']:e for e in read(p['train_roots'])+read(p['solver_roots'])}
    for role in ('train','solver'):
        entries=chosen[role];fig,axes=plt.subplots(2,4,figsize=(14,7))
        zero={e['root_id']:e for e in rows['0'][role]}
        for ax,e in zip(axes.flat,entries):
            rid=e['root_id'];m=meta[rid]
            curves=[('pi0',zero[rid]['combinations']['independent']['attempt'],'black'),(f'BC{selected}',e['combinations']['independent']['attempt'],'tab:orange')]
            if e['G_verified']:curves.append(('G_H16+pi0',m['teacher']['methods']['G']['verified_attempts'][-1],'tab:purple'))
            for name,a,color in curves:
                with CachedArchive(a['trace']) as z:
                    lane=a['trace_lane'];xy=z['qpos'][z['mask'][:,lane],lane,:2];ax.plot(xy[:,0],xy[:,1],label=name,color=color,lw=1);ax.scatter(*xy[-1],color=color,s=18,marker='o' if a['label']==1 else 'x')
            ax.set_title(rid,fontsize=8);ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');ax.set_aspect('equal',adjustable='datalim')
        if len(entries)<8:axes.flat[-1].axis('off')
        axes.flat[0].legend(fontsize=7);fig.suptitle(f'{role.upper()}: actual same-snapshot XY, endpoints; G without verified solution is NA')
        fig.tight_layout();fig.savefig(out/f'xy_{role}_roots.png',dpi=140);plt.close(fig)


def coverage_report(path):
    """New-stage results only. Prepared/failed outputs never claim policy improvement."""
    p=read(path);root=Path(p['output']);status=read(root/'status.json')
    number=max([0]+[int(x.name.split('_')[-1]) for x in root.glob('coverage_report_*') if x.is_dir()])+1
    out=root/f'coverage_report_{number:04d}';out.mkdir(exist_ok=False)
    summary=dict(schema=p['schema'],source_B=p['source_B'],budget=p['budget'],status=status,
        evaluation_status='not_started',new_keep_collection='PENDING',reliable_recovery_improvement='NOT_ESTABLISHED',
        learner_last=status.get('learner_last'),stage_candidate=status.get('stage_candidate'),
        frozen_R5=p['best_dev_candidate'],published_policy=None,automatic_next_stage=False)
    text=['# B_keep_coverage independent stage','',f"Execution: {status['phase']}; evaluation: not_started; reliable recovery improvement NOT_ESTABLISHED.",
        '',f"New stage cap {p['budget']['total']}; inherited D2 {p['budget']['prior_charged']}; cumulative maximum {p['budget']['D2_cumulative_maximum']}/2000000. Additional BC2000 separate. Original12h start {p['original_D2_started_unix']}; never reset.",
        '',f"Execution blockers: {status.get('execution_blockers',[])}. New successful keep, collection counts and stage coverage are PENDING until physical collection.",
        '', 'Original pi0/fresh Adam/RNG; Actor only, frozen statistics/critic; no PPO/E/G/TEST. Old source B retained. Prepared code and CPU contracts do not establish physical recovery.']
    outcomes=root/'new_keep_outcomes.json'
    if outcomes.exists():summary['new_keep_collection']=read(outcomes)
    if (root/'keep_receipt.json').exists():summary['merged_keep']=read(root/'keep_receipt.json')
    if status['phase']=='completed':
        rows=read(root/'B_checkpoint_results.json');finish=read(root/'B_completed.json');costs=read(root/'costs.json')
        derived=analyze(p,rows,read(root/'selected_diagnostics.json'),costs,out)
        compact_xy(p,rows,finish,out)
        selected=str(finish['selected_update']);chosen=rows[selected]
        parentroot=Path(read(p['source_B'])['output']);parent_rows=read(parentroot/'B_checkpoint_results.json');parent_finish=read(parentroot/'B_completed.json');parent=parent_rows[str(parent_finish['selected_update'])]
        from .retention_b import full_results
        baseline=full_results(p,Path(p['reused_zero']['DEV']['pi0']))
        repeated={}
        for step,row in rows.items():
            file=root/f'B{int(step):04d}_warning_repeat.json'
            if file.exists():
                pair=read(file);cells={}
                for g in 'ABCD':
                    a,b,ar,br=[np.asarray(pair[k]['labels'][g],bool) for k in ('baseline','original','repeated_pi0','repeated')]
                    cells[g]=dict(N01=int((~ar&br).sum()),N10=int((ar&~br).sum()),net=int(br.sum()-ar.sum()),source_label_flips=int((a!=ar).sum()),student_label_flips=int((b!=br).sum()),both_pair_old_loss=int((a&ar&~b&~br).sum()))
                repeated[step]=dict(warned=pair['warned'],confirmed=pair['confirmed'],cells=cells)
        old_stable=read(parentroot/'report_001/TRAIN_repeat_labels.json')
        old_stable_ids={e['root_id'] for e in old_stable if e['classification']=='stable_success'}
        new_repeat=read(out/'TRAIN_repeat_labels.json')
        lost_absorbed=[e['root_id'] for e in new_repeat if e['root_id'] in old_stable_ids and e['classification']=='stable_failure']
        old_loss=sum(c['N10'] for c in parent['cells'].values());new_loss=sum(c['N10'] for c in chosen['cells'].values())
        summary.update(derived,finish=finish,checkpoints=rows,baseline=baseline,warning_repetitions=repeated,
            learner_last=finish['learner_last'],stage_candidate=finish['candidate'],evaluation_status='complete',
            new_physics=sum(c['charged'] for c in costs),comparison_to_source_B=dict(
                source_selected_update=parent_finish['selected_update'],new_selected_update=finish['selected_update'],
                first_old_loss_before=old_loss,first_old_loss_after=new_loss,
                TRAIN_before=parent['train_success'],TRAIN_after=chosen['train_success'],
                SOLVER_DEV_before=parent['solver_success'],SOLVER_DEV_after=chosen['solver_success'],
                stable_absorbed_roots_now_stable_failed=lost_absorbed,comparison='used development data, conditional on fixed protocol; repeats not independent training seeds'))
        summary['decision']='Retain pi0/R5; stop this stage. If old losses did not decrease or absorbed roots are lost, next investigate TRAIN student-visited states; never automatically start PPO.'
        summary['first_old_loss_decreased']=new_loss<old_loss
        summary['repeat_old_loss_decreased']='UNKNOWN'
        source_rep=parentroot/f"B{parent_finish['selected_update']:04d}_warning_repeat.json"
        if selected in repeated and source_rep.exists():
            sr=read(source_rep);old_repeat=sum(sum(a and not b for a,b in zip(sr['repeated_pi0']['labels'][g],sr['repeated']['labels'][g])) for g in 'ABCD')
            summary['repeat_old_loss_decreased']=sum(c['N10'] for c in repeated[selected]['cells'].values())<old_repeat
        text=['# B_keep_coverage independent development results','',
            '![Actual complete-task XY](xy_DEV_overview.png)','![Seen TRAIN XY](xy_train_roots.png)','![All7 SOLVER_DEV XY](xy_solver_roots.png)',
            '', '|update|TRAIN /8|SOLVER_DEV /7|A /16|B /16|C /16|D /16|new|lost|net|eligible|',
            '|---|---|---|---|---|---|---|---|---|---|---|']
        for k,v in rows.items():
            c=v['cells'];gain=sum(x['N01'] for x in c.values());loss=sum(x['N10'] for x in c.values())
            text.append('|'+ '|'.join(map(str,[k,v['train_success'],v['solver_success'],*[c[g]['success'] for g in 'ABCD'],gain,loss,gain-loss,v['eligible']]))+'|')
        text+=['',f"Selected {selected}, learner_last {finish['completed_updates']}, frozen global candidate R5, published_policy null. Selection unchanged: all7 primary, earliest tie, fixed B/D confirmation.",
            '',f"Source B old loss {old_loss}, new stage {new_loss}; stable absorbed roots now stably failed {lost_absorbed}. Repeat decrease {summary['repeat_old_loss_decreased']}; first-pass and repeat outcomes remain separate.",
            '', 'G teacher reference remains NA on4 solver roots; all7 student outcomes evaluated. TRAIN absorption differs from DEV transfer; used DEV is never final TEST. Collection success/failure/ambiguity, phase coverage, weights and actual worker input provenance in summary. Actor gradients are actual same-batch contributions, fixed probes separate.',
            '',summary['decision'], '', 'This finite development comparison does not establish reliable recovery improvement or eliminate forgetting. No next stage automatically launches.']
    write(out/'summary.json',summary);(out/'INDEX.md').write_text('\n'.join(text)+'\n');return out
