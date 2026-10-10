"""Saved-data-only paired confirmation and teacher conversion reports."""
from pathlib import Path
import csv
import numpy as np
from .retention_repair import read,write,sha
from .retention_repair_report import CachedArchive,rows_for_batch,liftoff_tick


def xy_plot(plan,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(plan['output']);qi=plan['indices']['root_qpos'];fig,axes=plt.subplots(1,4,figsize=(18,5))
    for b in plan['batches']:
        if not (root/b['name']/'verification.json').exists():continue
        with CachedArchive(root/b['name']/'prefixes.npz') as f:
            for lane,c in enumerate(b['cases']):
                if c['condition_index']%16:continue
                n=int(f['prefix_mask'][:,lane].sum());q=f['qpos'][:n,lane];ax=axes['ABCD'.index(c['group'])]
                ax.plot(q[:,qi],q[:,qi+1],c='black' if b['model']=='pi0' else 'tab:blue',alpha=.35,lw=.8,label=b['model'])
                if not f['success'][:n,lane].any():ax.scatter(q[-1,qi],q[-1,qi+1],s=5,marker='x',c='black' if b['model']=='pi0' else 'tab:blue')
    for g,ax in zip('ABCD',axes):
        ax.set(title=g,xlabel='World x (m)',ylabel='World y (m)');ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
        h,l=ax.get_legend_handles_labels();d=dict(zip(l,h));ax.legend(d.values(),d.keys())
    fig.suptitle('V independent DEV_CONFIRM: pi0 and frozen R5; every16th condition shown\nActual terminal endpoints; unchanged jump task has no prescribed XY path')
    fig.tight_layout();fig.savefig(output/'xy_four_cells.png',dpi=160);plt.close(fig)


def report_v(plan,out):
    root=Path(plan['output']);rows=[];distributions={}
    for b in plan['batches']:
        if not (root/b['name']/'verification.json').exists():raise ValueError('incomplete V; report cannot evaluate missing batches')
        result=rows_for_batch(plan,b)
        for r in result:r['role']='DEV_CONFIRM'
        rows.extend(result)
        with CachedArchive(root/b['name']/'prefixes.npz') as f:
            for group in 'ABCD':
                lanes=[i for i,c in enumerate(b['cases']) if c['group']==group]
                if not lanes:continue
                mask=f['mask'][:,lanes]&f['prefix_mask'][:,lanes]
                key=b['model']+'_'+group
                d=distributions.setdefault(key,{'requested':[],'effective':[]})
                for field in ('requested','effective'):d[field].append(f[field+'_delta'][:,lanes][mask])
    for k,d in distributions.items():
        for field,values in d.items():
            a=np.concatenate(values)
            d[field]=dict(actions=len(a),mean=a.mean(0).tolist() if len(a) else None,
                quantiles=np.quantile(a,[0,.05,.5,.95,1],axis=0).tolist() if len(a) else None)
    baseline={r['case']:r for r in rows if r['model']=='pi0'};four={};strata=[]
    for model in plan['models']:
        four[model]={}
        for g in 'ABCD':
            rr=[r for r in rows if r['model']==model and r['group']==g]
            four[model][g]=dict(n=len(rr),successes=sum(r['success'] for r in rr),
                N01_new_vs_pi0=sum(not baseline[r['case']]['success'] and r['success'] for r in rr),
                N10_lost_vs_pi0=sum(baseline[r['case']]['success'] and not r['success'] for r in rr),
                first_failure_phases={p:sum(r['first_physical_failure_phase']==p for r in rr) for p in ('pre_liftoff','airborne_before_valid_contact','post_contact',None)})
            for onset in (0,5,10,15):
                sr=[r for r in rr if r['onset']==onset]
                if sr:strata.append(dict(model=model,cell=g,onset=onset,n=len(sr),successes=sum(r['success'] for r in sr),
                    N01=sum(not baseline[r['case']]['success'] and r['success'] for r in sr),N10=sum(baseline[r['case']]['success'] and not r['success'] for r in sr)))
    # Shared B/D initials and C/D requests resample together. A is one nominal condition.
    paired={r['case']:r for r in rows if r['model']=='R5'}
    diffs=np.array([[int(paired[f'{g}_{i:03d}']['success'])-int(baseline[f'{g}_{i:03d}']['success']) for g in 'BCD'] for i in range(256)])
    rng=np.random.default_rng(plan['config']['V_execution_seed']);boot=[]
    for _ in range(2000):
        ids=np.concatenate([rng.choice(np.arange(o*64,(o+1)*64),64,replace=True) for o in range(4)])
        boot.append(diffs[ids].mean(0))
    ci=np.quantile(boot,[.025,.975],axis=0)
    summary=dict(stage='V',four_cells=four,onset_strata=strata,paired_cluster_ci={g:dict(net_rate=float(diffs[:,i].mean()),ci95=ci[:,i].tolist()) for i,g in enumerate('BCD')},
        bootstrap='2000 stratified draws of paired condition index across B/C/D; conditional on fixed Actors and protocol, no training seed inference',
        A='32 nominal repeats; one unique physical initial state; excluded from unique capability aggregate',
        physical_charged=sum(read(root/b['name']/'status.json')['charged_interactions'] for b in plan['batches']),
        request_effective_distributions=distributions,updates=0,teachers='NOT_RUN',published_policy=None,
        best_dev_candidate=plan['models']['R5']['policy'],R5_confirmation='per-cell evidence only; no automatic publication')
    write(out/'summary.json',summary);write(out/'episodes.json',rows)
    with (out/'episodes.csv').open('x') as f:
        keys=[k for k in rows[0] if k!='reward_components'];w=csv.DictWriter(f,keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in rows)
    xy_plot(plan,out)
    text='# Frozen R5 independent confirmation\n\n![Actual XY](xy_four_cells.png)\n\nOnly pi0/R5, own Actor+normalizer; DEV_CONFIRM never TRAIN.\n\n|Model|A nominal no pulse|B random no pulse|C nominal pulse|D random pulse|\n|---|---|---|---|---|\n'
    for model,g in four.items():text+='|'+model+'|'+'|'.join(f"{g[x]['successes']}/{g[x]['n']} (new {g[x]['N01_new_vs_pi0']}, lost {g[x]['N10_lost_vs_pi0']})" for x in 'ABCD')+'|\n'
    text+=f"\nCharged physics {summary['physical_charged']} / {plan['budget']['hard_cap']}; 0 updates. R5 remains best_dev_candidate; published_policy=null. A contains nominal numerical repeats, not32 independent starts. B/D and C/D are paired. Stratified paired intervals, onset results, effective clipping and first failure phases: [summary](summary.json), [episodes](episodes.csv).\n"
    (out/'INDEX.md').write_text(text);(out/'analysis_update.md').write_text(text)
    return out/'INDEX.md'


def trace_anomalies(attempt,context=None):
    with CachedArchive(attempt['trace']) as f:
        lane=attempt['trace_lane'];mask=f['mask'][:,lane];n=int(mask.sum());first={}
        offset=0;front=f['front_wheel_clearance'][:n,lane];rear=f['rear_wheel_clearance'][:n,lane]
        if context is not None:
            offset=context['snapshot_control_step']
            with CachedArchive(Path(context['collection'])/'prefixes.npz') as prefix:
                front=np.concatenate((prefix['front_wheel_clearance'][:offset,context['collection_lane']],front))
                rear=np.concatenate((prefix['rear_wheel_clearance'][:offset,context['collection_lane']],rear))
        liftoff=liftoff_tick(front,rear)
        for name in ('roll_limit','prohibited_contact','physical_failure','first_valid_contact'):
            ids=np.flatnonzero(f[name][:,lane]&mask) if name in f.files else []
            tick=int(ids[0]) if len(ids) else None
            phase=None if tick is None else 'post_contact' if bool(f['valid_contact_seen_before'][tick,lane]) else 'airborne_before_valid_contact' if liftoff is not None and tick+offset>=liftoff else 'pre_liftoff_or_no_ground_in_suffix'
            first[name]=dict(suffix_tick=tick,original_episode_tick=None if tick is None else tick+offset,phase=phase,substep='UNKNOWN; control-step trace only')
        return dict(steps=n,label=attempt['label'],first=first,end_code=int(f['end_code'][n-1,lane]))


def d1_xy(plan,selected,results,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    qi=plan['indices']['root_qpos'];by={r['root_id']:r for r in results}
    fig,axes=plt.subplots(4,6,figsize=(24,16))
    for ax,r in zip(axes.flat,selected):
        with CachedArchive(Path(r['collection'])/'prefixes.npz') as f:
            lane=r['collection_lane'];n=r['snapshot_control_step'];q=f['qpos'][:n,lane]
            ax.plot(q[:,qi],q[:,qi+1],color='gray',label='pi0+E prefix');ax.scatter(q[-1,qi],q[-1,qi+1],s=12,color='gray')
        attempts=[('pi0',r['pi0_attempt'])]
        for method,m in by[r['root_id']]['methods'].items():
            if m['verified_attempts']:attempts.append((method+' fixed winner',m['verified_attempts'][-1]))
        for label,a in attempts:
            with CachedArchive(a['trace']) as f:
                lane=a['trace_lane'];mask=f['mask'][:,lane];q=f['qpos'][mask,lane]
                ax.plot(q[:,qi],q[:,qi+1],lw=1,label=label)
                ax.scatter(q[-1,qi],q[-1,qi+1],s=10,marker='o' if a['label']==1 else 'x')
        ax.set(title=f"{r['role']} {r['logical_episode_id']} onset{r['onset']}\n{by[r['root_id']]['conversion_cell']}",xlabel='World x (m)',ylabel='World y (m)');ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2);ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(selected):]:ax.set_visible(False)
    fig.suptitle('D1 all selected roots: actual XY prefix and independent continuations; endpoints at actual success/failure\nNo prescribed XY path; missing winner means search found none; source ambiguity remains quarantined')
    fig.tight_layout();fig.savefig(out/'xy_teacher_roots.png',dpi=120);plt.close(fig)


def report_d1(plan,out):
    root=Path(plan['output']);receipt=read(root/'D1_completed.json')
    if receipt['phase']!='completed' or receipt['teacher_results_sha256']!=sha(root/'teacher_results.json'):raise ValueError('D1 incomplete or changed')
    results=read(root/'teacher_results.json');ledger=read(root/'collection_ledger.json');selected=read(root/'selected_roots.json')
    matrix={role:{k:sum(r['role']==role and r['conversion_cell']==k for r in results) for k in ('G_only','Noise_only','both','neither','ambiguous')} for role in ('TRAIN','SOLVER_DEV')}
    qualified=[r for r in results if r['qualified_G_incremental_lesson']]
    gate=len(set(r['root_episode_id'] for r in qualified))>=plan['config']['D2_min_distinct_train'] and len(set(r['onset'] for r in qualified))>=plan['config']['D2_min_onsets']
    dataset=[]
    for r in qualified:
        m=r['methods']['G'];a=m['verified_attempts'][-1]
        if sha(a['trace'])!=a['trace_sha256']:raise ValueError('teacher verified trace changed')
        with CachedArchive(a['trace']) as f:
            lane=a['trace_lane'];mask=f['mask'][:,lane];obs=f['actor_observation_before'][mask,lane];actions=f['normalized_action_executed'][mask,lane]
            origins=f['action_origin_code'][mask,lane]
        path=out/f'lesson_{len(dataset):03d}.npz';np.savez_compressed(path,observations=obs,actions=actions,origins=origins)
        dataset.append(dict(root_episode_id=r['root_episode_id'],root_id=r['root_id'],onset=r['onset'],role='TRAIN',path=str(path),sha256=sha(path),
            prefix_label='G_H16',suffix_label='pi0_tail',source_labels=r['pi0_labels'],source_batch_labels=m['source_labels'],
            selected_candidate_id=m['selected_candidate_id'],winner_labels=m['winner_labels'],trace=a,tail=r['tail'],snapshot_context_sha256=r['snapshot_context_sha256']))
    manifest=dict(schema='jit_retention_teacher_dataset_v1',TRAIN=dataset,SOLVER_DEV=[r['root_id'] for r in results if r['role']=='SOLVER_DEV'],
        source_lock_sha256=sha(root/'source_lock.json'),D1_receipt=str(root/'D1_completed.json'),D1_receipt_sha256=sha(root/'D1_completed.json'),
        solver_dev_training_allowed=False,G_updates=0,student_updates=0,qualified_gate=gate)
    write(out/'teacher_dataset_manifest.json',manifest)
    contexts={r['root_id']:r for r in selected}
    anomalies={r['root_id']:{method:[trace_anomalies(a,contexts[r['root_id']]) for a in m['verified_attempts']] for method,m in r['methods'].items()} for r in results}
    for row in selected:anomalies[row['root_id']]['independent_pi0']=[trace_anomalies(row['pi0_attempt'],row)]
    write(out/'teacher_anomalies.json',anomalies)
    d1_xy(plan,selected,results,out)
    costs=read(root/'costs.json');summary=dict(stage='D1',teacher_conversion_matrix=matrix,qualified_incremental_lessons=len(dataset),
        distinct_TRAIN_ancestors=len(set(r['root_episode_id'] for r in qualified)),qualified_onsets=sorted(set(r['onset'] for r in qualified)),D2_preparation_gate=gate,
        collection_denominator=len(ledger),collection_outcomes={x:sum(r['pulse_outcome']==x for r in ledger) for x in ('pre_pulse_terminal','during_pulse_terminal','valid_post_pulse')},
        selected_train=sum(r['role']=='TRAIN' for r in selected),selected_solver_dev=sum(r['role']=='SOLVER_DEV' for r in selected),
        budget_charged=sum(c['charged_interactions'] for c in costs),budget_hard_cap=plan['budget']['hard_cap'],updates=dict(student=0,E=0,G=0),
        student_absorption='NOT_RUN',student_old_capability_loss='NOT_RUN',student_independent_success='NOT_RUN',
        G_matches_pi0_tail='finite root-specific evidence only; normalization-reference identity is not physical validation',
        D0_new87='different full-task DEV paired gain; never teacher conversion',published_policy=None)
    write(out/'summary.json',summary);write(out/'teacher_conversion_matrix.json',matrix)
    # Read-only source/R5 losses on TRAIN snapshots remain distinct from full independent performance.
    write(out/'R5_snapshot_contrast.json',dict(n=len(selected),pi0_fail_R5_success=sum(r['R5_label']==1 for r in selected),
        interpretation='R5 continuation after pi0+E prefix; not independent complete R5 success'))
    d2='Prepare a separate D2 plan; no automatic execution.' if gate else 'D2 preparation gate failed: do not start students or increase perturbation strength.'
    text=f'# D1 fixed teachers\n\n![Actual XY all roots](xy_teacher_roots.png)\n\n0 student/E/G updates. Charged {summary["budget_charged"]}/{summary["budget_hard_cap"]}.\n\n|Role|G only|Noise only|Both|Neither|Ambiguous|\n|---|---|---|---|---|---|\n'
    for role,row in matrix.items():text+='|'+role+'|'+'|'.join(str(row[k]) for k in ('G_only','Noise_only','both','neither','ambiguous'))+'|\n'
    text+=f'\nQualified G incremental lessons: {len(dataset)} distinct TRAIN ancestors, onsets {summary["qualified_onsets"]}. {d2}\n\nTeachers are fixed17-world search +2 original-batch replays with one frozen winner; source ambiguity is quarantined. Every selected baseline has3 independent pi0 labels before teacher searches. Complete collection denominator {len(ledger)}, outcomes {summary["collection_outcomes"]}. No DEV failures entered TRAIN.\n\nStudent absorption, student retention loss and independent student success: NOT_RUN. D0 gained87 is not a teacher conversion. G and Noise evidence does not establish diffusion necessity.\n\n[Full matrix](teacher_conversion_matrix.json), [qualified dataset](teacher_dataset_manifest.json), [first anomalies](teacher_anomalies.json), [summary](summary.json), [raw root results](../teacher_results.json).\n'
    (out/'INDEX.md').write_text(text);(out/'analysis_update.md').write_text(text)
    return out/'INDEX.md'


def report(path,output=None):
    plan=read(path);out=Path(output or Path(plan['output'])/'report_001');out.mkdir(parents=True,exist_ok=False)
    return report_v(plan,out) if plan['stage']=='V' else report_d1(plan,out)


def prepare_d2(evidence_dir,output,repository):
    from .retention_d2 import prepare
    return prepare(evidence_dir,output,repository)
