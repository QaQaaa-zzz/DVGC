"""Finite short-pulse discovery with bank evaluation and delayed learning quality."""
from pathlib import Path
import csv
import json
import time
import numpy as np
from .exploration_loop import read,write
from .probe_bank import load_probe_bank,lock_probe_bank,_file_sha
from .evidence_integrity import canonical_sha256


class InteractionBudgetExhausted(RuntimeError):
    """A complete next stage would exceed the declared finite cap."""


def reserve_interactions(used, maximum, cap):
    if any(type(v) is not int or v < 0 for v in (used, maximum, cap)):
        raise ValueError('interaction reservations require nonnegative integers')
    if used + maximum > cap:
        raise InteractionBudgetExhausted(f'{used} used + {maximum} next stage > {cap} cap')


def promoted_explorer_checkpoint(spec, checkpoint):
    # Historical runs may explicitly retain their old reset contract.
    return checkpoint if spec.get('explorer_backend')=='rsl_rl' or spec.get('continuous_explorer',False) else None


def round_delta_limit(spec, index):
    schedule=spec.get('delta_limit_schedule')
    values=schedule[index % len(schedule)] if schedule else spec['delta_limit']
    if len(values)!=4 or any(not np.isfinite(x) or not 0<=x<=1 for x in values):
        raise ValueError('four finite residual limits in [0,1] required')
    return list(values)


def prefix_budget(spec, index):
    from .pulse_schedule import lane_onsets, selected_event
    selected_event(spec)
    ticks=spec['horizon'] if spec.get('pulse_event_schedule') else int(lane_onsets(spec,index).max())+spec['pulse_steps']
    return spec['num_envs']*ticks


def seed_support_budget(spec):
    horizon=spec['horizon'];stride=spec.get('seed_stride',2)
    if type(stride) is not int or stride<1:raise ValueError('positive seed stride required')
    # At most ceil(horizon/stride) snapshots, each with a horizon-long suffix.
    return horizon + ((horizon+stride-1)//stride)*horizon


def pulse_feedback(rows,seen,weights,*,quality_mode='delayed'):
    """Newly visited cells stay visited after failure; unknown is not punished."""
    counts={};old=set(seen)
    if quality_mode not in ('delayed','current_policy','novelty_only'):
        raise ValueError('unsupported exploration quality mode')
    for r in rows:
        if not r.get('stage_reached',True):continue
        if r['cell'] not in old:counts[r['cell']]=counts.get(r['cell'],0)+1
    novelty=np.zeros(len(rows));quality=np.zeros(len(rows));mask=np.zeros(len(rows),bool)
    for i,r in enumerate(rows):
        label=r.get('initial_label',r['label']) if quality_mode=='current_policy' else r['label']
        mask[i]=r.get('stage_reached',True) and (label==1 or (label==0 and
            (quality_mode=='current_policy' or r['learning_attempted'] or r.get('prefix_terminal',False))))
        if mask[i]:
            novelty[i]=weights['novelty']/counts[r['cell']] if r['cell'] in counts else 0
            direct_failure=(r.get('prefix_terminal',False) and r.get('prefix_physical_failure',False)
                            and r.get('pulse_applied_steps',0)>0)
            penalty=weights.get('pulse_failure',weights['failure']) if direct_failure else weights['failure']
            quality[i]=0 if quality_mode=='novelty_only' else (weights['success'] if label==1 else -penalty)
    return novelty+quality,mask,sorted(old|{r['cell'] for r in rows if r.get('stage_reached',True)}),dict(novelty=novelty.tolist(),quality=quality.tolist())


def pulse_delay(spec,index):
    if spec.get('pulse_event_schedule') and not spec.get('nominal_source_rollout'):
        return -1  # An event-triggered pulse has no fixed global onset tick.
    schedule=spec.get('pulse_start_schedule',[0])
    if spec.get('nominal_source_rollout') is True:
        if (schedule!=[0] or spec['pulse_steps']!=spec['horizon'] or spec['num_envs']!=1
                or spec['delta_limit']!=[0.,0.,0.,0.] or spec.get('explorer_checkpoint')):
            raise ValueError('nominal rollout must be one full-horizon zero-residual source trajectory')
        return 0
    if not schedule or any(type(t) is not int or t<0 or t+spec['pulse_steps']>=spec['horizon'] for t in schedule):
        raise ValueError('invalid pulse start schedule')
    return schedule[index%len(schedule)]


def budget_contract(spec,evaluators):
    rounds=spec['rounds'];n=spec['num_envs'];h=spec['horizon']
    if any(type(spec[k]) is not int or spec[k]<=0 for k in ['rounds','num_envs','horizon','pulse_steps','policy_steps']):raise ValueError('positive pulse budgets required')
    if spec['pulse_steps']>5 or h!=400 or spec['policy_steps']%3200:raise ValueError('short-pulse/horizon/aligned learning contract')
    baseline=evaluators*h
    prefixes=sum(prefix_budget(spec,i) for i in range(rounds))
    mode=spec.get('training_mode','adaptive')
    if mode not in ('adaptive','fixed_policy'):raise ValueError('unsupported policy training mode')
    fixed=mode=='fixed_policy'
    current_only=spec.get('iteration_mode')=='current_policy_only_v1'
    if spec.get('iteration_mode') not in (None,'current_policy_only_v1'):
        raise ValueError('unsupported pulse iteration mode')
    if current_only and (type(spec.get('retention_samples_per_phase')) is not int or
            spec['retention_samples_per_phase']<=0 or not 0<=spec['minimum_retention']<=1):
        raise ValueError('valid current-policy retention panel required')
    suffixes=rounds*n*h if current_only else sum((evaluators+i)*n*h for i in range(rounds))
    learning=0 if fixed else rounds*(spec['policy_steps']+1600+n*h)
    extra={}
    if current_only:
        extra=dict(nominal_support=seed_support_budget(spec) if fixed else (rounds+1)*h*(h+1),
            retention_evaluation=0 if fixed else rounds*2*(1+2*spec['retention_samples_per_phase'])*h)
    return dict(baseline=baseline,prefixes=prefixes,bank_suffixes=suffixes,learning_and_reevaluation=learning,
        **extra,maximum_interactions=baseline+prefixes+suffixes+learning+sum(extra.values()))


def support_row(row,witnessed):
    return dict(key=row['snapshot_context_sha256'],phase=row['phase'],snapshot=row['snapshot'],state_sha256=row['state_sha256'],snapshot_context_sha256=row['snapshot_context_sha256'],trajectory_id=row['prefix_file']+'::'+str(row['index']),root_cell=row['cell'],witnessed=witnessed,evidence_status='witnessed' if witnessed else 'pending',role='train',sampling_weight=1.)


def plot_completed_rounds(root, rows):
    """An empty successful set is an experimental result, not a plot failure."""
    destination=Path(root)/'analysis/tube_xz'
    if not any(row['label']==1 for row in rows):
        destination.mkdir(parents=True,exist_ok=False)
        write(destination/'status.json',dict(phase='completed',successful_witnesses=0,
            reason='no_successful_witnesses',new_simulation_interactions=0))
        (destination/'INDEX.md').write_text('# x-z projection\n\nNo successful witnesses in completed rounds. No successful tube is drawn. See the retained candidates and outcomes.\n')
        return
    from .analysis.pulse_tube_xz import build
    build([root],destination)


def export(root,metrics,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name,items in [('training_process',metrics),('regions',[dict(cell=r['cell'],state_sha256=r['state_sha256'],context=r['snapshot_context_sha256'],status='successful' if r['label']==1 else ('unresolved_after_learning' if r['label']==0 and r['learning_attempted'] else 'pending'),snapshot=r['snapshot'],witness=r.get('witness'),round=r['round'],pulse_start_step=r.get('pulse_start_step',0)) for r in rows])]:
        if not items:continue
        with (root/(name+'.csv')).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in items for k in r)));w.writeheader();w.writerows(items)
    if not metrics:return
    fig,ax=plt.subplots(2,3,figsize=(14,7));x=[r['round'] for r in metrics]
    for k in ['actor_loss','critic_loss']:ax[0,0].plot(x,[r[k] for r in metrics],label=k)
    for k in ['reward_novelty','reward_quality']:ax[0,1].plot(x,[r[k] for r in metrics],label=k)
    ax[0,2].plot(x,[r['post_update_kl'] for r in metrics],label='post-update KL')
    for k in ['successes','failed_after_learning','new_cells']:ax[1,0].plot(x,[r[k] for r in metrics],label=k)
    ax[1,1].plot(x,[r['charged_interactions'] for r in metrics],label='cumulative charged steps')
    for label,color in [(1,'green'),(0,'red'),(None,'gray')]:
        selected=[r for r in rows if r['label']==label]
        ax[1,2].scatter([r['coordinates']['pitch_deg'] for r in selected],[r['coordinates']['root_vz_mps'] for r in selected],s=6,c=color,label=str(label),alpha=.4)
    ax[1,2].set(xlabel='handoff pitch (deg)',ylabel='handoff vertical velocity (m/s)')
    for a in ax.flat:a.legend();a.grid(alpha=.2)
    fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(root/('training_process.'+ext))
    plt.close(fig)


def neighborhood_seed_history(root):
    """Recover prior nominal map evidence without reading future round outcomes."""
    root=Path(root);paths=[root/'seed_support/evaluation/results.json']
    paths.extend(sorted(root.glob('round_*/next_source_seed/evaluation/results.json')))
    return [row for p in paths if p.exists() for row in read(p)]


def run(spec_path,output):
    from .gated_execution import run_gated_plan
    from .iterative_probe_training import candidate_support_view,make_config
    from .unified_policy_freeze import freeze_development_checkpoint
    spec=read(spec_path);root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    for path,expected in {**spec['input_files'],**spec['source_locks']}.items():
        if _file_sha(Path(path))!=expected:raise ValueError('pulse input/source drift: '+path)
    bank=load_probe_bank(Path(spec['bank']));bank_path=Path(spec['bank']);count=len(spec['order']);budget=budget_contract(spec,count)
    if spec['maximum_interactions']!=budget['maximum_interactions']:raise ValueError('pulse budget mismatch')
    actual_cap=spec.get('maximum_actual_interactions',budget['maximum_interactions'])
    reserve_interactions(0,0,actual_cap)
    fixed_policy=spec.get('training_mode','adaptive')=='fixed_policy'
    quality_mode=spec.get('quality_mode','delayed')
    if fixed_policy and quality_mode not in ('current_policy','novelty_only'):
        raise ValueError('fixed-policy exploration must explicitly declare current-policy quality or novelty only')
    source=next(m for m in bank['members'] if m['name']==spec['proposer'])
    current_only=spec.get('iteration_mode')=='current_policy_only_v1'
    if current_only:
        from .current_policy_iteration import validate_initial_bank,retention_candidates,promotion_decision
        validate_initial_bank(bank,spec['proposer'])
        if spec['order']!=[spec['proposer']] or any(spec.get(k) for k in
                ('resume_run','reuse_results','reuse_collection','reuse_prefix_collection','witnessed_support')):
            raise ValueError('clean source lineage cannot import historical support or replay')
    initial_bank_names={m['name'] for m in bank['members']}
    base=None
    if not current_only:
        base=read(spec['witnessed_support']);base['entries']=[r for r in base['entries'] if r.get('labels',{}).get(spec['proposer'])==1]
        if {r['phase'] for r in base['entries']}!={'upstream','downstream'}:raise ValueError('source pi needs witnessed support in both phases')
        base.pop('support_sha256');base['selection']='only frozen source-pi positive rows';base['support_sha256']=canonical_sha256(base)
        write(root/'source_pi_support.json',base)
    write(root/'declaration.json',dict(spec=spec,budget=budget,role='TRAIN',final_test_used=False))
    seen=sorted({r['root_cell'] for r in base['entries']}) if base else [];rows_all=[];neighborhood_seeds=[];metrics=[];costs=[];checkpoint=None;start=time.monotonic();support=json.loads(json.dumps(base));inherited_cost=0
    boundary=read(spec['resume_boundary']) if spec.get('resume_boundary') else None
    if boundary:
        from .current_policy_iteration import verify_stage_reuse
        verify_stage_reuse(read(Path(boundary['previous'])/'declaration.json')['spec'],spec)
        inherited_cost=boundary['charged_interactions']
    reuse_root=Path(spec['resume_stage_root']) if spec.get('resume_stage_root') else None
    if reuse_root is not None:
        if not current_only:raise ValueError('stage recovery is restricted to current-policy runs')
        from .current_policy_iteration import verify_stage_reuse
        verify_stage_reuse(read(reuse_root/'declaration.json')['spec'],spec)
        previous_status=read(reuse_root/'status.json')
        if any(c['accounting']!='actual' for c in previous_status['costs']):
            raise ValueError('stage recovery needs measured prior costs')
        inherited_cost=previous_status['charged_interactions']
        write(root/'recovery.json',dict(previous=str(reuse_root),inherited_interactions=inherited_cost,
            previous_status_sha256=_file_sha(reuse_root/'status.json'),no_completed_training_repeated=True))
    def status(phase,**kw):write(root/'status.json',dict(phase=phase,charged_interactions=inherited_cost+sum(c['charged_interactions'] for c in costs),new_interactions=sum(c['charged_interactions'] for c in costs),inherited_interactions=inherited_cost,maximum_interactions=budget['maximum_interactions'],maximum_actual_interactions=actual_cap,wall_seconds=time.monotonic()-start,costs=costs,**kw))
    def child(directory,name,argv,maximum,env=None):
        reserve_interactions(inherited_cost+sum(c['charged_interactions'] for c in costs),maximum,actual_cap)
        inputs={str((Path(spec['repo'])/p).resolve()):sha for p,sha in spec['input_files'].items()}
        inputs[str(Path(spec_path).resolve())]=_file_sha(Path(spec_path))
        for i,a in enumerate(argv[:-1]):
            if a in ['--spec','--config']:inputs[str(Path(argv[i+1]).resolve())]=_file_sha(Path(argv[i+1]))
        plan=dict(schema='jit_gated_plan_v1',gate=spec['gate'],input_files=inputs,source_locks=spec['source_locks'],max_interactions=max(1,maximum),wait_timeout_seconds=spec['wait_timeout_seconds'],stages=[dict(name=name,argv=[spec['python'],*map(str,argv)],cwd=spec['repo'],env=dict(JAX_PLATFORMS='cuda,cpu',CUDA_VISIBLE_DEVICES='0',PYTHONPATH=str(Path(spec['repo'])/'JIT/src'),XLA_PYTHON_CLIENT_PREALLOCATE='false',JIT_AUTO_PUBLISH='0',**(env or {})),timeout_seconds=spec['stage_timeout_seconds'],max_interactions=maximum)])
        path=directory/(name+'_plan.json');write(path,plan)
        cost=dict(stage=str(directory.relative_to(root))+'/'+name,charged_interactions=0,maximum_interactions=maximum,accounting='not_launched');costs.append(cost);status('running',stage=cost['stage'])
        result=run_gated_plan(path,directory/(name+'_execution'),wait=True,poll_seconds=30)
        if result['phase']!='completed':
            if result['reserved_interactions']>0:cost.update(charged_interactions=maximum,accounting='reserved_after_incomplete_child')
            raise RuntimeError('pulse child stopped: '+name+' '+result['phase'])
        return cost
    def runtime(directory,name,mode,args,maximum):
        if reuse_root is not None:
            prior=reuse_root/directory.relative_to(root)/name
            if (prior/'status.json').exists() and read(prior/'status.json')['phase']=='completed':
                verify_stage_reuse(read(prior.parent/(name+'_spec.json')),args)
                write(directory/(name+'_reuse.json'),dict(previous=str(prior),
                    receipt_sha256=_file_sha(prior/'status.json'),new_interactions=0))
                return prior
        p=directory/(name+'_spec.json');write(p,args);out=directory/name
        cost=child(directory,name,['JIT/cli/run_pulse_exploration.py','--mode',mode,'--spec',p,'--output',out],maximum)
        result=read(out/'status.json');actual=result['charged_interactions']
        if result['phase']!='completed' or not 0<=actual<=maximum:raise ValueError('runtime receipt/budget mismatch')
        cost.update(charged_interactions=actual,accounting='actual');return out
    try:
        if current_only and not boundary:
            seed=runtime(root,'seed_support','seed_support',spec,
                         seed_support_budget(spec) if fixed_policy else spec['horizon']*(spec['horizon']+1))
            base=read(seed/'support.json');support=json.loads(json.dumps(base))
            if spec.get('neighborhood'):neighborhood_seeds.extend(read(seed/'evaluation/results.json'))
            seen=sorted({r['root_cell'] for r in base['entries']})
            write(root/'source_pi_support.json',base)
        first_round=0
        if boundary:
            previous=Path(boundary['previous'])
            if spec.get('neighborhood'):neighborhood_seeds.extend(neighborhood_seed_history(previous))
            metrics=read(previous/'training_metrics.json');first_round=len(metrics)
            if first_round != boundary['completed_rounds']:raise ValueError('boundary round count drift')
            bank_path=Path(boundary['bank']);bank=load_probe_bank(bank_path)
            source=next(m for m in bank['members'] if m['name']==boundary['source'])
            checkpoint=boundary['explorer_checkpoint'];support=read(boundary['support'])
            seen=read(previous/'visited_cells.json')
            rows_all=[row for i in range(first_round) for row in read(previous/f'round_{i:04d}'/'outcomes.json')]
            write(root/'source_pi_support.json',read(previous/'source_pi_support.json'))
            write(root/'training_metrics.json',metrics);export(root,metrics,rows_all)
            write(root/'recovery.json',boundary)
        elif spec.get('resume_run'):
            previous=Path(spec['resume_run']);old=read(previous/'declaration.json')['spec']
            contract=['proposer','order','num_envs','pulse_steps','delta_limit','horizon','policy_steps','pending_fraction','seed','learning_rate','minibatch_size','epochs','clip','target_kl','reward_weights','value_coefficient','entropy_coefficient','max_grad_norm','jump_start_state_sha256']
            if old.get('pulse_start_schedule',[0])!=spec.get('pulse_start_schedule',[0]):raise ValueError('resume pulse schedule differs')
            if any(old[k]!=spec[k] for k in contract):raise ValueError('resumed pulse training contract differs')
            old_bank=load_probe_bank(Path(old['bank']))
            for name in spec['order']:
                if next(m['policy'] for m in old_bank['members'] if m['name']==name)!=next(m['policy'] for m in bank['members'] if m['name']==name):raise ValueError('resumed source bank policy differs')
            metrics=read(previous/'training_metrics.json');first_round=len(metrics)
            if first_round<1 or first_round>=spec['rounds']:raise ValueError('no resumable completed rounds')
            prior=previous/f'round_{first_round-1:04d}'
            checkpoint=str(prior/'update/state.msgpack');support=read(prior/'witnessed_support.json');seen=read(previous/'visited_cells.json')
            for i in range(first_round):
                ancestor=previous;visited=set()
                while not (ancestor/f'round_{i:04d}'/'outcomes.json').exists():
                    if str(ancestor) in visited:raise ValueError('resume ancestry cycle')
                    visited.add(str(ancestor));ancestor=Path(read(ancestor/'resume.json')['previous'])
                rows_all.extend(read(ancestor/f'round_{i:04d}'/'outcomes.json'))
            if any((previous/f'round_{i:04d}'/'expanded_bank.json').exists() for i in range(first_round)):raise ValueError('resume requires explicit expanded bank import')
            inherited_cost=read(spec['inherited_cost_receipt'])['actual_interactions'] if spec.get('inherited_cost_receipt') else read(previous/'status.json')['charged_interactions']
            write(root/'resume.json',dict(previous=str(previous),completed_rounds=first_round,checkpoint=checkpoint,checkpoint_sha256=_file_sha(Path(checkpoint)),inherited_interactions=inherited_cost))
            export(root,metrics,rows_all)
        else:
            runtime(root,'baseline','baseline',spec,count*spec['horizon'])
        for index in range(first_round,spec['rounds']):
            # Reserve enough for a complete collection/evaluation pair before
            # sampling a new round; no candidate disappears because of a cap.
            reserve_interactions(inherited_cost+sum(c['charged_interactions'] for c in costs),
                prefix_budget(spec,index)+count*spec['num_envs']*spec['horizon'],actual_cap)
            d=root/f'round_{index:04d}';d.mkdir()
            source_name=source['name'];adopt=None
            ex={**spec,'bank':str(bank_path),'round_index':index,'explorer_checkpoint':checkpoint,
                'delta_limit':round_delta_limit(spec,index),
                'proposer':source_name if current_only else spec['proposer']}
            if spec.get('neighborhood'):
                # Snapshot only already observed rows; this round never queries its future outcomes.
                actor=source['policy']['actor_sha256']
                history=[r for r in neighborhood_seeds+rows_all
                    if r.get('source_actor_sha256',(r.get('attempts') or [{}])[0].get('actor_sha256'))==actor]
                map_payload=dict(schema='jit_frozen_neighborhood_map_v1',source_actor_sha256=actor,
                    round_index=index,config=spec['neighborhood'],rows=history)
                map_path=d/'neighborhood_map.json';write(map_path,map_payload)
                ex.update(neighborhood_map=str(map_path),neighborhood_map_sha256=_file_sha(map_path))
            if index!=first_round:ex.pop('reuse_prefix_collection',None)
            if index==first_round and spec.get('reuse_collection'):
                collection=Path(spec['reuse_collection'])
                old=read(collection/'hyperparameters.json')
                for k in ['round_index','explorer_checkpoint','pulse_steps','num_envs','delta_limit','bank','seed','pulse_batch_mode','neighborhood','neighborhood_map_sha256']:
                    if old.get(k)!=ex.get(k):raise ValueError('reused pulse contract differs: '+k)
                if read(collection/'status.json')['phase']!='completed':raise ValueError('incomplete reused collection')
                write(d/'reused_collection.json',dict(path=str(collection),prefix_sha256=_file_sha(collection/'prefixes.npz')))
            else:
                collection=runtime(d,'collection','collect',ex,prefix_budget(spec,index))
            order=[source_name] if current_only else spec['order']+[m['name'] for m in bank['members'] if m['name'] not in initial_bank_names]
            evaluation=runtime(d,'bank_evaluation','evaluate',{**ex,'candidates':str(collection/'candidates.json'),'order':order,'budget':len(order)*spec['num_envs']*spec['horizon'], 'reuse_results':spec.get('reuse_results') if index==first_round else None},len(order)*spec['num_envs']*spec['horizon'])
            rows=read(evaluation/'results.json');pending=[r for r in rows if r['label']==0 and not r['prefix_terminal']]
            for r in rows:
                r['round']=index;r['pulse_start_step']=r.get('pulse_start_step',pulse_delay(spec,index));r['source_actor_sha256']=source['policy']['actor_sha256']
                r['initial_label']=r['label']
            # Persist evaluated candidates even when a later training stage
            # cannot fit the remaining budget. Missing training stays pending.
            write(d/'evaluated_candidates.json',rows)
            if pending and not fixed_policy:
                support_inputs={str(evaluation/'results.json'):_file_sha(evaluation/'results.json')}
                for r in pending:
                    for filename in ['identity.json','snapshot.pkl']:
                        p=Path(r['snapshot'])/filename;support_inputs[str(p)]=_file_sha(p)
                pending_view=candidate_support_view(support,[support_row(r,False) for r in pending],support_inputs,pending_fraction=spec['pending_fraction'],max_pending_per_phase=spec['num_envs'])
                sp=d/'training_support.json';write(sp,pending_view)
                config=d/'policy_config.json';run_id=root.name+f'_repair_{index:04d}'
                prior=reuse_root/d.relative_to(root) if reuse_root is not None else None
                old_training=prior/'policy_training'/run_id if prior is not None else None
                reused_training=old_training is not None and (old_training/'formal_report.json').exists()
                if reused_training:
                    if read(prior/'training_support.json')!=pending_view:raise ValueError('reused PPO support changed')
                    config=prior/'policy_config.json';training=old_training;report=read(training/'formal_report.json')
                    old_config=read(config)
                    if (old_config['initialization']['source_frozen_policy']!=source['frozen_policy'] or
                            report['completed_training_transitions']!=spec['policy_steps']):
                        raise ValueError('reused PPO source or completed budget changed')
                    write(d/'learn_policy_reuse.json',dict(previous=str(training),config=str(config),
                        report_sha256=_file_sha(training/'formal_report.json'),new_training_interactions=0))
                else:
                    make_config(sp,source['frozen_policy'],spec['bootstrap_config'],config,run_id,source['policy']['iteration']+index+1,spec['policy_steps'],spec['seed']+10000+index,checkpoints=[spec['policy_steps']],panel_support_path=root/'source_pi_support.json',pending_fraction=spec['pending_fraction'],reward_mode=spec.get('reward_mode'),kl_control=spec.get('kl_control'))
                    cost=child(d,'learn_policy',['JIT/cli/train_unified_from_pi0.py','--config',config,'--run-id',run_id],spec['policy_steps']+1600,dict(JIT_RUN_ROOT=str(d/'policy_training')))
                    training=d/'policy_training'/run_id;report=read(training/'formal_report.json');cost.update(charged_interactions=report['completed_training_transitions']+report['train_panel_interactions'],accounting='actual')
                if current_only and not reused_training:
                    from .retention_experiment import export_training
                    export_training(training)
                name=f'{root.name}_repair_{index:04d}'
                freeze_development_checkpoint(d/'frozen',config_path=config,checkpoint=training/'checkpoints'/f"transition_{spec['policy_steps']}",name=name)
                bank_spec=dict(version=name,task=bank['task'],max_ticks=bank['max_ticks'],label_interaction_budget=bank['label_interaction_budget'],max_candidates_per_process=bank['max_candidates_per_process'],members=[dict(frozen_policy=m['frozen_policy'],roles=m['roles']) for m in bank['members']]+[dict(frozen_policy=str(d/'frozen/frozen_unified_policy.json'),roles=['proposer','evaluator'] if current_only else ['evaluator'])])
                bank_path=d/'expanded_bank.json';bank=lock_probe_bank(bank_spec,bank_path)
                pp=d/'pending.json';write(pp,pending)
                after=runtime(d,'after_learning','evaluate',{**ex,'bank':str(bank_path),'candidates':str(pp),'order':[name],'budget':len(pending)*spec['horizon']},len(pending)*spec['horizon'])
                resolved={r['index']:r for r in read(after/'results.json')}
                for r in rows:
                    if r['index'] in resolved:
                        new=resolved[r['index']];r.update(label=new['label'],witness=new['witness'],learning_attempted=True,bank_attempts=r['attempts'],attempts=r['attempts']+new['attempts'],learning_config=str(config))
                if current_only:
                    baseline_path=(Path(boundary['previous']) if boundary else (reuse_root if reuse_root is not None else root))/'baseline/candidates.json'
                    panel=retention_candidates(support,read(baseline_path)[0],spec['retention_samples_per_phase'])
                    panel_path=d/'retention_candidates.json';write(panel_path,panel)
                    checked=runtime(d,'retention_evaluation','evaluate',{**ex,'bank':str(bank_path),
                        'candidates':str(panel_path),'order':[source_name,name],'full_matrix':True,
                        'budget':2*len(panel)*spec['horizon']},2*len(panel)*spec['horizon'])
                    decision=promotion_decision(read(checked/'results.json'),source_name,name,
                        sum(r['label']==1 for r in resolved.values()),spec['minimum_retention'])
                    write(d/'promotion.json',dict(source=source_name,successor=name,**decision))
                    if decision['promote']:adopt=next(m for m in bank['members'] if m['name']==name)
            # Accumulate positives only in witnessed reset support; failed cells stay in visited ledger.
            keys={r['key'] for r in support['entries']};support.pop('support_sha256')
            for r in rows:
                if r['label']==1 and not r.get('prefix_terminal',False) and r['snapshot_context_sha256'] not in keys:
                    support['entries'].append(support_row(r,True));keys.add(r['snapshot_context_sha256'])
                    for filename in ['identity.json','snapshot.pkl']:
                        p=Path(r['snapshot'])/filename;support['inputs'][str(p)]=_file_sha(p)
            outcomes=d/'outcomes.json';write(outcomes,rows);support['inputs'][str(outcomes)]=_file_sha(outcomes);support['support_sha256']=canonical_sha256(support);write(d/'witnessed_support.json',support)
            reward,eligible,next_seen,parts=pulse_feedback(rows,seen,spec['reward_weights'],quality_mode=quality_mode)
            feedback=d/'feedback.json';write(feedback,dict(rewards=reward.tolist(),eligible=eligible.tolist(),parts=parts,component_sums={k:float(sum(v)) for k,v in parts.items()},new_cells=len(next_seen)-len(seen),outcomes=str(outcomes),outcomes_sha256=_file_sha(outcomes)))
            update=runtime(d,'update','update',{**ex,'collection':str(collection),'feedback':str(feedback)},0)
            checkpoint=str(update/'state.msgpack');m=read(update/'metrics.json')
            metrics.append(dict(round=index+1,source_policy=source_name,pulse_start_step=None if spec.get('pulse_batch_mode')=='mixed' else pulse_delay(spec,index),pulse_batch_mode=spec.get('pulse_batch_mode','single'),successes=sum(r['label']==1 for r in rows),failed_after_learning=sum(r['label']==0 and r['learning_attempted'] for r in rows),new_cells=len(next_seen)-len(seen),charged_interactions=inherited_cost+sum(c['charged_interactions'] for c in costs),reward_novelty=m['reward_components']['novelty'],reward_quality=m['reward_components']['quality'],**{k:v for k,v in m.items() if k!='reward_components'}))
            seen=next_seen;rows_all.extend(rows);write(root/'visited_cells.json',seen);write(root/'training_metrics.json',metrics);export(root,metrics,rows_all);status('round_completed',completed_rounds=index+1)
            if current_only:
                write(d/'source_ledger.json',dict(source=source_name,cells=seen))
                if adopt is not None:
                    nominal=runtime(d,'next_source_seed','seed_support',{**ex,'bank':str(bank_path),
                        'proposer':adopt['name'],'explorer_checkpoint':None,'allow_nominal_failure':True},spec['horizon']*(spec['horizon']+1))
                    ready=read(nominal/'status.json').get('support_ready',True)
                    if not ready:
                        decision=read(d/'promotion.json')
                        write(d/'promotion.json',{**decision,'panel_promote':decision['promote'],
                            'promote':False,'reason':'nominal_recovery_failed','source_retained':source['name']})
                        adopt=None
                    else:
                        fresh=read(nominal/'support.json')
                        if spec.get('neighborhood'):neighborhood_seeds.extend(read(nominal/'evaluation/results.json'))
                        seen=sorted({r['root_cell'] for r in fresh['entries']})
                        support.pop('support_sha256');keys={r['key'] for r in support['entries']}
                        support['entries'].extend(r for r in fresh['entries'] if r['key'] not in keys)
                        support['inputs'].update(fresh['inputs']);support['support_sha256']=canonical_sha256(support)
                        write(d/'next_training_support.json',support)
                        write(root/'visited_cells.json',seen)
                if adopt is not None:
                    source=adopt
                    checkpoint=promoted_explorer_checkpoint(spec,checkpoint)
                write(root/'current_source.json',dict(source=source['name'],frozen_policy=source['frozen_policy'],
                    explorer_checkpoint=checkpoint,completed_rounds=index+1,historical_helpers_used=False))
        status('running',stage='plot_tube_xz',completed_rounds=spec['rounds'])
        plot_completed_rounds(root,rows_all)
        status('completed',completed_rounds=spec['rounds'],final_test_used=False,checkpoint=checkpoint)
    except InteractionBudgetExhausted as exc:
        export(root,metrics,rows_all)
        write(root/'training_metrics.json',metrics)
        write(root/'visited_cells.json',seen)
        if metrics:
            if not (root/'analysis/tube_xz').exists():
                plot_completed_rounds(root,rows_all)
        status('completed',stop_reason='actual_interaction_cap',detail=str(exc),
               completed_rounds=len(metrics),final_test_used=False,checkpoint=checkpoint)
    except BaseException as exc:
        status('error',error=f'{type(exc).__name__}: {exc}',no_automatic_retry=True);export(root,metrics,rows_all);raise
