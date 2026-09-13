"""Finite short-pulse discovery with bank evaluation and delayed learning quality."""
from pathlib import Path
import csv
import json
import time
import numpy as np
from .exploration_loop import read,write
from .probe_bank import load_probe_bank,lock_probe_bank,_file_sha
from .evidence_integrity import canonical_sha256


def pulse_feedback(rows,seen,weights):
    """Newly visited cells stay visited after failure; unknown is not punished."""
    counts={};old=set(seen)
    for r in rows:
        if r['cell'] not in old:counts[r['cell']]=counts.get(r['cell'],0)+1
    novelty=np.zeros(len(rows));quality=np.zeros(len(rows));mask=np.zeros(len(rows),bool)
    for i,r in enumerate(rows):
        mask[i]=r['label']==1 or (r['label']==0 and (r['learning_attempted'] or r.get('prefix_terminal',False)))
        if mask[i]:
            novelty[i]=weights['novelty']/counts[r['cell']] if r['cell'] in counts else 0
            quality[i]=weights['success'] if r['label']==1 else -weights['failure']
    return novelty+quality,mask,sorted(old|{r['cell'] for r in rows}),dict(novelty=novelty.tolist(),quality=quality.tolist())


def budget_contract(spec,evaluators):
    rounds=spec['rounds'];n=spec['num_envs'];h=spec['horizon']
    if any(type(spec[k]) is not int or spec[k]<=0 for k in ['rounds','num_envs','horizon','pulse_steps','policy_steps']):raise ValueError('positive pulse budgets required')
    if spec['pulse_steps']>5 or h!=400 or spec['policy_steps']%3200:raise ValueError('short-pulse/horizon/aligned learning contract')
    baseline=evaluators*h
    prefixes=rounds*n*spec['pulse_steps']
    suffixes=sum((evaluators+i)*n*h for i in range(rounds))
    learning=rounds*(spec['policy_steps']+1600+n*h)
    return dict(baseline=baseline,prefixes=prefixes,bank_suffixes=suffixes,learning_and_reevaluation=learning,maximum_interactions=baseline+prefixes+suffixes+learning)


def support_row(row,witnessed):
    return dict(key=row['snapshot_context_sha256'],phase=row['phase'],snapshot=row['snapshot'],state_sha256=row['state_sha256'],snapshot_context_sha256=row['snapshot_context_sha256'],trajectory_id=row['prefix_file']+'::'+str(row['index']),root_cell=row['cell'],witnessed=witnessed,evidence_status='witnessed' if witnessed else 'pending',role='train',sampling_weight=1.)


def export(root,metrics,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name,items in [('training_process',metrics),('regions',[dict(cell=r['cell'],state_sha256=r['state_sha256'],context=r['snapshot_context_sha256'],status='successful' if r['label']==1 else ('unresolved_after_learning' if r['label']==0 and r['learning_attempted'] else 'pending'),snapshot=r['snapshot'],witness=r.get('witness'),round=r['round']) for r in rows])]:
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


def run(spec_path,output):
    from .gated_execution import run_gated_plan
    from .iterative_probe_training import candidate_support_view,make_config
    from .unified_policy_freeze import freeze_development_checkpoint
    spec=read(spec_path);root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    for path,expected in {**spec['input_files'],**spec['source_locks']}.items():
        if _file_sha(Path(path))!=expected:raise ValueError('pulse input/source drift: '+path)
    bank=load_probe_bank(Path(spec['bank']));bank_path=Path(spec['bank']);count=len(spec['order']);budget=budget_contract(spec,count)
    if spec['maximum_interactions']!=budget['maximum_interactions']:raise ValueError('pulse budget mismatch')
    source=next(m for m in bank['members'] if m['name']==spec['proposer'])
    initial_bank_names={m['name'] for m in bank['members']}
    base=read(spec['witnessed_support']);base['entries']=[r for r in base['entries'] if r.get('labels',{}).get(spec['proposer'])==1]
    if {r['phase'] for r in base['entries']}!={'upstream','downstream'}:raise ValueError('source pi needs witnessed support in both phases')
    base.pop('support_sha256');base['selection']='only frozen source-pi positive rows';base['support_sha256']=canonical_sha256(base)
    write(root/'source_pi_support.json',base);write(root/'declaration.json',dict(spec=spec,budget=budget,role='TRAIN',final_test_used=False))
    seen=sorted({r['root_cell'] for r in base['entries']});rows_all=[];metrics=[];costs=[];checkpoint=None;start=time.monotonic();support=json.loads(json.dumps(base));inherited_cost=0
    def status(phase,**kw):write(root/'status.json',dict(phase=phase,charged_interactions=inherited_cost+sum(c['charged_interactions'] for c in costs),new_interactions=sum(c['charged_interactions'] for c in costs),inherited_interactions=inherited_cost,maximum_interactions=budget['maximum_interactions'],wall_seconds=time.monotonic()-start,costs=costs,**kw))
    def child(directory,name,argv,maximum,env=None):
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
        p=directory/(name+'_spec.json');write(p,args);out=directory/name
        cost=child(directory,name,['JIT/cli/run_pulse_exploration.py','--mode',mode,'--spec',p,'--output',out],maximum)
        result=read(out/'status.json');actual=result['charged_interactions']
        if result['phase']!='completed' or not 0<=actual<=maximum:raise ValueError('runtime receipt/budget mismatch')
        cost.update(charged_interactions=actual,accounting='actual');return out
    try:
        first_round=0
        if spec.get('resume_run'):
            previous=Path(spec['resume_run']);old=read(previous/'declaration.json')['spec']
            contract=['proposer','order','num_envs','pulse_steps','delta_limit','horizon','policy_steps','pending_fraction','seed','learning_rate','minibatch_size','epochs','clip','target_kl','reward_weights','value_coefficient','entropy_coefficient','max_grad_norm','jump_start_state_sha256']
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
            d=root/f'round_{index:04d}';d.mkdir()
            ex={**spec,'bank':str(bank_path),'round_index':index,'explorer_checkpoint':checkpoint}
            if index==first_round and spec.get('reuse_collection'):
                collection=Path(spec['reuse_collection'])
                old=read(collection/'hyperparameters.json')
                for k in ['round_index','explorer_checkpoint','pulse_steps','num_envs','delta_limit','bank','seed']:
                    if old[k]!=ex[k]:raise ValueError('reused pulse contract differs: '+k)
                if read(collection/'status.json')['phase']!='completed':raise ValueError('incomplete reused collection')
                write(d/'reused_collection.json',dict(path=str(collection),prefix_sha256=_file_sha(collection/'prefixes.npz')))
            else:
                collection=runtime(d,'collection','collect',ex,spec['num_envs']*spec['pulse_steps'])
            order=spec['order']+[m['name'] for m in bank['members'] if m['name'] not in initial_bank_names]
            evaluation=runtime(d,'bank_evaluation','evaluate',{**ex,'candidates':str(collection/'candidates.json'),'order':order,'budget':len(order)*spec['num_envs']*spec['horizon'], 'reuse_results':spec.get('reuse_results') if index==first_round else None},len(order)*spec['num_envs']*spec['horizon'])
            rows=read(evaluation/'results.json');pending=[r for r in rows if r['label']==0 and not r['prefix_terminal']]
            for r in rows:r['round']=index
            if pending:
                support_inputs={str(evaluation/'results.json'):_file_sha(evaluation/'results.json')}
                for r in pending:
                    for filename in ['identity.json','snapshot.pkl']:
                        p=Path(r['snapshot'])/filename;support_inputs[str(p)]=_file_sha(p)
                pending_view=candidate_support_view(support,[support_row(r,False) for r in pending],support_inputs,pending_fraction=spec['pending_fraction'],max_pending_per_phase=spec['num_envs'])
                sp=d/'training_support.json';write(sp,pending_view)
                config=d/'policy_config.json';run_id=root.name+f'_repair_{index:04d}'
                make_config(sp,source['frozen_policy'],spec['bootstrap_config'],config,run_id,source['policy']['iteration']+index+1,spec['policy_steps'],spec['seed']+10000+index,checkpoints=[spec['policy_steps']],panel_support_path=root/'source_pi_support.json',pending_fraction=spec['pending_fraction'])
                cost=child(d,'learn_policy',['JIT/cli/train_unified_from_pi0.py','--config',config,'--run-id',run_id],spec['policy_steps']+1600,dict(JIT_RUN_ROOT=str(d/'policy_training')))
                training=d/'policy_training'/run_id;report=read(training/'formal_report.json');cost.update(charged_interactions=report['completed_training_transitions']+report['train_panel_interactions'],accounting='actual')
                name=f'{root.name}_repair_{index:04d}'
                freeze_development_checkpoint(d/'frozen',config_path=config,checkpoint=training/'checkpoints'/f"transition_{spec['policy_steps']}",name=name)
                bank_spec=dict(version=name,task=bank['task'],max_ticks=bank['max_ticks'],label_interaction_budget=bank['label_interaction_budget'],max_candidates_per_process=bank['max_candidates_per_process'],members=[dict(frozen_policy=m['frozen_policy'],roles=m['roles']) for m in bank['members']]+[dict(frozen_policy=str(d/'frozen/frozen_unified_policy.json'),roles=['evaluator'])])
                bank_path=d/'expanded_bank.json';bank=lock_probe_bank(bank_spec,bank_path)
                pp=d/'pending.json';write(pp,pending)
                after=runtime(d,'after_learning','evaluate',{**ex,'bank':str(bank_path),'candidates':str(pp),'order':[name],'budget':len(pending)*spec['horizon']},len(pending)*spec['horizon'])
                resolved={r['index']:r for r in read(after/'results.json')}
                for r in rows:
                    if r['index'] in resolved:
                        new=resolved[r['index']];r.update(label=new['label'],witness=new['witness'],learning_attempted=True,bank_attempts=r['attempts'],attempts=r['attempts']+new['attempts'],learning_config=str(config))
            # Accumulate positives only in witnessed reset support; failed cells stay in visited ledger.
            keys={r['key'] for r in support['entries']};support.pop('support_sha256')
            for r in rows:
                if r['label']==1 and r['snapshot_context_sha256'] not in keys:
                    support['entries'].append(support_row(r,True));keys.add(r['snapshot_context_sha256'])
                    for filename in ['identity.json','snapshot.pkl']:
                        p=Path(r['snapshot'])/filename;support['inputs'][str(p)]=_file_sha(p)
            outcomes=d/'outcomes.json';write(outcomes,rows);support['inputs'][str(outcomes)]=_file_sha(outcomes);support['support_sha256']=canonical_sha256(support);write(d/'witnessed_support.json',support)
            reward,eligible,next_seen,parts=pulse_feedback(rows,seen,spec['reward_weights'])
            feedback=d/'feedback.json';write(feedback,dict(rewards=reward.tolist(),eligible=eligible.tolist(),parts=parts,component_sums={k:float(sum(v)) for k,v in parts.items()},new_cells=len(next_seen)-len(seen),outcomes=str(outcomes),outcomes_sha256=_file_sha(outcomes)))
            update=runtime(d,'update','update',{**ex,'collection':str(collection),'feedback':str(feedback)},0)
            checkpoint=str(update/'state.msgpack');m=read(update/'metrics.json')
            metrics.append(dict(round=index+1,successes=sum(r['label']==1 for r in rows),failed_after_learning=sum(r['label']==0 and r['learning_attempted'] for r in rows),new_cells=len(next_seen)-len(seen),charged_interactions=inherited_cost+sum(c['charged_interactions'] for c in costs),reward_novelty=m['reward_components']['novelty'],reward_quality=m['reward_components']['quality'],**{k:v for k,v in m.items() if k!='reward_components'}))
            seen=next_seen;rows_all.extend(rows);write(root/'visited_cells.json',seen);write(root/'training_metrics.json',metrics);export(root,metrics,rows_all);status('round_completed',completed_rounds=index+1)
        status('completed',completed_rounds=spec['rounds'],final_test_used=False,checkpoint=checkpoint)
    except BaseException as exc:
        status('error',error=f'{type(exc).__name__}: {exc}',no_automatic_retry=True);export(root,metrics,rows_all);raise
