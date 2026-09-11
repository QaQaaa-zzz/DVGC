"""Finite TRAIN loop: residual arrivals -> pending resets -> delayed witnesses.

The orchestrator runs on CPU. Every simulator child has its own external-job
check, time limit, source locks and interaction reservation. No automatic retry.
"""
from pathlib import Path
import csv
import json
import os
import time

from .evidence_integrity import canonical_sha256
from .probe_bank import _file_sha, load_probe_bank, lock_probe_bank
from .exploration_pool import import_candidates, select_pending, validate_pool

SCHEMA = 'jit_delayed_exploration_loop_v1'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


def budget_contract(spec, evaluator_count):
    """Conservative maxima include both arms, diagnostics, old pending and panels."""
    if type(evaluator_count) is not int or evaluator_count <= 0:
        raise ValueError('positive evaluator count required')
    for key in ('rounds', 'policy_steps', 'pending_limit', 'stage_timeout_seconds'):
        if type(spec.get(key)) is not int or spec[key] <= 0:
            raise ValueError('positive loop parameter required: ' + key)
    if spec['policy_steps'] % 3200 or spec['rounds'] > 2:
        raise ValueError('declared pilot supports at most two rounds and aligned PPO')
    ex = spec['explorer']
    for key in ('num_envs', 'batches', 'max_candidates_per_episode'):
        if type(ex.get(key)) is not int or ex[key] <= 0:
            raise ValueError('invalid explorer budget: ' + key)
    if ex['horizon'] != 400 or ex['evaluation_episodes'] != ex['num_envs']:
        raise ValueError('matched complete 400 tick collectors required')
    forward = 2 * ex['num_envs'] * 400 * (ex['batches'] + 2)
    # Initial zero-residual diagnostic produces no candidates; training + final do.
    arrivals = ex['num_envs'] * ex['max_candidates_per_episode'] * (ex['batches'] + 1)
    rows = []
    for index in range(spec['rounds']):
        available = arrivals * (index + 1)
        before = 2 * available * (evaluator_count + index) * 400
        after = 2 * available * 400
        policy = spec['policy_steps'] + 1600  # one four-state fixed TRAIN panel
        rows.append(dict(round_index=index, forward=forward, baseline_suffix=before,
                         policy_training_and_panel=policy, delayed_suffix=after,
                         maximum_interactions=forward+before+policy+after))
    return dict(rounds=rows, maximum_interactions=sum(r['maximum_interactions'] for r in rows),
                candidate_limit_per_arm_per_round=arrivals, attempts_per_stage=1)


def pending_rows(pool, limit):
    """Successful feedback removes rows from pending; failures remain learnable."""
    rows = []
    for entry in select_pending(pool, limit):
        source = entry['sources'][0]['candidate']
        rows.append(dict(key=entry['key'], phase=entry['phase'], snapshot=entry['snapshot_dir'],
            state_sha256=entry['state_sha256'], snapshot_context_sha256=entry['context_sha256'],
            trajectory_id=source['prefix_file']+'::'+str(source['episode_index']),
            root_cell=entry['root_cell'], witnessed=False, evidence_status='pending', role='train'))
    return rows


def transition_counts(before, after):
    """Keep bank-failure -> learned distinct from previously untested -> witnessed."""
    validate_pool(before); validate_pool(after)
    counts = dict(previous_bank_no_witness_to_witness=0, unknown_to_witness=0,
                  still_pending=0, previously_witnessed=0)
    for key, old in before['entries'].items():
        new = after['entries'][key]
        if old['status'] == 'witnessed':
            counts['previously_witnessed'] += 1
        elif new['status'] != 'witnessed':
            counts['still_pending'] += 1
        elif old['observations'] and old['observations'][-1]['label'] == 0:
            counts['previous_bank_no_witness_to_witness'] += 1
        else:
            counts['unknown_to_witness'] += 1
    counts['verified_root_cells'] = len({e['root_cell'] for e in after['entries'].values() if e['status']=='witnessed'})
    counts['pending_root_cells'] = len({e['root_cell'] for e in after['entries'].values() if e['status']=='pending'})
    return counts


def _bank_spec(bank, version, new_manifest):
    return dict(version=version, task=bank['task'], max_ticks=bank['max_ticks'],
        label_interaction_budget=bank['label_interaction_budget'],
        max_candidates_per_process=bank['max_candidates_per_process'],
        members=[dict(frozen_policy=m['frozen_policy'],roles=m['roles']) for m in bank['members']]
                 +[dict(frozen_policy=str(new_manifest),roles=['proposer','evaluator'])])


def export_evidence(root, metrics, pools, costs):
    """Complete candidate table and replot data; no interpolated reachable hull."""
    rows=[]
    for arm,pool in pools.items():
        for key,entry in sorted(pool['entries'].items()):
            rows.append(dict(arm=arm,key=key,phase=entry['phase'],root_cell=entry['root_cell'],
                status=entry['status'],origin_pi=entry['origin_source_pi'],
                first_discovery_round=entry['first_discovery_round'],
                observations=len(entry['observations']),snapshot_dir=entry['snapshot_dir']))
    for name, data in [('candidates',rows),('round_metrics',metrics),('costs',costs)]:
        if data:
            fields=list(dict.fromkeys(k for row in data for k in row))
            with (root/(name+'.csv')).open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(data)
    if pools:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8,4))
        for arm in ('learned_residual','fixed_random'):
            selected=[m for m in metrics if m['arm']==arm]
            if selected:
                ax.plot([m['round_index']+1 for m in selected],[m['verified_root_cells'] for m in selected],marker='o',label=arm)
            elif arm in pools:
                count=len({e['root_cell'] for e in pools[arm]['entries'].values() if e['status']=='witnessed'})
                ax.scatter([0],[count],label=arm)
        ax.set(xlabel='Round',ylabel='Witnessed root cells in each arrival pool',title='TRAIN pilot: shared evolving policy bank')
        ax.legend();fig.tight_layout()
        for suffix in ('png','pdf','svg'):fig.savefig(root/('coverage.'+suffix))
        plt.close(fig)
    (root/'INDEX.md').write_text('# Delayed JIT exploration loop\n\n'
        'TRAIN engineering/development evidence. Each arm explores the same frozen pi with matched forward budgets. '
        'Only learned-arm pending arrivals guide the shared next pi; the random arm is a conditional exploration control, '
        'not an independent end-to-end training baseline.\n\n'
        '- `pipeline_status.json`: completion/stop reason and total accounting.\n'
        '- `round_metrics.csv`, `candidates.csv`, `costs.csv`: complete replot tables.\n'
        '- `round_*/`: full prefixes, snapshots, checkpoints, labels, training and delayed-feedback records.\n'
        '- `coverage.png/pdf/svg`: exact witnessed-cell counts, no continuous-boundary claim.\n\n'
        'Pending is not infeasible. Old PPO trajectories are never relabeled for PPO replay. '
        'The frozen critic is telemetry only. Delayed success updates reset selection and the next source policy.\n')


def run(spec_path, output):
    from .gated_execution import run_gated_plan
    from .iterative_probe_training import candidate_support_view, make_config
    from .unified_policy_freeze import freeze_development_checkpoint
    from .exploration_loop_support import append_witnessed_support
    spec_path,root=Path(spec_path).resolve(),Path(output).resolve()
    spec=read(spec_path)
    if spec.get('schema')!=SCHEMA or spec.get('role')!='TRAIN' or spec.get('final_test_used') is not False:
        raise ValueError('explicit TRAIN delayed loop required')
    bank_path=Path(spec['bank']);bank=load_probe_bank(bank_path)
    budget=budget_contract(spec,sum('evaluator' in m['roles'] for m in bank['members']))
    if spec['maximum_interactions']!=budget['maximum_interactions']:
        raise ValueError('loop budget declaration mismatch')
    for path,sha in {**spec['input_files'],**spec['source_locks']}.items():
        if _file_sha(path)!=sha:raise ValueError('loop input/source drift: '+path)
    root.mkdir(parents=True,exist_ok=False)
    write(root/'declaration.json',dict(spec=spec,budget=budget,spec_sha256=_file_sha(spec_path)))
    started=time.monotonic();costs=[];metrics=[];pools={};pool_paths={}
    proposer=spec['proposer'];witnessed=read(spec['witnessed_support'])
    def status(phase,**extra):
        write(root/'pipeline_status.json',dict(phase=phase,wall_seconds=time.monotonic()-started,
            charged_interactions=sum(c['charged_interactions'] for c in costs),
            maximum_interactions=budget['maximum_interactions'],costs=costs,**extra))
    def child(directory,name,argv,maximum,extra_env=None):
        # Fresh plan per child also rechecks STTW before each training/evaluation.
        generated_inputs={}
        for position,arg in enumerate(argv[:-1]):
            if str(arg) in ('--config','--spec','--pool','--bank'):
                dependency=Path(argv[position+1]).resolve()
                generated_inputs[str(dependency)]=_file_sha(dependency)
        plan=dict(schema='jit_gated_plan_v1',max_interactions=maximum,wait_timeout_seconds=1,
            gate=spec['gate'],input_files={**spec['input_files'],**generated_inputs},source_locks=spec['source_locks'],
            stages=[dict(name=name,argv=[spec['python'],*map(str,argv)],cwd=spec['repo'],
                env=dict(JAX_PLATFORMS='cuda,cpu',CUDA_VISIBLE_DEVICES='0',XLA_PYTHON_CLIENT_PREALLOCATE='false',
                         PYTHONPATH=str(Path(spec['repo'])/'JIT/src'),JIT_AUTO_PUBLISH='0',**(extra_env or {})),
                timeout_seconds=spec['stage_timeout_seconds'],max_interactions=maximum)])
        path=directory/(name+'_plan.json');write(path,plan)
        reservation=dict(stage=str(directory.relative_to(root))+'/'+name,charged_interactions=maximum,
                         maximum_interactions=maximum,accounting='reserved_until_completion')
        costs.append(reservation);status('running',current_stage=reservation['stage'])
        then=time.monotonic()
        try:
            outcome=run_gated_plan(path,directory/(name+'_execution'),wait=False)
            if outcome['phase']!='completed':
                if outcome.get('reserved_interactions')==0 or outcome['phase'] in ('blocked','gate_timeout'):
                    reservation.update(charged_interactions=0,accounting='not_launched')
                raise RuntimeError('gated child did not complete: '+name+' phase='+outcome['phase'])
        finally:
            reservation['wall_seconds']=time.monotonic()-then
        return reservation
    def settle(cost,actual):
        if type(actual) is not int or not 0<=actual<=cost['maximum_interactions']:
            raise ValueError('child cost outside declaration')
        cost.update(charged_interactions=actual,accounting='actual_completed')
    def reevaluate(directory,name,pool_path,eval_bank,order,index):
        from .exploration_pool import select_pending
        available=len(select_pending(read(pool_path),10**9));maximum=available*len(order)*400
        if not available:return read(pool_path),pool_path
        out=directory/name
        cost=child(directory,name,['JIT/cli/reevaluate_exploration_pool.py','--pool',pool_path,'--bank',eval_bank,
            '--order',*order,'--horizon','400','--budget',str(maximum),'--output',out,'--round-index',str(index)],maximum)
        result=read(out/'status.json')
        if result['status']!='selection_complete':raise RuntimeError('incomplete delayed evaluation')
        settle(cost,result['charged_interactions'])
        return read(out/'pool.json'),out/'pool.json'
    try:
        for index in range(spec['rounds']):
            directory=root/f'round_{index:03d}';directory.mkdir()
            member=next(m for m in bank['members'] if m['name']==proposer)
            baseline=directory/'baseline_seed.json'
            write(baseline,dict(policy_sha256=member['policy']['actor_sha256'],cells=[]))
            # Per-pi zero-residual arrivals establish each arm's identical baseline.
            for arm in ('learned_residual','fixed_random'):
                arm_dir=directory/arm;arm_dir.mkdir()
                ex={**spec['explorer'],'bank':str(bank_path),'proposer':proposer,'baseline_cells':str(baseline),
                    'controller_mode':arm,'reward_mode':'arrival_novelty_v1','suffix_budget':0,
                    'candidate_selection':'phase_stratified_v1',
                    'evaluator_order':[m['name'] for m in bank['members'] if 'evaluator' in m['roles']],
                    'seed':spec['explorer']['seed']+index}
                path=arm_dir/'explorer.json';write(path,ex)
                cost=child(arm_dir,'explore',['JIT/cli/train_coverage_explorer.py','--spec',path,'--output',arm_dir/'arrivals'],
                           ex['num_envs']*400*(ex['batches']+2))
                result=read(arm_dir/'arrivals/status.json')
                if result['phase']!='completed':raise RuntimeError('explorer did not complete')
                settle(cost,result['charged_interactions'])
                pool=import_candidates(arm_dir/'arrivals',pools.get(arm),round_index=index)
                pool_path=arm_dir/'arrival_pool.json';write(pool_path,pool)
                pool,pool_path=reevaluate(arm_dir,'before_learning',pool_path,bank_path,ex['evaluator_order'],index)
                pools[arm],pool_paths[arm]=pool,pool_path
            rows=pending_rows(pools['learned_residual'],spec['pending_limit'])
            write(directory/'selected_pending.json',rows)
            if not rows:
                status('stopped_no_pending',completed_rounds=index,reason='current bank witnessed all selected learned arrivals; no fabricated pending training')
                export_evidence(root,metrics,pools,costs)
                export_evidence(directory,metrics,pools,costs)
                return read(root/'pipeline_status.json')
            refreshed=append_witnessed_support(witnessed,pools['learned_residual'],pool_paths['learned_residual'])
            write(directory/'witnessed_training_support.json',refreshed)
            support=candidate_support_view(refreshed,rows,{str(pool_paths['learned_residual']):_file_sha(pool_paths['learned_residual'])},
                pending_fraction=spec['pending_fraction'],max_pending_per_phase=spec['pending_limit'])
            support_path=directory/'training_support.json';write(support_path,support)
            run_id=root.name+f'_round_{index:03d}'
            config_path=directory/'policy_training.json'
            make_config(support_path,member['frozen_policy'],spec['bootstrap_config'],config_path,run_id,
                member['policy']['iteration']+1,spec['policy_steps'],spec['policy_seed']+index,
                checkpoints=[spec['policy_steps']],panel_support_path=spec['witnessed_support'],pending_fraction=spec['pending_fraction'])
            cost=child(directory,'train_policy',['JIT/cli/train_unified_from_pi0.py','--config',config_path,'--run-id',run_id],
                spec['policy_steps']+1600,dict(JIT_RUN_ROOT=str(directory/'training')))
            training_dir=directory/'training'/run_id;report=read(training_dir/'formal_report.json')
            settle(cost,report['completed_training_transitions']+report['train_panel_interactions'])
            name=root.name+f'_checkpoint_{index:03d}'
            freeze_development_checkpoint(directory/'frozen',config_path=config_path,
                checkpoint=training_dir/'checkpoints'/f"transition_{spec['policy_steps']}",name=name)
            new_manifest=directory/'frozen/frozen_unified_policy.json'
            new_bank_path=directory/'expanded_bank.json'
            new_bank=lock_probe_bank(_bank_spec(bank,root.name+f'_bank_{index:03d}',new_manifest),new_bank_path)
            for arm in ('learned_residual','fixed_random'):
                before=pools[arm]
                after,path=reevaluate(directory/arm,'after_learning',pool_paths[arm],new_bank_path,[name],index)
                metrics.append(dict(round_index=index,arm=arm,**transition_counts(before,after)))
                pools[arm],pool_paths[arm]=after,path
            write(directory/'delayed_feedback.json',dict(metrics=metrics[-2:],
                use='remove newly witnessed rows from pending; next round uses trained source pi; remaining pending reconsidered',
                old_ppo_replay=False,critic_reward_coefficient=0))
            proposer,bank_path,bank=name,new_bank_path,new_bank
            export_evidence(root,metrics,pools,costs)
            export_evidence(directory,metrics,pools,costs)
            status('round_completed',completed_rounds=index+1)
        status('completed',completed_rounds=spec['rounds'],stop_reason='declared_round_limit',
               final_test_used=False,formal_envelope_claim=False)
        return read(root/'pipeline_status.json')
    except BaseException as exc:
        status('error',error=f'{type(exc).__name__}: {exc}',no_automatic_retry=True)
        export_evidence(root,metrics,pools,costs)
        raise
