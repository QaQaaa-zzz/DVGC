"""Explicit receipt migration for the pre-PPO source-recheck guard failure."""
from copy import deepcopy
from pathlib import Path
import shutil
import time
from .contracts import digest,file_sha
from .protocol import StageJournal,atomic_json


def read(path):
    import json
    return json.loads(Path(path).read_text())


class RecoveryJournal(StageJournal):
    def __init__(self,path,contract):
        super().__init__(path,contract)
        self.recovery=contract.get('recovery',{})

    def stage(self,name,inputs,callback):
        inherited=self.recovery.get('stages',{}).get(name)
        if inherited and not (self.path/(name+'.json')).exists():
            if file_sha(inherited['path'])!=inherited['sha256']:
                raise ValueError('inherited receipt changed: '+name)
            record=read(inherited['path'])
            expected=digest({'round':self.recovery['previous_contract_identity'],'inputs':inputs})
            if record['input_sha256']!=expected:raise ValueError('inherited stage inputs changed: '+name)
            if record['output_sha256']!=digest(record['result']):raise ValueError('inherited output changed: '+name)
            return super().stage(name,inputs,lambda:record['result'])
        return super().stage(name,inputs,callback)


def prepare_recovery(previous,output,repository,*,replay_diagnostic=None):
    from .production import implementation_identity,implementation_files,read_teacher_traces
    from .artifacts import validate_generator_receipt,load_corpus
    previous=Path(previous).resolve();output=Path(output).resolve()
    # The original CLI parsed its JSON declaration with YAML (1e-05 became a string).
    # Preserve exactly the executed contract, rather than silently changing runtime types.
    import yaml
    old=read(previous/'production.json');status=read(previous/'status.json')
    if not old.get('recovery'):old=yaml.safe_load((previous/'production.json').read_text())
    error=status.get('error','')
    replay_failure='teacher replay invalid; never use empty-demo fallback' in error
    source_failure='source_succeeded_or_unknown_on_recheck' in error
    if status.get('phase')!='failed' or not (replay_failure or source_failure):
        raise ValueError('migration only handles diagnosed source recheck or teacher replay failures')
    if (previous/'student').exists():
        raise ValueError('migration requires no student execution')
    commit=implementation_identity(repository)
    started=read(previous/'started.json');costs=read(previous/'costs.json')
    if time.time()-started['started_unix']>=old['budgets']['max_wall_seconds']:
        raise TimeoutError('original wall budget exhausted')
    if any(c.get('phase')!='completed' for c in costs):raise ValueError('unreconciled child cost')
    contract=read(previous/'stages/round_contract.json')
    if contract['contract']!=old or contract['sha256']!=digest(old):raise ValueError('old contract changed')
    spec=deepcopy(old);locks=spec['locks']
    for path,sha in locks.items():
        if file_sha(path)!=sha:raise ValueError('source lock changed: '+path)
    visited=set()
    def pin(value):
        if isinstance(value,dict):
            for v in value.values():pin(v)
        elif isinstance(value,list):
            for v in value:pin(v)
        elif isinstance(value,str) and value.startswith(str(previous)+'/'):
            p=Path(value)
            if not p.is_file() or value in visited:return
            visited.add(value);locks[value]=file_sha(p)
            if p.suffix=='.json':pin(read(p))
    stages={}
    allowed={'semantic_smoke','bootstrap','generator_pretrain','nominal'}
    for path in sorted((previous/'stages').glob('*.json')):
        name=path.stem
        if name not in allowed and not (name.startswith('eval_') and not name.endswith('.running')):continue
        record=read(path)
        if record['output_sha256']!=digest(record['result']):raise ValueError('old stage output changed')
        stages[name]={'path':str(path),'sha256':file_sha(path)};pin(str(path))
    if not allowed<=stages.keys() or 'eval_teacher_source' not in stages:
        raise ValueError('required completed stages missing')
    rechecks=read(read(stages['eval_teacher_source']['path'])['result']['path'])
    if any(r['label'] not in (0,1) for r in rechecks):raise ValueError('unknown source recheck needs separate diagnosis')
    bootstrap=read(stages['bootstrap']['path'])['result']
    load_corpus(bootstrap['corpus'])
    if file_sha(bootstrap['dev_fixture'])!=bootstrap['dev_fixture_sha256']:raise ValueError('dev fixture changed')
    generator=read(stages['generator_pretrain']['path'])['result']
    validate_generator_receipt(generator)
    payload=Path(generator['checkpoint_manifest']).parent/'state.msgpack';pin(str(payload))
    teachers={};rejected_teachers={}
    for path in sorted((previous/'teachers').glob('*_result.json')):
        row=read(path)
        if row['teacher_status']=='not_scheduled':
            if row.get('source_recheck_label')!=1:raise ValueError('invalid skipped source recheck')
            continue
        if row['teacher_status']=='invalid' and replay_failure:
            rejected_teachers[row['root_id']]={'path':str(path),'sha256':file_sha(path)};pin(str(path))
            continue
        if (row['source_actor_sha256']!=old['source']['actor_sha256'] or row['generator']!=generator
            or row['teacher_status'] not in ('searched_no_solution','verified_solution')):
            raise ValueError('incompatible completed teacher receipt')
        if not any(r['root_id']==row['root_id'] and r['label']==0 for r in rechecks):
            raise ValueError('teacher receipt lacks current source negative')
        read_teacher_traces({row['root_id']:row})
        teachers[row['root_id']]={'path':str(path),'sha256':file_sha(path)};pin(str(path))
        proposal=Path(row['proposal_path']) if row.get('proposal_path') else path.with_name(path.name.replace('_result.json','_proposals.npz'))
        if not proposal.exists():
            inherited=old.get('recovery',{}).get('teachers',{}).get(row['root_id'])
            if inherited is None:raise ValueError('missing original teacher proposal receipt')
            ancestor=Path(inherited['path']);proposal=Path(inherited['proposal_path']) if inherited.get('proposal_path') else ancestor.with_name(ancestor.name.replace('_result.json','_proposals.npz'))
            if file_sha(ancestor)!=inherited['sha256']:raise ValueError('original teacher receipt changed')
        if file_sha(proposal)!=row['proposal_sha256']:raise ValueError('teacher proposals changed')
        teachers[row['root_id']]['proposal_path']=str(proposal)
        locks[str(proposal)]=file_sha(proposal);pin(str(proposal))
    for name in ('production.json','status.json','costs.json','started.json','stages/round_contract.json'):
        pin(str(previous/name))
    if replay_diagnostic is not None:
        if not replay_failure:raise ValueError('matched replay diagnostic requires replay failure')
        import_replay_diagnostic(previous,Path(replay_diagnostic).resolve(),old,contract,stages,locks,costs)
    if sum(c['charged_interactions'] for c in costs)>old['budgets']['max_physics']:
        raise ValueError('inherited recovery cost exceeds original cap')
    spec.update(output=str(output),repository=str(Path(repository).resolve()),implementation_commit=commit,
                implementation_files=implementation_files(repository))
    spec['recovery']={'previous':str(previous),'previous_contract_identity':contract['sha256'],
        'stages':stages,'teachers':teachers,'rejected_teachers':rejected_teachers,
        'reason':'matched_batch_teacher_replay' if replay_failure else 'source_recheck_success_is_legal',
        'budget_policy':'inherit all charged costs and original wall deadline; no budget reset'}
    output.mkdir(parents=True,exist_ok=False)
    for name in ('panels.json','training_support.json','retention_support.json'):
        shutil.copyfile(previous/name,output/name);locks[str(output/name)]=file_sha(output/name)
    atomic_json(output/'started.json',started);atomic_json(output/'costs.json',costs)
    atomic_json(output/'production.json',spec)
    atomic_json(output/'status.json',{'phase':'prepared','inherited_physics':sum(c['charged_interactions'] for c in costs),
        'inherited_supervised_updates':sum(c.get('charged_updates',0) for c in costs)})
    return spec


def import_replay_diagnostic(previous,diagnostic,old,contract,stages,locks,costs):
    """Import one independently executed full-batch replay, including its real cost."""
    from .production import lane_arrays
    execution=read(diagnostic/'execution/status.json');plan=read(diagnostic/'plan.json')
    if (execution.get('phase')!='completed' or len(execution.get('stages',[]))!=1
        or execution['stages'][0].get('returncode')!=0
        or execution.get('plan_sha256')!=file_sha(diagnostic/'plan.json')):
        raise ValueError('diagnostic execution receipt incomplete or changed')
    if plan['input_files'].get(str(diagnostic/'spec.json'))!=file_sha(diagnostic/'spec.json'):
        raise ValueError('diagnostic spec is not locked')
    for p,h in plan['input_files'].items():
        if file_sha(p)!=h:raise ValueError('diagnostic input changed')
    stage=execution['stages'][0]
    if (len(plan['stages'])!=1 or stage['argv']!=plan['stages'][0]['argv']
        or stage['argv'][-6:]!=['--mode','evaluate','--spec',str(diagnostic/'spec.json'),'--output',str(diagnostic/'rollout')]):
        raise ValueError('diagnostic executed a different command')
    spec=read(diagnostic/'spec.json')
    matches=[p for p in (previous/'evaluations').glob('teacher_*/spec.json') if read(p)==spec]
    if len(matches)!=1:raise ValueError('diagnostic must exactly repeat one original search spec')
    original=matches[0];ordinal=original.parent.name.removeprefix('teacher_')
    old_teacher=read(previous/'teachers'/f'{ordinal}_result.json')
    if old_teacher['teacher_status']!='invalid':raise ValueError('diagnostic not bound to failed replay')
    expected=read(spec['candidates']);rows=read(diagnostic/'rollout/results.json')
    if ([r['candidate_id'] for r in expected]!=list(range(1,32))
        or [r['candidate_id'] for r in rows]!=list(range(1,32))):
        raise ValueError('diagnostic must preserve all candidate lanes')
    for lane,(before,after) in enumerate(zip(expected,rows)):
        a=after['attempts'][0]
        if (before['root_id']!=after['root_id'] or before['snapshot_context_sha256']!=after['snapshot_context_sha256']
            or a['trace_lane']!=lane or a['actor_sha256']!=old['source']['actor_sha256']
            or a['normalizer_sha256']!=old['source']['normalizer_sha256'] or after['label'] not in (0,1)):
            raise ValueError('diagnostic trace identity changed or unknown')
        lane_arrays(a)
        locks[a['trace']]=a['trace_sha256']
    if rows[old_teacher['selected_candidate_id']-1]['label']!=1:
        raise ValueError('selected teacher did not pass matched-batch diagnostic')
    measured=read(diagnostic/'rollout/status.json')
    if (measured['phase']!='completed' or type(measured['charged_interactions']) is not int
        or not 0<measured['charged_interactions']<=plan['max_interactions']):
        raise ValueError('invalid diagnostic cost')
    import numpy as np
    with np.load(rows[0]['attempts'][0]['trace'],allow_pickle=False) as tape:
        expected_cost=int(np.asarray(tape['mask']).size)
    if measured['charged_interactions']!=expected_cost:raise ValueError('diagnostic charged cost differs from trace')
    name='replay_batch_'+ordinal
    receipt={'path':str(diagnostic/'rollout/results.json'),'sha256':file_sha(diagnostic/'rollout/results.json')}
    search_stage=read(previous/'stages'/f'eval_teacher_{ordinal}.json')
    # Stage identity binds inputs and old contract; stage name is deliberately separate.
    target=diagnostic/'migration_receipt.json'
    record={'input_sha256':search_stage['input_sha256'],'output_sha256':digest(receipt),'result':receipt}
    if target.exists() and read(target)!=record:raise ValueError('diagnostic migration receipt changed')
    if not target.exists():atomic_json(target,record)
    stages['eval_'+name]={'path':str(target),'sha256':file_sha(target)}
    for p in (original,Path(spec['candidates']),Path(spec['bridge_action_plan']['path']),
              diagnostic/'spec.json',diagnostic/'budget.json',diagnostic/'plan.json',diagnostic/'execution/status.json',
              diagnostic/'rollout/status.json',diagnostic/'rollout/results.json',target):
        locks[str(p)]=file_sha(p)
    costs.append({'stage':name+'_diagnostic','phase':'completed','charged_interactions':measured['charged_interactions'],
        'charged_updates':0,'accounting':'measured independent matched-batch replay, reused after recovery',
        'wall_seconds':measured['wall_seconds'],'receipt':str(diagnostic/'rollout/status.json')})


def prepare_campaign_recovery(previous,output,repository,*,max_wall_seconds=86400):
    """Resume a completed A1 preparation after a finite source-control conflict.

    A new user-authorized wall window is explicit; lifetime physics/optimizer
    costs remain charged. No collector, optimizer or simulator runs here.
    """
    from .production import implementation_identity,implementation_files,read_teacher_traces
    from .artifacts import validate_generator_receipt,load_corpus
    previous=Path(previous).resolve();output=Path(output).resolve()
    old=read(previous/'production.json');status=read(previous/'status.json')
    if old.get('schema')!='jit_bridge_campaign_v1_2' or status.get('phase')!='failed':
        raise ValueError('requires failed v1.2 campaign')
    if 'teacher replay invalid' not in status.get('error','') or (previous/'students').exists() or (previous/'warmup').exists():
        raise ValueError('only diagnosed pre-student teacher conflict recovery allowed')
    if type(max_wall_seconds) not in (int,float) or not 0<max_wall_seconds<=86400:
        raise ValueError('explicit positive wall window at most 24 hours required')
    contract=read(previous/'stages/round_contract.json')
    if contract!={'contract':old,'sha256':digest(old)}:raise ValueError('old contract drift')
    costs=read(previous/'costs.json')
    if any(c.get('phase','completed')!='completed' for c in costs):raise ValueError('unreconciled child cost')
    locks=dict(old['locks'])
    for path,sha in locks.items():
        if file_sha(path)!=sha:raise ValueError('old source lock drift: '+path)
    def pin(path):
        path=str(Path(path).resolve());locks[path]=file_sha(path);return path
    for name in ['production.json','status.json','costs.json','panels.json','source_rows.json',
                 'training_support.json','seed_support/support.json','source_collections/plan.json',
                 'retention_observations.json','retention_observations.npz','generator_dev_fixture.npz',
                 'generator_pretrain_0000_config.json','pretrain_result.json','gpu_semantic_smoke.json']:
        pin(previous/name)
    source_rows=read(previous/'source_rows.json')
    if source_rows['source_actor_sha256']!=old['source']['actor_sha256']:raise ValueError('source rows drift')
    # Verify each distinct recorded trajectory at migration, keeping its own receipt.
    verified=set()
    for row in source_rows['rows']:
        refs=[(row['prefix_file'],row['prefix_sha256'])]
        refs.extend((a['trace'],a['trace_sha256']) for a in row.get('attempts',[]))
        for path,sha in refs:
            if path in verified:continue
            if file_sha(path)!=sha:raise ValueError('source trajectory drift')
            verified.add(path)
    gen=read(previous/'pretrain_result.json');validate_generator_receipt(gen)
    pin(gen['checkpoint_manifest']);pin(Path(gen['checkpoint_manifest']).parent/'state.msgpack')
    config=read(previous/'generator_pretrain_0000_config.json');load_corpus(config['corpus'])
    if file_sha(config['dev_fixture'])!=config['dev_fixture_sha256']:raise ValueError('G dev fixture drift')
    if read(previous/'gpu_semantic_smoke.json')['status']!='passed':raise ValueError('smoke not passed')
    stages={}
    for path in sorted((previous/'stages').glob('eval_*.json')):
        if path.name.endswith('.running.json'):continue
        record=read(path)
        if record['output_sha256']!=digest(record['result']):raise ValueError('evaluation stage drift')
        result=record['result']
        if file_sha(result['path'])!=result['sha256']:raise ValueError('evaluation result drift')
        pin(path);pin(result['path'])
        for row in read(result['path']):
            for attempt in row.get('attempts',[]):
                if file_sha(attempt['trace'])!=attempt['trace_sha256']:raise ValueError('evaluation trace drift')
        stages[path.stem]={'path':str(path),'sha256':file_sha(path)}
    teachers={};conflicts=[]
    for path in sorted((previous/'teachers').glob('*_result.json')):
        row=read(path)
        if row['teacher_status']=='verified_solution':
            read_teacher_traces({row['root_id']:row})
            proposal=path.with_name(path.name.replace('_result.json','_proposals.npz'))
            if file_sha(proposal)!=row['proposal_sha256']:raise ValueError('teacher proposal drift')
            pin(path);pin(proposal);pin(row['demo']['path'])
            teachers[row['root_id']]={'path':str(path),'sha256':file_sha(path),'proposal_path':str(proposal)}
        elif row['teacher_status']=='invalid':
            repeat=path.with_name(path.name.replace('_result.json','_layout_repeat.json'));e=read(repeat)
            cid=e['selected_candidate_id']
            if (e['changed_candidate_ids']!=[0] or e['search_labels'][0]!=0 or e['repeat_labels'][0]!=1
                or e['search_labels'][cid]!=1 or e['repeat_labels'][cid]!=1
                or any(v not in (0,1) for v in e['search_labels']+e['repeat_labels'])):
                raise ValueError('failure not a complete finite source-only conflict')
            pin(path);pin(repeat);conflicts.append(row['root_id'])
    if not conflicts:raise ValueError('no diagnosed finite conflict')
    # Prior two engineering failures are accounted by the previous recovery receipt.
    ledger=read(previous.parent/'recovery_audit.json')
    historical=ledger['old_reserved_physics']
    lifetime_cap=ledger['total_physics_cap']
    lifetime_spent=historical+sum(c['charged_interactions'] for c in costs)
    if lifetime_spent>=lifetime_cap:raise ValueError('lifetime physics budget exhausted')
    commit=implementation_identity(repository)
    output.mkdir(parents=True,exist_ok=False)
    preparation=dict(source_actor_sha256=old['source']['actor_sha256'],panels=read(previous/'panels.json'),
        aggregate_path=str(previous/'source_rows.json'),support_path=str(previous/'training_support.json'),
        seed_dir=str(previous/'seed_support'),collection_plan=str(previous/'source_collections/plan.json'),
        retention_ref=read(previous/'retention_observations.json'),incumbent=gen,
        bootstrap={'corpus':config['corpus'],'dev_fixture':config['dev_fixture'],'dev_fixture_sha256':config['dev_fixture_sha256']},
        inputs={p:h for p,h in locks.items() if p.startswith(str(previous)+'/')})
    atomic_json(output/'reused_preparation.json',preparation)
    spec=deepcopy(old);spec.update(output=str(output),repository=str(Path(repository).resolve()),
        implementation_commit=commit,implementation_files=implementation_files(repository),locks=locks,
        recovery_preparation={'path':str(output/'reused_preparation.json'),'sha256':file_sha(output/'reused_preparation.json')},
        recovery={'previous':str(previous),'previous_contract_identity':digest(old),'stages':stages,'teachers':teachers},
        stage_plan=str(output/'stage_plan.json'))
    spec['budgets'].update(max_physics=lifetime_cap,max_wall_seconds=max_wall_seconds)
    stage_plan=read(old['stage_plan']);stage_plan['budgets'].update(physics_cap=lifetime_cap,wall_seconds=max_wall_seconds)
    atomic_json(output/'stage_plan.json',stage_plan)
    costs=[{'stage':'prior_failed_reservations','charged_interactions':historical,'charged_updates':0,
            'accounting':'inherited failed-attempt reservations','phase':'completed'}]+costs
    atomic_json(output/'costs.json',costs);atomic_json(output/'panels.json',preparation['panels'])
    atomic_json(output/'production.json',spec)
    atomic_json(output/'recovery_audit.json',dict(previous=str(previous),diagnosed_conflict_roots=conflicts,
        inherited_physics_charged_or_reserved=lifetime_spent,total_physics_cap=lifetime_cap,
        remaining_physics=lifetime_cap-lifetime_spent,old_reserved_physics=historical,
        new_wall_window_seconds=max_wall_seconds,old_deadline_expired=True,
        reused_G_updates=gen['updates'],new_pretraining_updates=0,new_physics=0,
        source_trajectories_verified=len(verified)))
    atomic_json(output/'status.json',{'phase':'prepared','charged_interactions':lifetime_spent,'supervised_updates':20000})
    return spec
