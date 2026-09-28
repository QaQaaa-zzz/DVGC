"""Finite fixed-tail Actor series with cumulative generator training data."""
from copy import deepcopy
from pathlib import Path
import time
from .contracts import digest,file_sha
from .protocol import atomic_json
from .proposals import stable_seed
from .production import read,implementation_identity,implementation_files,ProductionRunner,start_notifications
from .artifacts import load_corpus,validate_generator_receipt


def corpus_history(corpus):
    return [trace for group in ('history','teacher_new','actor_new') for trace in corpus['groups'][group]]


def plan_panels(panels,rounds,per_round,seed):
    if type(rounds) is not int or rounds<=0 or type(per_round) is not int or per_round<=0:
        raise ValueError('positive finite round and root counts required')
    used={r['root_episode_id'] for r in panels['new_roots']}
    available=sorted((r for r in panels['original_pending']
        if panels['splits'][r['root_episode_id']]=='generator_train' and r['root_episode_id'] not in used),
        key=lambda r:stable_seed(seed,r['root_id'],'series_panel'))
    chosen=[]
    for r in available:
        if r['root_episode_id'] not in used:chosen.append(r);used.add(r['root_episode_id'])
    if len(chosen)<rounds*per_round:raise ValueError('insufficient distinct TRAIN ancestors for declared series')
    return [deepcopy(chosen[i*per_round:(i+1)*per_round]) for i in range(rounds)]


def complete_bundle(root):
    root=Path(root);bundle=read(root/'current_source.json');receipt=read(root/'stages/complete_round.json')
    if (read(root/'status.json')['phase']!='completed' or receipt['result']!=bundle
        or receipt['output_sha256']!=digest(bundle)):
        raise ValueError('parent round is not committed and completed')
    validate_generator_receipt(bundle['generator']);load_corpus(bundle['corpus'])
    return bundle


def prepare_series(previous,output,repository,*,rounds=3,max_wall_seconds=43200):
    previous=Path(previous).resolve();output=Path(output).resolve();repo=Path(repository).resolve()
    bundle=complete_bundle(previous);base=read(previous/'production.json')
    commit=implementation_identity(repo)
    if not 0<max_wall_seconds<=43200:raise ValueError('finite series wall budget at most 12 hours')
    panels=plan_panels(read(previous/'panels.json'),rounds,32,base['seed'])
    locks={**base['locks'],str(previous/'production.json'):file_sha(previous/'production.json'),
        str(previous/'current_source.json'):file_sha(previous/'current_source.json')}
    spec={'schema':'jit_bridge_fixed_actor_series_v1','output':str(output),'previous':str(previous),
        'repository':str(repo),'implementation_commit':commit,'implementation_files':implementation_files(repo),
        'rounds':rounds,'round_panels':panels,'actor_policy':'fixed_source_for_all_students_and_teacher_tails',
        'fixed_actor_sha256':base['source']['actor_sha256'],'adoption_policy':'record_existing_gate_without_stopping_series',
        'budgets':{'per_round_physics':1060000,'max_physics':1060000*rounds,'max_supervised_updates':2000*rounds,
                   'student_transitions':128000*rounds,'max_wall_seconds':max_wall_seconds},
        'prior_physics':sum(c['charged_interactions'] for c in read(previous/'costs.json')),
        'prior_supervised_updates':sum(c.get('charged_updates',0) for c in read(previous/'costs.json')),
        'locks':locks,'final_test_open':False,'automatic_extension':False}
    output.mkdir(parents=True,exist_ok=False);atomic_json(output/'series.json',spec)
    atomic_json(output/'status.json',{'phase':'prepared','completed_rounds':0})
    return spec


def prepare_next_round(series,previous,index,started_unix):
    previous=Path(previous);bundle=complete_bundle(previous)
    base=read(Path(series['previous'])/'production.json')
    if base['source']['actor_sha256']!=series['fixed_actor_sha256']:raise ValueError('fixed Actor drift')
    bootstrap=read(previous/'stages/bootstrap.json')['result']
    continuation={'corpus':bundle['corpus'],'generator':bundle['generator'],
        'dev_fixture':bootstrap['dev_fixture'],'dev_fixture_sha256':bootstrap['dev_fixture_sha256']}
    if file_sha(continuation['dev_fixture'])!=continuation['dev_fixture_sha256']:raise ValueError('fixed G dev fixture drift')
    # Reuse only fixed-source smoke and before panels. No previous root search,
    # demonstrations, student, adoption, feedback or generator update is replayed.
    first=Path(series['previous']);stages={}
    allowed={'semantic_smoke','nominal','eval_before_core_batch_0000','eval_before_core_batch_0001',
             'eval_before_protected_batch_0000','eval_before_protected_batch_0001'}
    for name in allowed:
        path=first/'stages'/(name+'.json')
        stages[name]={'path':str(path),'sha256':file_sha(path)}
    root=Path(series['output'])/f'round_{index+1:04d}'
    if root.exists():raise ValueError('round already exists; reconcile receipts before resuming')
    spec=deepcopy(base)
    spec.update(output=str(root),repository=series['repository'],implementation_commit=series['implementation_commit'],
        implementation_files=series['implementation_files'],continuation=continuation,
        recovery={'previous_contract_identity':read(first/'stages/round_contract.json')['sha256'],
                  'stages':stages,'teachers':{}},series={'path':str(Path(series['output'])/'series.json'),
                  'sha256':file_sha(Path(series['output'])/'series.json'),'round':index+1,'fixed_actor':True})
    spec['budgets']={**base['budgets'],'bootstrap_recapture':0,'semantic_smoke':0,
        'teacher':32*31*400*2,'max_physics':series['budgets']['per_round_physics'],
        'max_supervised_updates':2000,'max_wall_seconds':series['budgets']['max_wall_seconds']}
    spec['locks']={**series['locks'],str(previous/'current_source.json'):file_sha(previous/'current_source.json'),
        str(previous/'stages/complete_round.json'):file_sha(previous/'stages/complete_round.json')}
    for receipt in stages.values():spec['locks'][receipt['path']]=receipt['sha256']
    spec['locks'][continuation['corpus']['path']]=continuation['corpus']['sha256']
    spec['locks'][continuation['generator']['checkpoint_manifest']]=continuation['generator']['checkpoint_manifest_sha256']
    spec['locks'][str(Path(continuation['generator']['checkpoint_manifest']).parent/'state.msgpack')]=file_sha(Path(continuation['generator']['checkpoint_manifest']).parent/'state.msgpack')
    spec['locks'][continuation['dev_fixture']]=continuation['dev_fixture_sha256']
    root.mkdir()
    panels=read(first/'panels.json');panels['new_roots']=series['round_panels'][index]
    atomic_json(root/'panels.json',panels)
    for name in ('training_support.json','retention_support.json'):(root/name).write_bytes((first/name).read_bytes())
    for name in ('panels.json','training_support.json','retention_support.json'):spec['locks'][str(root/name)]=file_sha(root/name)
    atomic_json(root/'production.json',spec);atomic_json(root/'started.json',{'started_unix':started_unix})
    atomic_json(root/'costs.json',[]);atomic_json(root/'status.json',{'phase':'prepared'})
    return spec


def series_costs(series):
    physical=updates=0
    for p in Path(series['output']).glob('round_*/costs.json'):
        for c in read(p):physical+=c['charged_interactions'];updates+=c.get('charged_updates',0)
    if physical>series['budgets']['max_physics'] or updates>series['budgets']['max_supervised_updates']:
        raise ValueError('whole series budget exceeded')
    return {'new_physics_charged_or_reserved':physical,'new_supervised_updates_charged':updates,
            'lifetime_physics_charged_or_reserved':physical+series['prior_physics']}


def run_series(series):
    root=Path(series['output']);started=root/'started.json'
    if started.exists():raise ValueError('series was already started; reconcile before explicit recovery')
    atomic_json(started,{'started_unix':time.time()});start=read(started)['started_unix']
    previous=Path(series['previous']);summaries=[]
    try:
        if implementation_identity(series['repository'])!=series['implementation_commit']:raise ValueError('series code drift')
        for p,h in series['locks'].items():
            if file_sha(p)!=h:raise ValueError('series source lock drift: '+p)
        start_notifications(series)
        for index in range(series['rounds']):
            if time.time()-start>=series['budgets']['max_wall_seconds']:raise TimeoutError('series wall budget exhausted')
            spec=prepare_next_round(series,previous,index,start)
            atomic_json(root/'ACTIVE_RUN.json',{'name':root.name,'lineage':str(root/'status.json'),'execution':str(Path(spec['output'])/'status.json')})
            atomic_json(root/'status.json',{'phase':'running','round':index+1,'completed_rounds':len(summaries),
                'current_round':spec['output'],**series_costs(series)})
            bundle=ProductionRunner(spec).run()
            previous=Path(spec['output']);summary={'round':index+1,'path':str(previous),
                'acceptance':bundle['actor_acceptance'],'generator':bundle['generator'],
                'matrix':read(previous/'result_matrix.json'),'costs':read(previous/'costs.json')}
            summaries.append(summary);atomic_json(root/'round_summaries.json',summaries)
            atomic_json(root/'latest_completed.json',{'round':index+1,'path':str(previous),'bundle':bundle})
        atomic_json(root/'status.json',{'phase':'completed','completed_rounds':len(summaries),
            'wall_seconds':time.time()-start,**series_costs(series)})
    except BaseException as e:
        atomic_json(root/'status.json',{'phase':'failed','completed_rounds':len(summaries),
            'error':repr(e),'wall_seconds':time.time()-start,**series_costs(series)})
        raise


def run_promotion_series(plan, callbacks):
    """Explicit v1.2 opt-in; production stage callbacks are required separately."""
    from .promotion import run_promotion_series as run
    return run(plan, callbacks)
