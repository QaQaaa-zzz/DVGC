"""Source-bound, bounded production stages for the opt-in v1.1 pilot."""
from .performance import timed
from copy import deepcopy
import json
from pathlib import Path
import time
from contextlib import contextmanager


@contextmanager
def wall_deadline(seconds):
    import signal
    if seconds<=0:raise TimeoutError("whole pilot wall budget exhausted")
    previous=signal.getsignal(signal.SIGALRM)
    def expire(*_):raise TimeoutError("whole pilot wall budget exhausted")
    signal.signal(signal.SIGALRM,expire);signal.setitimer(signal.ITIMER_REAL,seconds)
    try:yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)

import numpy as np
from .contracts import digest,file_sha
from .protocol import atomic_json
from .proposals import stable_seed


def read(path):return json.loads(Path(path).read_text())


def implementation_identity(repository):
    import subprocess
    repo=str(Path(repository).resolve())
    def git(*args):return subprocess.check_output(['git','-C',repo,*args],text=True).strip()
    if git('status','--porcelain','--untracked-files=normal','--','JIT/src','JIT/cli'):
        raise ValueError('production code must be committed before binding or running')
    return git('rev-parse','HEAD')


def implementation_files(repository):
    import subprocess
    repo=Path(repository).resolve()
    names=subprocess.check_output(['git','-C',str(repo),'ls-files','JIT/src','JIT/cli'],text=True).splitlines()
    return {str(repo/name):file_sha(repo/name) for name in names if (repo/name).is_file()}


def select_panels(rows,*,seed,new_roots=32,bootstrap=32,core=64,protected=64):
    """Select before new execution; never filter the original training support."""
    rows=deepcopy(rows)
    if any(not r.get('root_episode_id') for r in rows):raise ValueError('ancestor identity required')
    ancestors=sorted({r['root_episode_id'] for r in rows},key=lambda a:stable_seed(seed,a,'split'))
    ndev=max(1,round(len(ancestors)*.2))
    splits={a:('generator_dev' if i<ndev else 'generator_train') for i,a in enumerate(ancestors)}
    original=[r for r in rows if r['label']==0 and not r.get('prefix_terminal',False)]
    available=sorted(rows,key=lambda r:stable_seed(seed,r['snapshot_context_sha256'],'panel'))
    used=set()
    def choose(count,label,split=None):
        chosen=[]
        for r in available:
            a=r['root_episode_id']
            if (a in used or r['label']!=label or r.get('prefix_terminal',False)
                or (split and splits[a]!=split)):continue
            chosen.append(r);used.add(a)
            if len(chosen)==count:break
        if len(chosen)!=count:raise ValueError(f'insufficient distinct ancestors for panel {label}/{split}: {len(chosen)}/{count}')
        return chosen
    roots=choose(new_roots,0,'generator_train')
    dev=max(1,round(bootstrap*.2))
    boot=choose(bootstrap-dev,1,'generator_train')+choose(dev,1,'generator_dev')
    return dict(original_pending=original,new_roots=roots,bootstrap=boot,
        core=choose(core,1),protected=choose(protected,1),splits=splits,
        selection='seeded distinct ancestor selection before new rollouts; original pending support unchanged')


def acceptance_decision(before,after,student):
    from ..policy_retention import paired_counts
    if set(before)!=set(after) or set(before)!={'new_roots','core','protected','nominal'}:
        raise ValueError('all four acceptance panels required')
    counts={k:paired_counts(before[k],after[k]) for k in before}
    protected=counts['protected'];retention=protected['retained']/protected['old_positive'] if protected['old_positive'] else 0.
    adopted=(not any(c['unknown'] for c in counts.values()) and counts['core']['lost']==0
        and counts['core']['old_positive']>0 and retention>=.98
        and counts['new_roots']['gained']-counts['new_roots']['lost']>=1
        and before['nominal']==[1] and after['nominal']==[1])
    return {**{k:student[k] for k in ('actor_sha256','normalizer_sha256')},'adopted':bool(adopted),
            'counts':counts,'protected_retention':retention,'thresholds':{'core_max_new_failures':0,
            'protected_min_retention':.98,'new_net_gain':1},'empirical_development_check':True}


def prepare_bound_run(*,pointer,source_round,output,repository,max_interactions,
                      max_supervised_updates,max_wall_seconds,seed=270901):
    """Bind explicit files. This function prepares, never starts computation."""
    from ..probe_bank import lock_probe_bank
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    source_round=Path(source_round).resolve();p=read(pointer);frozen=Path(p['frozen_policy']);source=read(frozen)['policy']
    source_spec=read(source_round/'bank_evaluation_spec.json')
    if source_spec['proposer']!=source['name']:raise ValueError('pointer differs from supplied round Actor')
    support_path=source_round/'training_support.json';support=read(support_path)
    bycontext={r['snapshot_context_sha256']:r for r in support['entries']}
    rows=[r for r in read(source_round/'bank_evaluation/results.json') if not r.get('prefix_terminal',False)]
    for row in rows:
        attempt=next((a for a in row['attempts'] if a['actor_sha256']==source['actor_sha256']),None)
        if attempt is None or attempt['label']!=row['label']:raise ValueError('source-only labels required')
        entry=bycontext.get(row['snapshot_context_sha256'])
        row['root_episode_id']=read(Path(row['snapshot'])/'identity.json')['parent_trajectory']
        if entry and entry['trajectory_id']!=row['root_episode_id']:raise ValueError('ancestor provenance mismatch')
        row['root_id']=row['snapshot_context_sha256']
    panels=select_panels(rows,seed=seed)
    atomic_json(root/'panels.json',panels)
    # Original complete view, including every pending weight, remains byte-identical.
    (root/'training_support.json').write_bytes(support_path.read_bytes())
    if len(panels['original_pending'])!=sum(r['evidence_status']=='pending' for r in support['entries']):
        raise ValueError('original pending pool and support differ')
    original_bank=read(source_spec['bank'])
    bank_spec={k:original_bank[k] for k in ('task','max_ticks','label_interaction_budget','max_candidates_per_process')}
    bank_spec.update(version='bridge_source',members=[{'frozen_policy':str(frozen),'roles':['proposer','evaluator']}])
    lock_probe_bank(bank_spec,root/'source_bank.json')
    anchors=[r for r in support['entries'] if r.get('labels',{}).get(source['name'])==1 and r.get('witnessed')]
    if {r['phase'] for r in anchors}!={'upstream','downstream'}:raise ValueError('source retention anchors missing phase')
    anchor=dict(schema='jit_bridge_retention_support_v1_1',role='train',final_test_used=False,entries=anchors,
        inputs={str(frozen):file_sha(frozen)},selection='all current-source witnessed anchors in original support')
    for row in anchors:
        for filename in ('identity.json','snapshot.pkl'):
            path=Path(row['snapshot'])/filename;anchor['inputs'][str(path)]=file_sha(path)
    anchor['support_sha256']=digest(anchor);atomic_json(root/'retention_support.json',anchor)
    budget={'semantic_smoke':4800,'bootstrap_recapture':12800,'teacher':422400,
        'student_and_acceptance':258400,'max_physics':max_interactions,'max_supervised_updates':max_supervised_updates,
        'max_wall_seconds':max_wall_seconds}
    if max_interactions<698400 or max_supervised_updates<22000 or not 0<max_wall_seconds<float('inf'):
        raise ValueError('bounded pilot requires declared full stage reservations')
    selected={k:v for k,v in source_spec.items() if k not in ('input_files','source_locks','reuse_results','reuse_collection','resume_evaluation_root','evaluation_batch_size','explorer_checkpoint')}
    selected.update(bank=str(root/'source_bank.json'),proposer=source['name'],order=[source['name']],
        seed=seed,record_actor_preobservations=True,full_matrix=True)
    inputs=[Path(pointer),frozen,source_round/'bank_evaluation/results.json',support_path,
            Path(source['formal_config']),root/'source_bank.json',source_round/'source_ledger.json',root/'panels.json',root/'training_support.json',root/'retention_support.json']
    manifest=dict(schema='jit_bridge_bound_production_v1_1',execute=True,repository=str(Path(repository).resolve()),
        implementation_commit=implementation_identity(repository),implementation_files=implementation_files(repository),
        output=str(root),source_frozen_policy=str(frozen),source=source,source_round=str(source_round),
        source_runtime=selected,seed=seed,budgets=budget,locks={str(p):file_sha(p) for p in inputs},
        original_pending_count=len(panels['original_pending']),declared_new_roots=32,
        remaining_pending='retain training eligibility; not part of pilot evaluation claims',
        final_test_open=False,automatic_retry=False,iterations=1)
    atomic_json(root/'production.json',manifest)
    atomic_json(root/'status.json',{'phase':'prepared','charged_interactions':0,'supervised_updates':0})
    return manifest


def lane_arrays(attempt):
    from .verified_trace_cache import TRACE_CACHE
    arrays=TRACE_CACHE.lane(attempt['trace'],attempt['trace_sha256'],attempt['trace_lane'])
    arrays['valid_mask']=np.ones(len(arrays['mask']),bool)
    arrays['failure']=arrays['physical_failure']
    return arrays


def trace_metadata(row,policy,protocol,splits,origin):
    return dict(root_id=row['root_id'],root_episode_id=row['root_episode_id'],
        root_context_sha256=row['snapshot_context_sha256'],actor_sha256=policy['actor_sha256'],
        normalizer_sha256=policy['normalizer_sha256'],source_actor_sha256=policy['actor_sha256'],
        model_sha256=policy['xml_sha256'],protocol_sha256=protocol,origin_type=origin,
        role=row.get('data_role','train'),inherited_split=splits[row['root_episode_id']],complete=True,full_success=True,
        onset=row.get('pulse_scheduled_start_step',row.get('pulse_start_step')),
        trace_start_step=row.get('snapshot_control_step',row.get('pulse_start_step',0)+row.get('pulse_applied_steps',0)),
        success_criterion='stable_forward_recovery')


def actor_traces(results,policy,protocol,splits,*,bootstrap=False):
    from .feedback_data import trace_from_evaluation
    traces=[]
    for row in results:
        if row['label']!=1:continue
        attempt=row['attempts'][0]
        m=trace_metadata(row,policy,protocol,splits,'bootstrap_actor' if bootstrap else 'adopted_actor')
        trace=trace_from_evaluation(attempt,m)
        if bootstrap:trace['metadata']['origin_type']='bootstrap_actor'
        traces.append(trace)
    return traces


@timed('candidate_scoring')
def teacher_candidate(row,kind,candidate_id,last_action):
    from .teacher import action_delta_cost
    attempt=row['attempts'][0];a=lane_arrays(attempt)
    return dict(candidate_id=candidate_id,kind=kind,full_success=row['label']==1,
        invalid=row['label'] is None,task_return=attempt['task_return'],
        action_delta_cost=action_delta_cost(a['normalized_action_executed'],last_action),
        attempt=attempt)


def source_recheck_disposition(label):
    if label==0:return None
    if label!=1:raise ValueError('unknown source recheck is invalid, not a normal no-solution result')
    return dict(teacher_status='not_scheduled',reason='source_recheck_succeeded',
                historical_source_label=0,source_recheck_label=1,new_gain_eligible=False,training_eligible=True)


def is_source_conflict_quarantine(row):
    return (row.get('teacher_status')=='invalid' and row.get('reason')=='source_control_repeat_conflict'
        and row.get('source_control_labels') in ([0,1],[1,0])
        and row.get('selected_replay_label') in (0,1) and row.get('complete_finite_replay') is True
        and row.get('new_gain_eligible') is False and row.get('training_eligible') is True
        and row.get('source_recheck_label') is None and 'demo' not in row)


def quarantine_source_conflict(row,replay):
    # A finite selected failure does not resolve a contradictory source label.
    # Quarantine both finite outcomes; unknown/incomplete execution still stops.
    if (replay.get('source_control_repeat_conflict') is not True or replay.get('label') not in (0,1)
        or replay.get('source_control_labels') not in ([0,1],[1,0])
        or replay.get('complete_finite_replay') is not True):return None
    return {**row,'teacher_status':'invalid','reason':'source_control_repeat_conflict',
        'source_control_labels':replay['source_control_labels'],'selected_replay_label':replay['label'],
        'complete_finite_replay':True,'source_recheck_label':None,
        'new_gain_eligible':False,'training_eligible':True,'quarantined':True}


def reject_finite_replay(row,replay):
    """Reject an unrepeatable teacher without inventing a verified demonstration."""
    proof=replay.get('finite_replay_rejection')
    if (not proof or replay.get('label')!=0 or proof.get('source_labels')!=[0,0]
        or proof.get('selected_labels')!=[1,0] or proof.get('complete_finite') is not True):return None
    if 'demo' in row:raise ValueError('rejected replay cannot contain a demonstration')
    return {**row,'teacher_status':'replay_rejected','reason':'selected_candidate_failed_repeat',
        'selected_candidate_id':replay['candidate_id'],'selected_replay_label':0,
        'source_recheck_label':0,'training_eligible':True,'replay_rejection':proof,
        'verification':{'candidate_id':replay['candidate_id'],'full_success':False}}


class ProductionRunner:
    def __init__(self,manifest,*,_teacher_worker=False):
        import jax
        if _teacher_worker and (type(self).__name__!='PersistentTeacherRunner' or jax.default_backend()!='gpu'):
            raise ValueError('teacher worker requires dedicated GPU entrypoint')
        if not _teacher_worker and jax.default_backend()!='cpu':raise ValueError('production supervisor must use JAX_PLATFORMS=cpu; GPU work belongs to gated children')
        self.spec=manifest;self.root=Path(manifest['output']);self.start=time.monotonic()
        import os
        os.environ['JIT_PERFORMANCE_FILE']=str(self.root/'performance.jsonl')
        self.retry_generator=False;self.active_teacher_root=None;self.teacher_in_progress=False
        self.manifest_sha256=file_sha(self.root/'production.json')
        started=self.root/'started.json'
        if not started.exists():atomic_json(started,{'started_unix':time.time()})
        self.started_unix=read(started)['started_unix']
        self.costs=read(self.root/'costs.json') if (self.root/'costs.json').exists() else []
        self.runtime=manifest['source_runtime'];self.repo=Path(manifest['repository'])
        self.source=manifest['source'];self.panels=read(self.root/'panels.json')
        self.protocol=digest({'version':manifest.get('protocol_version','1.1'),'task':'stable_forward_recovery','H':16,'horizon':400,
            'source_physics':self.source['xml_sha256'],'source_reward':self.runtime['reward_mode']})
        from .recovery import RecoveryJournal
        self.journal=RecoveryJournal(self.root/'stages',manifest)

    def status(self,phase,**kwargs):
        atomic_json(self.root/'status.json',{'phase':phase,'charged_interactions':sum(c['charged_interactions'] for c in self.costs),
            'supervised_updates':sum(c.get('charged_updates',0) for c in self.costs),
            'wall_seconds':time.time()-self.started_unix,'costs':self.costs,**kwargs})

    def child(self,name,argv,maximum,*,updates=0,extra_env=None):
        from ..gated_execution import run_gated_plan
        import sys
        if file_sha(self.root/'production.json')!=self.manifest_sha256:raise ValueError('production declaration changed')
        if implementation_identity(self.repo)!=self.spec['implementation_commit']:raise ValueError('reviewed implementation changed')
        remaining=self.spec['budgets']['max_wall_seconds']-(time.time()-self.started_unix)
        if remaining<=0:raise TimeoutError('whole pilot wall budget exhausted')
        if sum(c['charged_interactions'] for c in self.costs)+maximum>self.spec['budgets']['max_physics']:
            raise ValueError('physical budget exhausted before child')
        if sum(c.get('charged_updates',0) for c in self.costs)+updates>self.spec['budgets']['max_supervised_updates']:
            raise ValueError('supervised update budget exhausted before child')
        from .verified_files import LOCK_CACHE
        for p,h in self.spec['locks'].items():
            LOCK_CACHE.verify(p,h)
        inputs={}
        for i,a in enumerate(argv[:-1]):
            if a in ('--spec','--config','--request'):inputs[str(Path(argv[i+1]).resolve())]=file_sha(argv[i+1])
        if '--request' in argv:
            request=read(argv[argv.index('--request')+1])
            inputs.update(request.get('input_files',{}))
        inputs[str(self.root/'production.json')]=file_sha(self.root/'production.json')
        plan=dict(schema='jit_gated_plan_v1',gate={**self.runtime['gate'],'wait_until_idle':False},
            input_files=inputs,source_locks={**self.spec['locks'],**self.spec['implementation_files']},max_interactions=max(1,maximum),
            wait_timeout_seconds=remaining,stages=[dict(name=name,argv=[sys.executable,*map(str,argv)],
            cwd=str(self.repo),env=dict(JAX_PLATFORMS='cuda,cpu',PYTHONPATH=str(self.repo/'JIT/src'),
            XLA_PYTHON_CLIENT_PREALLOCATE='false',JIT_AUTO_PUBLISH='0',
            JIT_PERFORMANCE_FILE=str(self.root/'performance.jsonl'),**(extra_env or {})),
            timeout_seconds=remaining,max_interactions=maximum)])
        directory=self.root/'execution';directory.mkdir(exist_ok=True)
        planpath=directory/(name+'_plan.json');atomic_json(planpath,plan)
        cost=dict(stage=name,charged_interactions=maximum,charged_updates=updates,
            accounting='conservative reservation until actual receipt; failed attempts remain charged')
        self.costs.append(cost);atomic_json(self.root/'costs.json',self.costs);self.status('running',stage=name)
        result=run_gated_plan(planpath,directory/name,wait=True,poll_seconds=10)
        cost['phase']=result['phase'];atomic_json(self.root/'costs.json',self.costs)
        if result['phase']!='completed':raise RuntimeError(f'{name}: {result["phase"]}')
        cost['wall_seconds']=sum(r.get('wall_seconds',0) for r in result['stages'])
        atomic_json(self.root/'costs.json',self.costs)
        return cost

    def evaluate(self,name,rows,*,policy_bank=None,policy_name=None,prefixes=None,source_only=None):
        if prefixes is None and len(rows)>32:
            result=[]
            for i in range(0,len(rows),32):
                result.extend(self.evaluate(name+f'_batch_{i//32:04d}',rows[i:i+32],
                    policy_bank=policy_bank,policy_name=policy_name))
            return result
        inputs={'rows':rows,'bank':policy_bank or self.runtime['bank'],'actor':policy_name or self.source['name'],
            'prefixes_sha256':None if prefixes is None else digest(np.asarray(prefixes).tolist())}
        def execute():
            d=self.root/'evaluations'/name;d.mkdir(parents=True,exist_ok=False)
            atomic_json(d/'candidates.json',rows)
            spec={**self.runtime,'bank':inputs['bank'],'order':[inputs['actor']],
                'proposer':inputs['actor'],'candidates':str(d/'candidates.json'),'budget':len(rows)*400,
                'record_actor_preobservations':True,'full_matrix':True}
            if prefixes is not None:
                path=d/'prefixes.npz';np.savez_compressed(path,actions=prefixes,source_only=source_only,
                    root_contexts=np.array([r['snapshot_context_sha256'] for r in rows]))
                spec['bridge_action_plan']={'path':str(path),'sha256':file_sha(path)}
            atomic_json(d/'spec.json',spec)
            cost=self.child(name,['JIT/cli/run_pulse_exploration.py','--mode','evaluate','--spec',d/'spec.json','--output',d/'rollout'],len(rows)*400)
            status=read(d/'rollout/status.json');cost.update(charged_interactions=status['charged_interactions'],
                active_interactions=status['active_interactions'],accounting='measured')
            atomic_json(self.root/'costs.json',self.costs)
            return {'path':str(d/'rollout/results.json'),'sha256':file_sha(d/'rollout/results.json')}
        receipt=self.journal.stage('eval_'+name,inputs,execute)
        if file_sha(receipt['path'])!=receipt['sha256']:raise ValueError('evaluation receipt changed')
        return read(receipt['path'])

    def generator(self,mode,corpus,dev_fixture,*,incumbent=None):
        def execute():
            if mode=='incremental' and not corpus['new_data']:
                from .artifacts import validate_generator_full_state,load_corpus
                load_corpus(corpus)
                state=validate_generator_full_state(incumbent)
                policy=self.spec.get('generator_update_policy','fixed_dev_best')
                result={**incumbent,'status':'skipped_no_new_data','updates':0,'charged_updates':0,
                    'generator_update_policy':policy,'state_updates':int(state['updates']),
                    'initial_state_updates':int(state['updates']),'rng_advanced':False,
                    'old_dev_metric':'monitor_only' if policy=='last_valid' else 'selection',
                    'inference_parameters':'ema','monitoring_evaluation_performed':False}
                if self.spec.get('generator_update_policy')=='last_valid':
                    result.update(total_charged_updates=self.spec['generator_charged_updates_before'],
                        charged_updates_scope='lifetime',
                        historical_charge_accounting=self.spec.get('generator_charged_updates_scope','lifetime'))
                return result
            previous=[c for c in self.costs if c['stage'].startswith('generator_'+mode+'_')]
            if previous and (mode!='incremental' or not self.retry_generator):
                raise RuntimeError('generator retry requires explicit --retry-generator after receipt reconciliation')
            updates=20000 if mode=='pretrain' else 2000
            charged=sum(c['charged_updates'] for c in previous)
            if charged>=updates:raise RuntimeError('generator cumulative update budget exhausted')
            name='generator_'+mode+f'_{len(previous):04d}'
            config={'mode':mode,'source_frozen_policy':self.spec['source_frozen_policy'],
                'seed':self.spec['seed'],'corpus':corpus,'dev_fixture':str(dev_fixture),
                'dev_fixture_sha256':file_sha(dev_fixture),'updates':updates,
                'output':str(self.root/('generator_'+mode)),'result':str(self.root/(mode+'_result.json')),
                'max_wall_seconds':self.spec['budgets']['max_wall_seconds'],
                'incumbent':incumbent,'generator_update_policy':self.spec.get('generator_update_policy','fixed_dev_best')}
            if self.spec.get('generator_charged_updates_before') is not None:
                config['charged_updates_before']=self.spec['generator_charged_updates_before']
            if 'generator_reference_frozen_policy' in self.spec:
                config['generator_reference_frozen_policy']=self.spec['generator_reference_frozen_policy']
            path=self.root/(name+'_config.json');atomic_json(path,config)
            try:
                self.child(name,['JIT/cli/run_generative_bridge.py','worker','--spec',path],0,updates=updates-charged)
            except BaseException:
                # Only durable completed-attempt receipts can reduce a reservation.
                if mode=='incremental':
                    receipts=sorted((self.root/'generator_incremental').glob('attempt_*/cost_receipt.json'))
                    attempts=list((self.root/'generator_incremental').glob('attempt_*'))
                    if receipts and len(receipts)==len(attempts) and self.costs[-1]['stage']==name:
                        measured=sum(read(p)['charged_updates'] for p in receipts)
                        self.costs[-1]['charged_updates']=max(0,measured-charged)
                        self.costs[-1]['accounting']='durable G attempt receipts'
                        atomic_json(self.root/'costs.json',self.costs)
                raise
            result=read(config['result'])
            return result
        stage='generator_selection' if mode=='incremental' else 'generator_pretrain'
        inputs={'corpus':corpus,'incumbent':incumbent}
        if 'generator_update_policy' in self.spec:inputs['generator_update_policy']=self.spec['generator_update_policy']
        if 'generator_reference_frozen_policy' in self.spec:
            from .worker import generator_reference
            generator_reference(self.spec)
            inputs['generator_reference_frozen_policy']=self.spec['generator_reference_frozen_policy']
        result=self.journal.stage(stage,inputs,execute)
        from .artifacts import validate_generator_receipt
        if 'checkpoint_manifest' in result:validate_generator_receipt(result)
        return result

    def smoke(self):
        rows=self.panels['bootstrap'][:4]
        reference=self.evaluate('smoke_actor',rows)
        prefixes=[]
        for row in reference:
            a=lane_arrays(row['attempts'][0])['normalized_action_executed'][:16]
            if len(a)<16:raise ValueError('semantic smoke needs a real H16 source prefix')
            prefixes.append(a)
        for name in ('smoke_bridge','smoke_replay'):
            result=self.evaluate(name,rows,prefixes=np.asarray(prefixes),source_only=np.array([True,False,False,False]))
            for i,row in enumerate(result):
                a=lane_arrays(row['attempts'][0]);n=len(a['done'])
                if not np.array_equal(a['actor_observation_after'][:-1],a['actor_observation_before'][1:]):
                    raise ValueError('GPU pre/post continuity drift')
                origins=a['action_origin_code']
                expected=np.zeros(n,int) if i==0 else np.where(np.arange(n)<16,1,2)
                if not np.array_equal(origins,expected):raise ValueError('GPU handoff timing drift')
                if i and not np.allclose(a['normalized_action_executed'][:16],prefixes[i],atol=1e-6):
                    raise ValueError('GPU executed prefix drift')
                if row['label']!=reference[i]['label']:raise ValueError('source-prefix replay endpoint differs in semantic smoke')
        report={'status':'passed','physical_validation':True,'horizon':400,'bridge_horizon':16,
            'source_actor_sha256':self.source['actor_sha256'],'checks':['real pre/post continuity','H16 boundary','executed prefix','source-prefix endpoint replay'],
            'not_checked':['GPU empty-demo PPO optimization','research performance'],'maximum_interactions':4800}
        atomic_json(self.root/'gpu_semantic_smoke.json',report)
        return report

    def bootstrap(self):
        from .feedback_data import build_corpus
        from .artifacts import save_corpus
        from .data import build_action_windows
        results=self.evaluate('bootstrap',self.panels['bootstrap'])
        traces=actor_traces(results,self.source,self.protocol,self.panels['splits'],bootstrap=True)
        corpus=build_corpus(traces,[],[],adoption={},expected={'model_sha256':self.source['xml_sha256'],
            'protocol_sha256':self.protocol},splits=self.panels['splits'])
        receipt=save_corpus(corpus,self.root/'bootstrap_corpus')
        dev=[t for t in traces if t['metadata']['inherited_split']=='generator_dev']
        if not corpus['groups']['history'] or not dev:raise ValueError('real train and dev successes are required for G bootstrap; no fake data')
        obs=[];actions=[];ancestors=[]
        for trace in dev:
            w=build_action_windows(trace);obs.extend(w['observations']);actions.extend(w['actions'])
            ancestors.extend([trace['metadata']['root_episode_id']]*len(w['actions']))
        if not obs:raise ValueError('no real full-H16 generator dev windows')
        rng=np.random.default_rng(self.spec['seed']);n=min(256,len(obs));ids=rng.choice(len(obs),n,replace=False)
        fixture=self.root/'generator_dev_fixture.npz'
        np.savez_compressed(fixture,observations=np.asarray(obs)[ids],actions=np.asarray(actions)[ids],
            timesteps=rng.integers(0,100,n,dtype=np.int32),noise=rng.normal(size=(n,16,4)).astype(np.float32))
        atomic_json(self.root/'generator_dev_manifest.json',{'path':str(fixture),'sha256':file_sha(fixture),
            'ancestors':sorted(set(ancestors)),'role':'generator_dev','training_allowed':False})
        return {'corpus':receipt,'dev_fixture':str(fixture),'dev_fixture_sha256':file_sha(fixture)}

    def replay_teacher(self,ordinal,candidates,actions,selected_id,*,search_results=None):
        # Changing the search batch into one replay world changes the GPU physics
        # layout. Repeat the complete frozen batch, then inspect the same lane.
        ids=[r['candidate_id'] for r in candidates]
        full=self.spec.get('teacher_layout') in ('source_control_in_32_world_batch','source_control_in_candidate_batch')
        count=17+self.spec.get('teacher_colored_noise_candidates',15)
        if ids!=list(range(0 if full else 1,count)) or selected_id not in ids:
            raise ValueError('replay requires original candidate identities')
        results=self.evaluate(f'replay_batch_{ordinal:04d}',candidates,
            prefixes=actions,source_only=np.asarray([cid==0 for cid in ids],bool))
        if [r['candidate_id'] for r in results]!=ids:
            raise ValueError('replay candidate ordering changed or incomplete')
        selected=results[ids.index(selected_id)]
        if full and search_results is not None:
            if [r['candidate_id'] for r in search_results]!=ids:
                raise ValueError('search candidate ordering changed or incomplete')
            changed=[cid for cid,a,b in zip(ids,search_results,results) if a['label']!=b['label']]
            atomic_json(self.root/'teachers'/f'{ordinal:04d}_layout_repeat.json',{
                'candidate_ids':ids,'search_labels':[r['label'] for r in search_results],
                'repeat_labels':[r['label'] for r in results],'changed_candidate_ids':changed,
                'selected_candidate_id':selected_id,'robust_probability_claim':False})
            if 0 in changed or results[0]['label'] is None:
                selected={**selected,'source_control_repeat_conflict':True,
                    'source_control_labels':[search_results[0]['label'],results[0]['label']],
                    'complete_finite_replay':all(r['label'] in (0,1) for r in search_results+results)}
            if (self.spec.get('teacher_replay_failure_policy')=='reject_finite_candidate'
                and search_results[ids.index(selected_id)]['label']==1 and selected['label']==0
                and search_results[0]['label']==results[0]['label']==0
                and all(r['label'] in (0,1) for r in search_results+results)):
                # Verify durable real trajectories, not merely the scalar labels.
                from ..pulse_exploration_runtime import suffix_label
                source=self.spec['source'];horizon=self.spec['source_runtime']['horizon']
                for batch in (search_results,results):
                    for lane,result in enumerate(batch):
                        original=candidates[lane]
                        if any(result.get(k)!=original[k] for k in ('root_id','snapshot_context_sha256')):
                            raise ValueError('replay root/context identity changed')
                        if len(result.get('attempts',[]))!=1:raise ValueError('missing completed replay trace')
                        attempt=result['attempts'][0]
                        expected=dict(trace_lane=lane,actor_sha256=source['actor_sha256'],
                            normalizer_sha256=source['normalizer_sha256'],model_sha256=source['xml_sha256'],
                            snapshot_context_sha256=original['snapshot_context_sha256'],label=result['label'])
                        if any(attempt.get(k)!=v for k,v in expected.items()):
                            raise ValueError('replay trace identity/label changed')
                        arrays=lane_arrays(attempt)
                        if any(not np.isfinite(a).all() for a in arrays.values() if a.dtype.kind in 'fc'):
                            raise ValueError('nonfinite teacher replay trajectory')
                        n=len(arrays['done'])
                        if n!=attempt.get('steps') or not (arrays['done'][-1] or n>=horizon):
                            raise ValueError('incomplete teacher replay trajectory')
                        label,_=suffix_label(bool(arrays['valid_contact'][-1]),bool(arrays['physical_failure'][-1]),
                            bool(arrays['timeout'][-1]),bool(arrays['done'][-1]),n>=horizon)
                        if label!=result['label']:raise ValueError('replay terminal label mismatch')
                proof=self.root/'teachers'/f'{ordinal:04d}_layout_repeat.json'
                selected={**selected,'finite_replay_rejection':dict(source_labels=[0,0],
                    selected_labels=[1,0],complete_finite=True,repeat_receipt=str(proof),
                    repeat_receipt_sha256=file_sha(proof),search=search_results,repeat=results)}
        return selected

    def teacher_search(self,incumbent):
        if not self.panels['new_roots']:return {}
        if self.spec.get('teacher_execution')=='persistent_b1':
            from .teacher_worker import dispatch_teacher
            return dispatch_teacher(self,incumbent)
        from .worker import generator_template,generator_reference
        from .diffusion import restore_state,ddim_sample
        from .proposals import make_candidate_pool
        from .teacher import select_teacher,search_status
        from ..unified_envelope_snapshot import load_unified_envelope_snapshot
        import jax
        import jax.numpy as jp
        from .performance import measure
        with measure('model_actor_generator_load',backend=jax.default_backend(),component='generator'):
            net,template,identity=generator_template(generator_reference(self.spec),self.spec['seed'])
            state=restore_state(Path(incumbent['checkpoint_manifest']).parent,template,identity)
        generation=jax.jit(lambda observations,noise:ddim_sample(
            lambda x,o,k:net.apply(state['ema'],x,o,k),observations,noise)) if getattr(self,'persistent_teacher',False) else None
        generation_compiled=None
        self.teacher_in_progress=True
        source_results=self.evaluate('teacher_source',self.panels['new_roots'])
        output={};teacher_dir=self.root/'teachers';teacher_dir.mkdir(exist_ok=True)
        for ordinal,base in enumerate(source_results):
            rid=base['root_id'];self.active_teacher_root=rid;source_attempt=base['attempts'][0]
            disposition=source_recheck_disposition(base['label'])
            if disposition is not None:
                row=dict(root_id=rid,source_actor_sha256=self.source['actor_sha256'],**disposition)
                atomic_json(teacher_dir/f'{ordinal:04d}_result.json',row);output[rid]=row
                continue
            inherited=self.spec.get('recovery',{}).get('teachers',{}).get(rid)
            if inherited:
                if file_sha(inherited['path'])!=inherited['sha256']:raise ValueError('inherited teacher changed')
                row=read(inherited['path'])
                if inherited.get('proposal_path'):row['proposal_path']=inherited['proposal_path']
                if row['generator']!=incumbent or row['source_actor_sha256']!=self.source['actor_sha256']:
                    raise ValueError('inherited teacher model changed')
                read_teacher_traces({rid:row})
                row.update(historical_source_label=0,source_recheck_label=0,new_gain_eligible=True,training_eligible=True)
                atomic_json(teacher_dir/f'{ordinal:04d}_result.json',row);output[rid]=row
                continue
            snapshot=load_unified_envelope_snapshot(Path(base['snapshot']))
            actual=lane_arrays(source_attempt);obs=actual['actor_observation_before'][0]
            noise=np.stack([np.random.default_rng(stable_seed(self.spec['seed'],rid,cid)).normal(size=(16,4)) for cid in range(1,17)]).astype(np.float32)
            normalized=(jp.asarray(np.repeat(obs[None],16,axis=0))-state['normalizer']['mean'])/state['normalizer']['std']
            from .performance import measure
            if generation is not None and generation_compiled is None:
                with measure('ddim_compile',backend=jax.default_backend(),batch_size=16):
                    generation_compiled=generation.lower(normalized,jp.asarray(noise)).compile()
            with measure('ddim_generation',backend=jax.default_backend(),root=rid,batch_size=16):
                generated=jax.device_get(generation_compiled(normalized,jp.asarray(noise)) if generation is not None
                    else ddim_sample(lambda x,o,k:net.apply(state['ema'],x,o,k),normalized,jp.asarray(noise)))
            pool=make_candidate_pool(rid,actual['normalized_action_executed'][:16],generated,seed=self.spec['seed'],
                colored_noise_candidates=self.spec.get('teacher_colored_noise_candidates',15))
            poolpath=teacher_dir/f'{ordinal:04d}_proposals.npz'
            if not poolpath.exists():np.savez_compressed(poolpath,actions=pool['actions'],diffusion_noise=noise,colored_raw_draws=pool['colored_raw_draws'])
            full=self.spec.get('teacher_layout') in ('source_control_in_32_world_batch','source_control_in_candidate_batch')
            first=0 if full else 1
            candidates=[{**base,'index':cid,'candidate_id':cid} for cid in range(first,len(pool['actions']))]
            results=self.evaluate(f'teacher_{ordinal:04d}',candidates,prefixes=pool['actions'][first:],
                source_only=np.asarray([r['candidate_id']==0 for r in candidates],bool))
            if full and results[0]['label']!=0:
                conflict=dict(root_id=rid,teacher_status='not_scheduled' if results[0]['label']==1 else 'invalid',
                    reason='same_layout_source_recheck',source_recheck_label=results[0]['label'],
                    historical_source_label=0,new_gain_eligible=False,training_eligible=True,
                    source_actor_sha256=self.source['actor_sha256'])
                atomic_json(teacher_dir/f'{ordinal:04d}_result.json',conflict);output[rid]=conflict
                if results[0]['label'] is None:raise ValueError('unknown same-layout source recheck')
                continue
            scored=[] if full else [teacher_candidate(base,'source_only',0,snapshot.last_action)]
            scored.extend(teacher_candidate(r,pool['kinds'][r['candidate_id']],r['candidate_id'],snapshot.last_action) for r in results)
            selected=select_teacher(scored)
            row={'root_id':rid,'teacher_status':'searched_no_solution','source_actor_sha256':self.source['actor_sha256'],
                'generator':incumbent,'candidates':scored,'proposal_sha256':file_sha(poolpath),
                'historical_source_label':0,'source_recheck_label':0,'new_gain_eligible':True,'training_eligible':True}
            if selected is not None:
                cid=selected['candidate_id'];replay=self.replay_teacher(ordinal,candidates,pool['actions'][first:],cid,search_results=results)
                quarantine=quarantine_source_conflict(row,replay)
                if quarantine is not None:
                    quarantine['selected_candidate_id']=cid
                    atomic_json(teacher_dir/f'{ordinal:04d}_result.json',quarantine);output[rid]=quarantine
                    continue
                if replay.get('source_control_repeat_conflict'):
                    raise ValueError('unknown or failed teacher/control repeat; incomplete execution is not quarantine')
                rejected=reject_finite_replay(row,replay)
                if rejected is not None:
                    atomic_json(teacher_dir/f'{ordinal:04d}_result.json',rejected);output[rid]=rejected
                    continue
                verified={'candidate_id':cid,'full_success':replay['label']==1,
                    'batch_size':len(candidates),'selected_lane':cid-first,'layout':'same_as_search'}
                status=search_status(scored,expected_count=len(pool['actions']),verified=verified)
                row.update(teacher_status=status,selected_candidate_id=cid,verification=verified)
                if status!='verified_solution':
                    atomic_json(teacher_dir/f'{ordinal:04d}_result.json',row)
                    raise ValueError('teacher replay invalid; never use empty-demo fallback')
                a=lane_arrays(replay['attempts'][0]);a['action_origin']=np.where(a['action_origin_code']==1,'bridge_prefix','source_tail')
                m=trace_metadata(base,self.source,self.protocol,self.panels['splits'],'verified_teacher')
                m.update(teacher_status=status,verification_receipt_sha256=digest(replay),generator_checkpoint_manifest_sha256=incumbent['checkpoint_manifest_sha256'])
                trace_path=teacher_dir/f'{ordinal:04d}_verified.npz';np.savez_compressed(trace_path,**a)
                row['demo']={'count':len(a['done']),'path':str(trace_path),'sha256':file_sha(trace_path),'metadata':m}
            atomic_json(teacher_dir/f'{ordinal:04d}_result.json',row);output[rid]=row
        self.teacher_in_progress=False;self.active_teacher_root=None
        return output

    def nominal(self):
        d=self.root/'nominal';d.mkdir(exist_ok=False)
        spec={**self.runtime,'budget':400};atomic_json(d/'spec.json',spec)
        cost=self.child('nominal_source',['JIT/cli/run_pulse_exploration.py','--mode','baseline',
            '--spec',d/'spec.json','--output',d/'source'],400)
        status=read(d/'source/status.json');cost.update(charged_interactions=status['charged_interactions'],accounting='measured')
        atomic_json(self.root/'costs.json',self.costs)
        return {'rows':read(d/'source/candidates.json'),'results':read(d/'source/evaluation/results.json')}

    def student(self,teachers):
        from .data import export_student_demonstrations
        from ..iterative_probe_training import make_config
        from ..unified_policy_freeze import freeze_development_checkpoint
        from ..probe_bank import lock_probe_bank
        d=self.root/'student';d.mkdir(exist_ok=False)
        traces=read_teacher_traces(teachers)
        export_student_demonstrations(traces,d/'demo')
        run_id=self.root.name+'_student';config=d/'config.json'
        original=read(self.source['formal_config'])
        raw=make_config(self.root/'training_support.json',self.spec['source_frozen_policy'],self.runtime['bootstrap_config'],
            config,run_id,self.source['iteration']+1,128000,self.spec['seed'],checkpoints=[128000],
            panel_support_path=original['panel_support'],pending_fraction=original['pending_fraction'],
            reward_mode=original['reward_mode'],kl_control=original.get('kl_control'))
        manifest=d/'demo/manifest.json';anchors=self.root/'retention_support.json'
        raw['generative_bridge_student']={'schema':'jit_bridge_student_v1_1',
            'source_actor_sha256':self.source['actor_sha256'],'source_normalizer_sha256':self.source['normalizer_sha256'],
            'demo_manifest':{'path':str(manifest),'sha256':file_sha(manifest)},
            'demo_coefficient_start':.2,'demo_coefficient_end':.05,'retention_coefficient':.2,
            'demo_batch_size':256,'retention_batch_size':256,'retention':{'schema':'jit_source_action_retention_v1',
            'support':str(anchors),'support_file_sha256':file_sha(anchors),'coefficient':.2,'batch_size':256}}
        atomic_json(config,raw)
        cost=self.child('student_ppo',['JIT/cli/train_unified_from_pi0.py','--config',config,'--run-id',run_id],129600,
            extra_env={'JIT_RUN_ROOT':str(d/'training')})
        training=d/'training'/run_id;report=read(training/'formal_report.json')
        cost.update(charged_interactions=report['completed_training_transitions']+report['train_panel_interactions'],accounting='measured')
        atomic_json(self.root/'costs.json',self.costs)
        freeze_development_checkpoint(d/'frozen',config_path=config,checkpoint=training/'checkpoints/transition_128000',name=run_id)
        frozen=d/'frozen/frozen_unified_policy.json';policy=read(frozen)['policy']
        bank=read(self.runtime['bank']);bank={k:bank[k] for k in ('task','max_ticks','label_interaction_budget','max_candidates_per_process')}
        bank.update(version=run_id,members=[{'frozen_policy':str(frozen),'roles':['proposer','evaluator']}])
        lock_probe_bank(bank,d/'bank.json')
        return {'frozen_policy':str(frozen),'policy':policy,'bank':str(d/'bank.json'),
            'actor_sha256':policy['actor_sha256'],'normalizer_sha256':policy['normalizer_sha256'],
            **read(training/'bridge_demo_usage.json')}

    def write_feedback(self,after,decision):
        from .rewards import feedback,WEIGHTS
        source_round=Path(self.spec['source_round']);ledger=read(source_round/'source_ledger.json')
        if ledger['source']!=self.source['name']:raise ValueError('feedback source arrival ledger mismatch')
        rows=read(source_round/'bank_evaluation/results.json')
        selected={r['snapshot_context_sha256']:r['label'] for r in after}
        rechecked={r['snapshot_context_sha256']:r['label'] for r in self.evaluate('teacher_source',self.panels['new_roots'])}
        for r in rows:
            if r['snapshot_context_sha256'] in rechecked:r['source_recheck_label']=rechecked[r['snapshot_context_sha256']]
            initial=r['label'];r['initial_label']=initial
            r['learning_attempted']=r['snapshot_context_sha256'] in selected
            if r['learning_attempted']:r['label']=selected[r['snapshot_context_sha256']]
            r['successor_adopted']=decision['adopted']
        reward,mask,seen,parts=feedback(rows,ledger['cells'])
        atomic_json(self.root/'feedback.json',{'weights':WEIGHTS,'teacher_success_bonus':0.,
            'rewards':reward.tolist(),'eligible':mask.tolist(),'components':parts,
            'source_ledger_sha256':file_sha(source_round/'source_ledger.json'),
            'scope':'diagnostic feedback on reused TRAIN pulse roots; original acquisition already charged historically',
            'explorer_optimizer_updated':False,'new_arrival_cells':len(seen)-len(ledger['cells'])})

    def write_failure_matrix(self,error):
        if not self.teacher_in_progress:return
        from .outcomes import root_outcome,result_matrix
        recorded={}
        for path in (self.root/'teachers').glob('*_result.json'):
            record=read(path)
            if record.get('demo'):recorded[record['demo']['metadata']['root_id']]=record
            elif record.get('root_id'):recorded[record['root_id']]=record
        scheduled={r['root_id'] for r in self.panels['new_roots']};rows=[]
        for r in self.panels['original_pending']:
            rid=r['root_id'];status=recorded.get(rid,{}).get('teacher_status',
                'invalid' if rid==self.active_teacher_root else 'incomplete' if rid in scheduled else 'not_scheduled')
            rows.append(root_outcome({'root_id':rid,'teacher_status':status,'student_label':None,
                'student_adopted':None,'training_eligible':True,'error':repr(error) if rid==self.active_teacher_root else None}))
        atomic_json(self.root/'failed_root_outcomes.json',rows)
        atomic_json(self.root/'failed_result_matrix.json',result_matrix(rows))

    def run(self):
        try:
            with wall_deadline(self.spec['budgets']['max_wall_seconds']-(time.time()-self.started_unix)):
                return self._run()
        except BaseException as error:
            self.status('failed',error=repr(error));raise

    def _run(self):
        from .artifacts import load_corpus,save_corpus
        from .feedback_data import build_corpus
        from .outcomes import root_outcome,result_matrix
        try:
            self.journal.stage('semantic_smoke',{},self.smoke)
            continuation=self.spec.get('continuation')
            if continuation:
                from .artifacts import validate_generator_receipt
                validate_generator_receipt(continuation['generator']);load_corpus(continuation['corpus'])
            bootstrap=self.journal.stage('bootstrap',{},
                (lambda:{k:continuation[k] for k in ('corpus','dev_fixture','dev_fixture_sha256')}) if continuation else self.bootstrap)
            if file_sha(bootstrap['dev_fixture'])!=bootstrap['dev_fixture_sha256']:raise ValueError('dev fixture changed')
            incumbent=continuation['generator'] if continuation else self.generator('pretrain',bootstrap['corpus'],bootstrap['dev_fixture'])
            nominal=self.journal.stage('nominal',{},self.nominal)
            before={k:self.evaluate('before_'+k,self.panels[k]) for k in ('core','protected')}
            teachers=self.journal.stage('teacher_search',incumbent,lambda:self.teacher_search(incumbent))
            before['new_roots']=self.evaluate('teacher_source',self.panels['new_roots'])
            before['nominal']=nominal['results']
            student=self.journal.stage('student_training',teachers,lambda:self.student(teachers))
            after={k:self.evaluate('after_'+k,nominal['rows'] if k=='nominal' else self.panels[k],
                policy_bank=student['bank'],policy_name=student['policy']['name']) for k in before}
            decision=self.journal.stage('actor_acceptance',{'before':before,'after':after,'student':student},
                lambda:acceptance_decision({k:[r['label'] for r in v] for k,v in before.items()},
                    {k:[r['label'] for r in v] for k,v in after.items()},student))
            atomic_json(self.root/'actor_acceptance.json',decision)
            self.write_feedback(after['new_roots'],decision)
            usage=student['demo_samples_by_root'];total=sum(usage.values());rows=[]
            for r in after['new_roots']:
                t=teachers[r['root_id']]
                rows.append(root_outcome(dict(root_id=r['root_id'],teacher_status=t['teacher_status'],
                    historical_source_label=t['historical_source_label'],source_recheck_label=t['source_recheck_label'],
                    new_gain_eligible=t['new_gain_eligible'],training_eligible=True,teacher_reason=t.get('reason'),
                    student_label=r['label'],student_adopted=decision['adopted'],
                    student_checkpoint_sha256=student['actor_sha256'],student_normalizer_sha256=student['normalizer_sha256'],
                    n_direct_demo_examples_available=t.get('demo',{}).get('count',0),
                    n_direct_demo_samples_used=usage.get(r['root_id'],0),round_has_teacher_demo=any('demo' in x for x in teachers.values())),round_demo_samples_used=total))
            selected_roots={r['root_id'] for r in rows}
            for r in self.panels['original_pending']:
                if r['root_id'] not in selected_roots:
                    rows.append(root_outcome({'root_id':r['root_id'],'teacher_status':'not_scheduled',
                        'student_label':None,'student_adopted':decision['adopted'],
                        'evaluation_status':'outside_predeclared_pilot_panel','training_eligible':True,
                        'student_checkpoint_sha256':student['actor_sha256'],
                        'student_normalizer_sha256':student['normalizer_sha256']},round_demo_samples_used=total))
            atomic_json(self.root/'root_outcomes.json',rows);atomic_json(self.root/'result_matrix.json',result_matrix(rows))
            def export():
                from .series import corpus_history
                history=corpus_history(load_corpus(bootstrap['corpus']))
                actors=actor_traces(after['new_roots'],student['policy'],self.protocol,self.panels['splits'])
                corpus=build_corpus(history,read_teacher_traces(teachers),actors,adoption=decision,
                    expected={'model_sha256':self.source['xml_sha256'],'protocol_sha256':self.protocol},splits=self.panels['splits'])
                return save_corpus(corpus,self.root/'updated_corpus')
            corpus=self.journal.stage('corpus_update',{'decision':decision,'after':after,'teachers':teachers},export)
            selected=self.generator('incremental',corpus,bootstrap['dev_fixture'],incumbent=incumbent)
            if selected['status']=='skipped_no_new_data':selected={**incumbent,'update_status':'skipped_no_new_data'}
            bundle={'actor':student if decision['adopted'] else {'frozen_policy':self.spec['source_frozen_policy'],'policy':self.source},
                'generator':selected,'corpus':corpus,'actor_acceptance':decision,'iterations':1}
            self.journal.stage('complete_round',bundle,lambda:bundle)
            atomic_json(self.root/'current_source.json',bundle);self.status('completed')
            return bundle
        except BaseException as e:
            self.write_failure_matrix(e)
            self.status('failed',error=repr(e));raise


def read_teacher_traces(teachers):
    result=[]
    for row in teachers.values():
        if row['teacher_status']!='verified_solution':continue
        d=row['demo']
        if file_sha(d['path'])!=d['sha256']:raise ValueError('verified demonstration changed')
        with np.load(d['path'],allow_pickle=False) as a:result.append({'metadata':d['metadata'],'arrays':{k:a[k] for k in a.files}})
    return result


def start_notifications(manifest):
    """Use the existing JIT watcher and verify its actual heartbeat."""
    import os,subprocess,sys
    owner=manifest.get('notification_owner')
    if owner:
        record=read(owner['status_path'])
        argv=Path(f"/proc/{record['pid']}/cmdline").read_bytes().split(b'\0')
        if (str(owner['active_run']).encode() not in argv or record.get('delivery_errors')
            or not 0<=time.time()-record.get('checked_unix',0)<30):
            raise RuntimeError('campaign notification watcher unhealthy')
        return record  # Owner sends errors, ten-round progress, and final completion.
    root=Path(manifest['output']);repo=Path(manifest['repository']);state=root/'notifications'
    active=root/'ACTIVE_RUN.json'
    atomic_json(active,{'name':root.name,'lineage':str(root/'status.json')})
    state.mkdir(exist_ok=True)
    old=state/'notification_status.json'
    if old.exists():
        record=read(old)
        try:
            argv=Path(f"/proc/{record['pid']}/cmdline").read_bytes().split(b'\0')
            healthy=(str(active).encode() in argv and not record.get('delivery_errors')
                and 0<=time.time()-record.get('checked_unix',0)<30)
        except (OSError,KeyError):healthy=False
        if healthy:return record
    env={**os.environ,'PYTHONPATH':str(repo/'JIT/src'),'JAX_PLATFORMS':'cpu'}
    env.setdefault('DBUS_SESSION_BUS_ADDRESS',f'unix:path=/run/user/{os.getuid()}/bus')
    # A successful notify-send is delivery evidence, not proof the user saw it.
    subprocess.run(['notify-send','--app-name=JIT','JIT 训练监视已启动',
        '双向学习 '+manifest.get('protocol_version','1.1')+'：错误和整轮结束将单独通知。'],env=env,check=True,timeout=10)
    with (state/'watcher.log').open('ab') as log:
        process=subprocess.Popen([sys.executable,str(repo/'JIT/cli/watch_run_errors.py'),
            '--active-run',str(active),'--state-dir',str(state)],cwd=repo,env=env,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(50):
        if process.poll() is not None:raise RuntimeError('notification watcher exited')
        if old.exists():
            record=read(old)
            if record.get('pid')==process.pid and not record.get('delivery_errors'):
                atomic_json(state/'launch_receipt.json',{'pid':process.pid,'initial_notification_delivery':'notify-send exit0',
                    'heartbeat_checked':True,'active_run':str(active)})
                return record
        time.sleep(.1)
    raise RuntimeError('notification watcher heartbeat missing')
