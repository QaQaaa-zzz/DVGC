"""Finite BC+keep student stage; independently launched, never starts PPO."""
from pathlib import Path
import os,time,subprocess,pickle
import numpy as np
from .retention_repair import read,write,sha
from .retention_repair_execution import PY

BASE=Path('/home/qy/DVGC/JIT/runs/experiments/retention_next_20261010/D2_002')


def budget(solver_roots,train_roots=8):
    b=dict(reset_GPU=0,DEV_baselines=2*64*400,snapshot_nodes=5*(solver_roots+train_roots)*17*400,
        B_full_DEV=4*64*400,B_warning_repeat=4*2*64*400,
        selected_train_repeat=2*train_roots*17*400,R5_focus_repeat=2*3*17*400,
        selected_train_composites=4*2*17*400)
    b.update(total=sum(b.values()),B_updates=2000,PPO_updates=0,hard_cap=2000000,
        prior_charged=12900,wall_seconds=43200)
    b['D2_cumulative_maximum']=b['total']+b['prior_charged']
    if b['D2_cumulative_maximum']>b['hard_cap']:raise ValueError('D2 cumulative layout over cap')
    return b


def warning_cells(baseline,candidate):
    return [g for g in ('B','D') if np.mean(baseline[g])-np.mean(candidate[g])>.05+1e-9]


def select_B(scores,eligible):
    valid=[k for k in scores if eligible[k]]
    if not valid:raise ValueError('baseline must remain eligible')
    return min(valid,key=lambda k:(-scores[k],k))


def prepare(output,repository,previous_zero=None):
    code=Path(repository).resolve();root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    original=read(BASE/'plan.json');supp=read(BASE/'report_004/formal_preparation.json');locks={}
    for f,h in supp['locks'].items():
        if sha(f)!=h:raise ValueError('D2 supplement changed')
        locks[f]=h
    from .retention_next import identities
    identities(original['models'])
    for f,h in original['locks'].items():
        f=Path(f)
        # Preserve old execution version; current code is separately pinned below.
        if f.is_relative_to(Path(original['code'])) and str(f).endswith('.py'):
            import hashlib
            blob=subprocess.check_output(['git','show',original['code_revision']+':'+str(f.relative_to(original['code']))],cwd=code)
            if hashlib.sha256(blob).hexdigest()!=h:raise ValueError('old collected code drift')
            locks[str(f)]=sha(f)
        else:
            if sha(f)!=h:raise ValueError('old D2 inputs changed')
            locks[str(f)]=h
    for f in [BASE/'plan.json',BASE/'keep_receipt.json',BASE/'gpu_micro/verification.json',BASE/'report_004/formal_preparation.json']:
        locks[str(f)]=sha(f)
    if read(BASE/'gpu_micro/verification.json')['phase']!='passed':raise ValueError('real GPU micro prerequisite')
    keep=read(BASE/'keep_receipt.json');locks[keep['path']]=keep['sha256']
    inventory=read(supp['reset_pool_inventory']) # read-only; PPO reset is not a BC gate
    for group in ('pi0_success_snapshots','teacher_recoverable'):
        for entry in inventory[group]['entries']:
            f=Path(entry['snapshot'])/'snapshot.pkl';locks[str(f)]=sha(f)
    old=read(original['D1_receipt']);d1root=Path(original['D1_receipt']).parent
    if old['phase']!='completed' or sha(original['D1_receipt'])!=original['D1_receipt_sha256']:raise ValueError('D1 receipt changed')
    allroots=read(d1root/'selected_roots.json');allresults=read(d1root/'teacher_results.json')
    results={r['root_id']:r for r in allresults};solvers=[];train=[]
    for r in allroots:
        t=results[r['root_id']]
        if r['role']=='TRAIN' and t['methods']['G']['repeat_verified']:
            train.append(dict(root=r,teacher=t,lane=t['methods']['G']['selected_candidate_id']))
            locks[t['proposals']]=t['proposals_sha256']
        if r['role']!='SOLVER_DEV':continue
        if r['pi0_labels']!=[0,0,0]:raise ValueError('solver root not stable baseline')
        t=results[r['root_id']];solvers.append(dict(root=r,teacher=t,lane=t['methods']['G']['selected_candidate_id'] if t['methods']['G']['repeat_verified'] else 1))
        locks[t['proposals']]=t['proposals_sha256']
    if len(train)!=8 or len(solvers)!=7:raise ValueError('expected locked TRAIN8 and SOLVER_DEV7')
    write(root/'solver_roots.json',solvers);write(root/'train_roots.json',train)
    cases=read(supp['new_DEV']);initial=root/'DEV_initial.npz';requests=root/'DEV_requests.npz'
    np.savez_compressed(initial,qpos=np.array([c['qpos'] for c in cases],np.float32),qvel=np.array([c['qvel'] for c in cases],np.float32))
    table=np.zeros((400,64,4),np.float32)
    for i,c in enumerate(cases):table[c['onset']:c['onset']+3,i]=c['request']
    np.savez_compressed(requests,requested=table,onsets=np.array([c['onset'] for c in cases],np.int32))
    template=read(BASE/'keep_spec.json')
    batches=[]
    for key,m in original['models'].items():
        spec={**template,'bank':m['bank'],'proposer':m['proposer'],'num_envs':64,'role':'DEV','seed':1010258103,'initial_state_bank':str(initial),'frozen_request_table':dict(path=str(requests),sha256=sha(requests))}
        sp=root/f'DEV_{key}_spec.json';write(sp,spec);batches.append(dict(name='DEV_'+key,spec=str(sp),capacity=64,horizon=400,model=key,scored=64,cases=cases));locks[str(sp)]=sha(sp)
    for entry in train+solvers:
        f=Path(entry['root']['snapshot'])/'snapshot.pkl';locks[str(f)]=sha(f)
        for method in entry['teacher']['methods'].values():
            for attempt in method.get('verified_attempts',[]):
                if sha(attempt['trace'])!=attempt['trace_sha256']:raise ValueError('teacher trace drift')
                locks[attempt['trace']]=attempt['trace_sha256']
    for f in (root/'solver_roots.json',root/'train_roots.json',initial,requests):locks[str(f)]=sha(f)
    for name in ('retention_b.py','retention_reset.py','pulse_exploration_runtime.py','generative_bridge/rollout.py','generative_bridge/warmup.py'):
        f=code/'JIT/src/jit_dvgc'/name;locks[str(f)]=sha(f)
    start=(BASE/'keep_collection/behavior.msgpack').stat().st_mtime
    remaining=43200-(time.time()-start)
    if remaining<=0:raise ValueError('original D2 wall budget exhausted')
    layout=budget(len(solvers));layout['remaining_wall_seconds']=remaining
    reused=None
    if previous_zero is not None:
        oldp=read(previous_zero);oldroot=Path(oldp['output']);oldrows=read(oldroot/'B_checkpoint_results.json')
        if set(oldrows)!= {'0'} or (oldroot/'B/metrics.jsonl').exists():raise ValueError('zero-update-only retry; no BC resume implemented')
        if oldp['models']!=original['models'] or oldp['demo_manifest']!=original['demo_manifest']:raise ValueError('reuse source identity changed')
        for oldbatch,newbatch in zip(oldp['batches'],batches):
            if oldbatch['cases']!=newbatch['cases']:raise ValueError('new DEV cases differ from physical baseline')
            from .retention_repair_execution import verify
            verify(oldbatch,oldroot)
        for f,h in oldp['locks'].items():
            f=Path(f)
            if f.is_relative_to(code) and f.suffix=='.py':
                import hashlib
                blob=subprocess.check_output(['git','show',oldp['code_revision']+':'+str(f.relative_to(code))],cwd=code)
                if hashlib.sha256(blob).hexdigest()!=h:raise ValueError('old execution source drift')
            elif sha(f)!=h:raise ValueError('reused input drift '+str(f))
        for row in oldrows['0']['train']+oldrows['0']['solver']:
            a=row['combinations']['independent']['attempt']
            if sha(a['trace'])!=a['trace_sha256']:raise ValueError('reused zero trace drift')
            locks[a['trace']]=a['trace_sha256']
        for f in (Path(previous_zero),oldroot/'B_checkpoint_results.json',oldroot/'costs.json'):
            locks[str(f)]=sha(f)
        prior=sum(x['charged'] for x in read(oldroot/'costs.json'))+oldp['budget']['prior_charged']
        layout['prior_charged']=prior;layout['DEV_baselines']=0;layout['snapshot_nodes']-=15*17*400
        layout['total']=sum(layout[k] for k in ('reset_GPU','DEV_baselines','snapshot_nodes','B_full_DEV','B_warning_repeat','selected_train_repeat','R5_focus_repeat','selected_train_composites'))
        layout['D2_cumulative_maximum']=prior+layout['total']
        reused=dict(plan=str(previous_zero),zero_checkpoint=str(oldroot/'B_checkpoint_results.json'),DEV={x['model']:str(oldroot/x['name']) for x in oldp['batches']},reason='failed fixed-probe import before any BC update; continuous fresh pi0 BC, not optimizer resume')
    plan=dict(schema='jit_retention_B_v1',stage='D2_B',output=str(root),code=str(code),code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip(),locks=locks,
        source_plan=str(BASE/'plan.json'),source_supplement=str(BASE/'report_004/formal_preparation.json'),models=original['models'],indices=original['indices'],budget=layout,
        reused_zero=reused,tensorboard_port=6027 if reused else 6026,original_D2_started_unix=start,keep_receipt=str(BASE/'keep_receipt.json'),demo_manifest=original['demo_manifest'],reset_inventory=supp['reset_pool_inventory'],batches=batches,
        BC=original['BC'],BC_selection=supp['BC_selection'],solver_roots=str(root/'solver_roots.json'),train_roots=str(root/'train_roots.json'),seed=1010267101,training_updates=2000,ppo_updates=0,E_G_updates=0,
        learner_last=None,best_dev_candidate=original['models']['R5'],published_policy=None,authorized_stage='DEV baselines, finite B only',reset_gate=False,diagnostic_train_roots=[x['root']['root_id'] for x in train if x['root']['R5_label']!=1]+[next(x['root']['root_id'] for x in train if x['root']['R5_label']==1)],automatic_next_stage=False)
    write(root/'budget_dry_run.json',layout);write(root/'source_lock.json',dict(identities=original['identities'],locks=locks));write(root/'plan.json',plan)
    write(root/'status.json',dict(phase='prepared',stage='D2_B',charged_interactions=0,completed_supervised_updates=0,planned_supervised_updates=2000,PPO_updates=0))
    return root/'plan.json'


class Run:
    def __init__(self,path):
        self.p=read(path);self.path=Path(path);self.root=Path(self.p['output']);self.costs=[]
        self.env=dict(os.environ,PYTHONPATH=str(Path(self.p['code'])/'JIT/src'),JAX_PLATFORMS='cuda,cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    def status(self,phase,stage,**extra):
        write(self.root/'status.json',dict(phase=phase,stage=stage,pid=os.getpid(),updated_unix=time.time(),charged_interactions=sum(c['charged'] for c in self.costs),D2_prior_charged=self.p['budget']['prior_charged'],planned_supervised_updates=2000,PPO_updates=0,E_G_updates=0,completed_supervised_updates=getattr(self,'current_update',0),**extra))
    def child(self,name,command,maximum):
        if time.time()-self.p['original_D2_started_unix']>=43200:raise TimeoutError('D2 original12h budget')
        if self.p.get('schema')==COVERAGE_SCHEMA and sum(c['charged'] for c in self.costs)+maximum>self.p['budget']['total']:
            raise ValueError('coverage stage physics reserve exceeds declared958000')
        if self.p['budget']['prior_charged']+sum(c['charged'] for c in self.costs)+maximum>2000000:raise ValueError('D2 physics reserve before dispatch')
        c=dict(stage=name,charged=maximum,reservation=maximum,phase='running');self.costs.append(c);write(self.root/'costs.json',self.costs);self.status('running',name)
        with (self.root/(name+'.log')).open('x') as f:
            proc=subprocess.Popen(command,cwd=self.p['code'],env=self.env,stdout=f,stderr=subprocess.STDOUT)
            self.status('running',name,child_pid=proc.pid)
            try:rc=proc.wait(timeout=max(1,43200-(time.time()-self.p['original_D2_started_unix'])))
            except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=15);raise
        if rc:raise RuntimeError(f'{name} exited{rc}; reservation retained; no automatic retry')
        receipt=read(self.root/name/'status.json')
        if receipt['phase']!='completed':raise ValueError('incomplete child receipt')
        c.update(phase='completed',charged=receipt.get('charged_interactions',maximum));write(self.root/'costs.json',self.costs)
        return self.root/name


def reference(plan,step):
    path=Path(plan['output'])/'B'/f'update_{step:04d}.pkl';m=plan['models']['pi0']['policy']
    return dict(path=str(path),sha256=sha(path),source_actor_sha256=m['actor_sha256'],normalizer_sha256=m['normalizer_sha256'])


def evaluate_checkpoint(run,step):
    p=run.p;root=run.root;model=p['models']['pi0'];ref=reference(p,step)
    batch=p['batches'][0];spec=read(batch['spec']);spec['warmup_initializer']=ref;sp=root/f'B{step:04d}_DEV_spec.json';write(sp,spec)
    name=f'B{step:04d}_DEV';out=run.child(name,[PY,str(Path(p['code'])/'JIT/cli/run_pulse_exploration.py'),'--mode','collect','--spec',str(sp),'--output',str(root/name)],25600)
    from .retention_repair_execution import verify
    verify({**batch,'name':name,'spec':str(sp)},root)
    # Baseline exact zero weights reuse complete pi0 DEV, avoiding duplicate physics.
    return full_results(p,out),solver_combinations(run,step,zero_only=True),solver_combinations(run,step,zero_only=True,role='TRAIN')


def full_results(plan,out):
    from .retention_repair_report import CachedArchive
    cases=plan['batches'][0]['cases'];labels={g:[] for g in 'ABCD'}
    with CachedArchive(Path(out)/'prefixes.npz') as f:
        for lane,c in enumerate(cases):labels[c['group']].append(int(np.any(f['success'][:,lane]&f['prefix_mask'][:,lane])))
        rewards=float(f['reward'][f['prefix_mask']].mean())
    return dict(labels=labels,reward_per_transition=rewards,output=str(out))


def solver_combinations(run,step,zero_only=False,role='SOLVER_DEV',repeat='',model='pi0',only_roots=None):
    p=run.p;roots=read(p['solver_roots'] if role=='SOLVER_DEV' else p['train_roots']);summary=[]
    if only_roots is not None:roots=[x for x in roots if x['root']['root_id'] in only_roots]
    for ordinal,entry in enumerate(roots):
        row=entry['root'];t=entry['teacher'];g=t['methods']['G'];lane=entry['lane'];combos={}
        variants=['independent'] if zero_only else ['student_H16_pi0']+(['teacher_H16_student'] if g['repeat_verified'] else [])
        for kind in variants:
            rows=[{**row,'index':i,'candidate_id':i} for i in range(17)];name=f'B{step:04d}_{role}_root{ordinal:02d}_{kind}{repeat}_{model}'
            rowsfile=run.root/(name+'_rows.json');write(rowsfile,rows)
            spec=dict(bank=p['models']['pi0']['bank'],proposer=p['models']['pi0']['proposer'],order=[p['models']['pi0']['proposer']],horizon=400,budget=6800,
                candidates=str(rowsfile),full_matrix=True,record_actor_preobservations=True,record_retention_diagnostics=True,reward_mode='original_all_phases',success_criterion='stable_forward_recovery',preserve_snapshot_episode_context=True,
                suffix_rng_count=17,suffix_rng_indices=list(range(17)))
            ref=reference(p,step)
            if kind=='student_H16_pi0':spec.update(closed_loop_prefix_policy=p['models']['pi0']['proposer'],closed_loop_prefix_initializer=ref,closed_loop_prefix_source_only=[True]+[False]*16)
            elif model=='R5':
                spec.update(bank=p['models']['R5']['bank'],proposer=p['models']['R5']['proposer'],order=[p['models']['R5']['proposer']])
            else:spec['warmup_initializer']=ref
            if kind=='teacher_H16_student':
                with np.load(t['proposals']) as f:actions=f['G']
                dest=run.root/(name+'_actions.npz');np.savez_compressed(dest,actions=actions,source_only=np.arange(17)==0,root_contexts=np.array([row['snapshot_context_sha256']]*17))
                spec['bridge_action_plan']=dict(path=str(dest),sha256=sha(dest))
            sp=run.root/(name+'_spec.json');write(sp,spec)
            out=run.child(name,[PY,str(Path(p['code'])/'JIT/cli/run_pulse_exploration.py'),'--mode','evaluate','--spec',str(sp),'--output',str(run.root/name)],6800)
            results=read(out/'results.json');combos[kind]=dict(label=results[lane]['label'],lane=lane,attempt=results[lane]['attempts'][0],all_lane_labels=[r['label'] for r in results])
        combos['teacher_H16_pi0']=dict(label=1,attempt=g['verified_attempts'][-1],reused=True) if g['repeat_verified'] else dict(label=None,reason='G has no verified winner; NA')
        if not g['repeat_verified']:combos['teacher_H16_student']=dict(label=None,reason='no fixed verified teacher; NA')
        summary.append(dict(root_id=row['root_id'],role=role,pi0_labels=row['pi0_labels'],R5_label=row['R5_label'],G_verified=g['repeat_verified'],Noise_verified=t['methods']['Noise']['repeat_verified'],combinations=combos))
        write(run.root/f'B{step:04d}_{role}_combinations{repeat}_{model}.json',summary)
    return summary


def audit(path):
    p=read(path)
    coverage=p.get('schema')==COVERAGE_SCHEMA
    if p['stage']!=('D2_B_keep_coverage' if coverage else 'D2_B') or p['automatic_next_stage'] or p['ppo_updates'] or p['E_G_updates']:raise ValueError('B-only authorization contract')
    for f,h in p['locks'].items():
        if sha(f)!=h:raise ValueError('input drift '+f)
    if len(read(p['train_roots']))!=8 or len(read(p['solver_roots']))!=7:raise ValueError('root count')
    extra=audit_keep_coverage(p) if coverage else {}
    return dict(phase='passed',budget=p['budget'],source='pi0',normalizer_critic_frozen=True,automatic_PPO=False,**extra)


def reward_http(root,step,port):
    import urllib.request,json
    last=None
    for _ in range(20):
        try:
            url=f'http://localhost:{port}/data/plugin/scalars/scalars?run=.&tag=DEV%2FB_reward_per_transition'
            values=json.loads(urllib.request.urlopen(url,timeout=3).read())
            if any(int(v[1])==step for v in values):
                write(Path(root)/f'TensorBoard_reward_http_{step:04d}.json',dict(url=url,values=values,verified_step=step));return
        except Exception as e:last=repr(e)
        time.sleep(1)
    raise RuntimeError('current B DEV reward scalar not visible '+str(last))


def run(path,authorization=None):
    audit(path)
    p=read(path);root=Path(p['output'])
    for f,h in p['locks'].items():
        if sha(f)!=h:raise ValueError('input drift '+f)
    if read(root/'status.json')['phase']!='prepared':raise ValueError('no implicit rerun')
    coverage=p.get('schema')==COVERAGE_SCHEMA
    if coverage:
        grant=require_coverage_execution(path,authorization)
        claim_coverage_stage(root,'execution',path)
        write(root/'execution_authorization.json',dict(authorization=str(Path(authorization).resolve()),sha256=sha(authorization),grant=grant))
    r=Run(path);write(root/'ACTIVE_RUN.json',dict(execution=str(root/'status.json'),lineage=str(root/'status.json'),name='retention D2 finite BC+keep'))
    with (root/'watcher.log').open('x') as f:w=subprocess.Popen([PY,str(Path(p['code'])/'JIT/cli/watch_run_errors.py'),'--active-run',str(root/'ACTIVE_RUN.json'),'--state-dir',str(root/'notifications')],cwd=p['code'],env=r.env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    with (root/'tensorboard.log').open('x') as f:
        tb=subprocess.Popen([PY,'-m','tensorboard.main','--logdir',str(root/'tensorboard'),'--port',str(p['tensorboard_port']),'--host','0.0.0.0','--reload_interval','1'],env=r.env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    write(root/'launch.json',dict(pid=os.getpid(),watcher_pid=w.pid,tensorboard_pid=tb.pid,tensorboard_url=f"http://localhost:{p['tensorboard_port']}",started_unix=time.time(),plan=str(path)))
    try:
        if coverage:collect_coverage_keep(r)
        for b in ([] if p['reused_zero'] else p['batches']):
            out=r.child(b['name'],[PY,str(Path(p['code'])/'JIT/cli/run_pulse_exploration.py'),'--mode','collect','--spec',b['spec'],'--output',str(root/b['name'])],25600)
            from .retention_repair_execution import verify
            verify(b,root)
        # BC worker owns updates and inherited cost ledger while checkpoint callbacks
        # launch bounded GPU evaluation subprocesses; no additional student exists.
        r.status('running','BC+keep');write(root/'pre_B_costs.json',r.costs)
        with (root/'B.log').open('x') as f:
            subprocess.run([PY,str(Path(p['code'])/'JIT/cli/run_retention_b.py'),'worker-B','--plan',str(path)],cwd=p['code'],env=r.env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=max(1,43200-(time.time()-p['original_D2_started_unix'])))
        r.costs=read(root/'costs.json');state=read(root/'B_completed.json')
        r.current_update=state['completed_updates']
        r.status('completed',p['stage'],result='B complete or confirmed retention stop; no automatic PPO',learner_last=state['learner_last'],stage_candidate=state['candidate'],B_selected_candidate=state['candidate'],best_dev_candidate=p['models']['R5'],published_policy=None)
    except BaseException as e:
        if (root/'costs.json').exists():r.costs=read(root/'costs.json')
        if (root/'B/warmup_status.json').exists():r.current_update=read(root/'B/warmup_status.json')['completed_updates']
        r.status('failed','D2_B',error=repr(e));raise


def reset_worker(path):
    if read(path).get('schema')==COVERAGE_SCHEMA:raise ValueError('keep coverage has no PPO reset worker')
    p=read(path);root=Path(p['output'])/'reset_GPU';root.mkdir(exist_ok=False)
    import jax,jax.numpy as jp
    if jax.default_backend()!='gpu':raise ValueError('real GPU reset check')
    from .unified_formal import build_unified_formal_environment
    from .retention_reset import FourPoolReset,group_indices
    from .generative_bridge.worker import source_payload
    from .ppo import make_checkpoint_policy
    from .continuation.device_rollout import stack_worlds,prepare_parallel_worlds
    from .pulse_exploration_runtime import freeze_inactive_worlds
    pol,payload=source_payload(p['models']['pi0']['frozen_policy']);_,_,env=build_unified_formal_environment(Path(pol['formal_config']))
    env._training_action_pulse=None;adapter=FourPoolReset(env,read(p['reset_inventory']));adapter.install()
    key=jax.random.PRNGKey(1010267102);states=[adapter.reset_group(jax.random.fold_in(key,i),jp.asarray(i//2,jp.int32)) for i in range(8)]
    initial=prepare_parallel_worlds(stack_worlds(states),env,8);policy=make_checkpoint_policy(env,payload,deterministic=True)
    def rollout(s):
        def one(carry,t):
            state,alive=carry;keys=jax.random.split(jax.random.fold_in(key,t),8);action=jax.vmap(policy)(state.obs,keys)[0]
            advanced=jax.vmap(adapter.step)(state,jp.where(alive[:,None],action,0));advanced=freeze_inactive_worlds(advanced,state,alive)
            record=dict(qpos=advanced.data.qpos,observation=advanced.obs['state'],group=state.info['retention_reset_group'],active=alive,finite=jp.all(jp.isfinite(advanced.data.qpos),axis=-1),time=advanced.data.time)
            return (advanced,alive&~advanced.done.astype(bool)),record
        return jax.lax.scan(one,(s,jp.ones(8,bool)),jp.arange(25))
    (_,alive),tape=jax.device_get(jax.jit(rollout)(initial));np.savez_compressed(root/'reset_trace.npz',**tape)
    counts=np.bincount(np.asarray(group_indices(jax.random.PRNGKey(17),4096)),minlength=4)
    if not tape['finite'].all() or not np.all(np.abs(counts/4096-.25)<.03):raise ValueError('four-pool reset GPU invariant')
    write(root/'verification.json',dict(phase='passed',real_GPU=True,forced_reset_counts=[2]*4,random_draw_counts=counts.tolist(),actual_transitions_per_group=[int(tape['active'][:,i*2:(i+1)*2].sum()) for i in range(4)],
        charged_transitions_per_group=[50]*4,charged_interactions=200,full_snapshot_history_and_observation_checked=True,canonical_context_reconstruction=True,no_pulse=True,entries=adapter.entries,trace_sha256=sha(root/'reset_trace.npz'),claim='engineering reset validation only'))
    write(root/'status.json',dict(phase='completed',charged_interactions=200,training_updates=0))


def B_worker(path):
    declared=read(path);coverage=declared.get('schema')==COVERAGE_SCHEMA
    if coverage:
        # Direct worker invocation must not bypass permission, clock, or file locks.
        audit(path)
        receipt=read(Path(declared['output'])/'execution_authorization.json')
        if sha(receipt['authorization'])!=receipt['sha256']:raise ValueError('authorization drift')
        require_coverage_execution(path,receipt['authorization'])
        worker_keep_inputs(declared)
        claim_coverage_stage(Path(declared['output']),'worker_B',path)
    import jax
    if jax.default_backend()!='gpu':raise ValueError('B requires real GPU')
    from torch.utils.tensorboard import SummaryWriter
    from .generative_bridge.worker import source_payload
    from .generative_bridge.student import load_retention_traces
    from .generative_bridge.warmup import warmup_actor
    from .ppo import make_network_factory
    from brax.training.acme import running_statistics
    from .handoff_bank import pytree_sha256
    r=Run(path);r.costs=read(r.root/'pre_B_costs.json');p=r.p;root=r.root;rows={};scores={};eligible={};writer=SummaryWriter(str(root/'tensorboard'))
    pol,payload=source_payload(p['models']['pi0']['frozen_policy']);net=make_network_factory()({'state':76,'privileged_state':106},4,preprocess_observations_fn=running_statistics.normalize)
    keep_path=worker_keep_inputs(p);keep_receipt=read(keep_path)
    keep=load_retention_traces(keep_receipt);demo=read(p['demo_manifest']);baseline=full_results(p,Path(p['reused_zero']['DEV']['pi0']) if p['reused_zero'] else root/'DEV_pi0')
    write(root/'worker_inputs.json',dict(schema=p['schema'],keep_receipt=keep_path,keep_receipt_sha256=sha(keep_path),keep_data=keep_receipt['path'],keep_data_sha256=keep_receipt['sha256'],demo_manifest=p['demo_manifest'],demo_manifest_sha256=sha(p['demo_manifest']),seed=p['seed'],initializer='original_pi0_fresh_Adam_RNG',original_D2_started_unix=p['original_D2_started_unix']))
    writer.add_scalar('DEV/pi0_reward_per_transition',baseline['reward_per_transition'],0);writer.add_scalar('DEV/R5_reward_per_transition',full_results(p,Path(p['reused_zero']['DEV']['R5']) if p['reused_zero'] else root/'DEV_R5')['reward_per_transition'],0);writer.flush()
    def metrics(row):
        r.current_update=row['update']
        for k,v in row.items():writer.add_scalar('BC/'+k,v,row['update'])
        if row['update']%10==0:writer.flush();r.status('running','BC+keep')
    def probe(step,params):
        out=fixed_train_probe(net,params,(payload.observation_normalizer,payload.actor_params),demo,keep,keep_receipt)
        write(root/f'B{step:04d}_TRAIN_probes.json',out);return out
    def checkpoint(step,params):
        r.current_update=step
        if pytree_sha256(params[0])!=p['models']['pi0']['policy']['normalizer_sha256'] or pytree_sha256(params[2])!=pytree_sha256(payload.critic_params):raise ValueError('BC frozen statistics or critic changed')
        if step==0 and pytree_sha256(params[1])!=p['models']['pi0']['policy']['actor_sha256']:raise ValueError('BC did not start from original pi0')
        if step==0:
            full=baseline
            if p['reused_zero'] and not coverage:
                zero=read(p['reused_zero']['zero_checkpoint'])['0'];solver=zero['solver'];train=zero['train']
            else:
                solver=solver_combinations(r,0,zero_only=True);train=solver_combinations(r,0,zero_only=True,role='TRAIN')
            warnings=[];confirmed=[]
        else:
            full,solver,train=evaluate_checkpoint(r,step);warnings=warning_cells(baseline['labels'],full['labels']);confirmed=[]
            if warnings:
                spec=read(root/f'B{step:04d}_DEV_spec.json');sp=root/f'B{step:04d}_repeat_DEV_spec.json';write(sp,spec)
                name=f'B{step:04d}_repeat_DEV';out=r.child(name,[PY,str(Path(p['code'])/'JIT/cli/run_pulse_exploration.py'),'--mode','collect','--spec',str(sp),'--output',str(root/name)],25600)
                from .retention_repair_execution import verify
                verify({**p['batches'][0],'name':name,'spec':str(sp)},root)
                repeated=full_results(p,out)
                name=f'B{step:04d}_repeat_pi0_DEV';out=r.child(name,[PY,str(Path(p['code'])/'JIT/cli/run_pulse_exploration.py'),'--mode','collect','--spec',p['batches'][0]['spec'],'--output',str(root/name)],25600)
                verify({**p['batches'][0],'name':name},root)
                repeated_pi0=full_results(p,out)
                confirmed=[g for g in warnings if g in warning_cells(repeated_pi0['labels'],repeated['labels'])]
                write(root/f'B{step:04d}_warning_repeat.json',dict(warned=warnings,confirmed=confirmed,original=full,repeated=repeated,baseline=baseline,repeated_pi0=repeated_pi0))
        labels=full['labels'];old=baseline['labels'];cells={g:dict(n=len(labels[g]),success=sum(labels[g]),N01=sum(not a and b for a,b in zip(old[g],labels[g])),N10=sum(a and not b for a,b in zip(old[g],labels[g])),net=sum(labels[g])-sum(old[g])) for g in 'ABCD'}
        score=sum(x['combinations']['independent']['label']==1 for x in solver);scores[step]=score;eligible[step]=not bool(confirmed)
        rows[step]=dict(update=step,cells=cells,solver_success=score,solver_n=len(solver),warning_cells=warnings,confirmed_regression=confirmed,eligible=eligible[step],full=full,solver=solver,train=train,train_success=sum(x['combinations']['independent']['label']==1 for x in train),
            actor_sha256=pytree_sha256(params[1]),normalizer_sha256=pytree_sha256(params[0]),critic_sha256=pytree_sha256(params[2]))
        write(root/'B_checkpoint_results.json',rows)
        writer.add_scalar('DEV/B_reward_per_transition',full['reward_per_transition'],step)
        writer.add_scalar('SOLVER_DEV/independent_success',score,step)
        for g,c in cells.items():
            for k in ('success','N01','N10'):writer.add_scalar('DEV/'+g+'/'+k,c[k],step)
        writer.flush();reward_http(root,step,p['tensorboard_port']);r.status('running','B_checkpoint_checked')
        return bool(confirmed)
    try:
        warmup_actor(net,(payload.observation_normalizer,payload.actor_params,payload.critic_params),(payload.observation_normalizer,payload.actor_params),demo,keep,root/'B',seed=p['seed'],updates=2000,batch_size=256,full_learner=True,actual_gradient_audit=True,metrics_callback=metrics,checkpoint_callback=checkpoint,probe_callback=probe)
        status=read(root/'B/warmup_status.json');selected=select_B(scores,eligible);completed=status['completed_updates']
        diagnostic_step=selected if coverage else selected if selected>0 else min((k for k in rows if k>0),key=lambda k:(-rows[k]['train_success'],k))
        diagnostics=[] if coverage else solver_combinations(r,diagnostic_step,role='TRAIN',only_roots=p['diagnostic_train_roots'])
        confirmations=[solver_combinations(r,diagnostic_step,zero_only=True,role='TRAIN',repeat=f'_confirm{i}') for i in range(2)]
        focus=[x['root']['root_id'] for x in read(p['train_roots']) if x['root']['R5_label']!=1]
        R5_confirmations=[] if coverage else [solver_combinations(r,diagnostic_step,zero_only=True,role='TRAIN',repeat=f'_confirm{i}',model='R5',only_roots=focus) for i in range(2)]
        write(root/'selected_diagnostics.json',dict(selected_update=selected,diagnostic_update=diagnostic_step,TRAIN_confirmation_update=diagnostic_step,diagnostic_rule='selected nonzero; if baseline wins tie, earliest maximal TRAIN absorption nonzero is diagnostic only',combinations=diagnostics,TRAIN_confirmations=confirmations,R5_focus_confirmations=R5_confirmations))
        learner=root/'B'/f'learner_update_{completed:04d}.json';chosen=root/'B'/f'update_{selected:04d}.pkl'
        with (root/'B'/f'learner_update_{completed:04d}.pkl').open('rb') as f:saved=pickle.load(f)
        if saved['completed_supervised_updates']!=completed or pytree_sha256(saved['normalizer'])!=pytree_sha256(payload.observation_normalizer) or pytree_sha256(saved['critic'])!=pytree_sha256(payload.critic_params):raise ValueError('saved BC full-state invariants')
        if sha(read(learner)['path'])!=read(learner)['sha256']:raise ValueError('saved BC state SHA')
        write(root/'B_completed.json',dict(phase='completed',completed_updates=completed,termination=status['status'],selected_update=selected,selection_scores=scores,eligible=eligible,
            learner_last=dict(path=str(learner),sha256=sha(learner)),candidate=dict(path=str(chosen),sha256=sha(chosen),update=selected),published_policy=None,
            PPO_launched=False,source='pi0',normalizer_frozen=True,critic_frozen=True,BC_optimizer_preserved=True,BC_RNG_and_supervised_clock_preserved=True))
    except BaseException:
        write(root/'costs.json',r.costs);raise
    finally:writer.close()


def report(path):
    from .retention_b_report import build_report
    return build_report(path)


def fixed_train_probe(net,params,reference,manifest,keep,keep_receipt):
    from .generative_bridge.student import load_optional_demo
    import jax.numpy as jp
    obs,targets,_=load_optional_demo(manifest)
    predict=lambda x:np.asarray(net.parametric_action_distribution.mode(net.policy_network.apply(params[0],params[1],{'state':jp.asarray(x)})))
    prediction=predict(obs)
    roots=np.asarray(manifest['sample_roots']);origins=np.asarray(manifest['sample_origins'])
    result={}
    for ancestor in np.unique(roots):
        result[str(ancestor)]={}
        for origin in np.unique(origins):
            ids=(roots==ancestor)&(origins==origin)
            result[str(ancestor)][str(origin)]=dict(n=int(ids.sum()),mse_by_action=np.mean((prediction[ids]-targets[ids])**2,axis=0).tolist())
    ko,kp=keep
    reference_action=np.asarray(net.parametric_action_distribution.mode(net.policy_network.apply(reference[0],reference[1],{'state':jp.asarray(ko)})))
    groups=np.concatenate([np.repeat(x['group'],x['steps']) for x in keep_receipt['episodes']])
    keep_error=(predict(ko)-reference_action)**2
    keep_rows={str(g):dict(n=int((groups==g).sum()),mse_by_action=keep_error[groups==g].mean(axis=0).tolist()) for g in np.unique(groups)}
    out=dict(description='fixed TRAIN probes; separate from actual optimizer batch contributions',demo=result,keep=keep_rows)
    return out


COVERAGE_SCHEMA='jit_retention_B_keep_coverage_v1'


def coverage_budget():
    """Original17 snapshot layout, no newly paid R5/four-combination diagnostics."""
    b=dict(keep_collection=80*400,DEV_baselines=0,snapshot_nodes=5*15*17*400,
        B_full_DEV=4*64*400,B_warning_repeat=4*2*64*400,selected_train_repeat=2*8*17*400,
        selected_train_composites=0,R5_focus_repeat=0,reset_GPU=0)
    b.update(total=sum(b.values()),prior_charged=388036,hard_cap=2000000,wall_seconds=43200,
        B_updates=2000,PPO_updates=0,D2_cumulative_maximum=1346036)
    return b


def balanced_keep_records(records,forbidden):
    """Half per group, equal ancestor mass, then equal trajectory mass and frames."""
    accepted=[];excluded=[]
    for r in records:
        if r.get('role')!='TRAIN' or r.get('group') not in ('nominal','random'):
            raise ValueError('successful keep requires TRAIN nominal/random roles')
        ancestry=set(r.get('ancestry',[]))|{r['ancestor']}
        if not r.get('ancestry') or ancestry&set(forbidden):raise ValueError('keep shared protected ancestor')
        if r['success'] is not True:
            excluded.append({k:r[k] for k in ('ancestor','group','reason')});continue
        a=np.asarray(r['observations'],np.float32)
        if a.ndim!=2 or a.shape[1]!=76 or not len(a) or not np.isfinite(a).all():
            raise ValueError('keep observations must be finite nonempty76D')
        accepted.append((r,a))
    grouped={g:{} for g in ('nominal','random')}
    for r,a in accepted:grouped[r['group']].setdefault(r['ancestor'],[]).append((r,a))
    if any(not group for group in grouped.values()):raise ValueError('empty successful keep group')
    obs=[];weights=[];episodes=[]
    for g,ancestors in grouped.items():
        for ancestor,trajectories in ancestors.items():
            for r,a in trajectories:
                obs.extend(a);weights.extend(np.full(len(a),.5/len(ancestors)/len(trajectories)/len(a)))
                episodes.append({**{k:v for k,v in r.items() if k!='observations'},'steps':len(a)})
    return np.asarray(obs,np.float32),np.asarray(weights,np.float64),episodes,dict(
        excluded=excluded,unique_ancestors=sum(map(len,grouped.values())),
        group_ancestors={g:len(v) for g,v in grouped.items()},
        observations={g:sum(e['steps'] for e in episodes if e['group']==g) for g in grouped},
        post_contact_observations='UNKNOWN')


def worker_keep_inputs(plan):
    """Shared real worker route: coverage cannot fall back to the old anchors."""
    if plan.get('schema')!=COVERAGE_SCHEMA:return plan['keep_receipt']
    root=Path(plan['output']);lock=root/'merged_keep_lock.json'
    if not lock.exists():raise ValueError('merged keep missing; collection must complete first')
    bound=read(lock)
    for key,digest in (('receipt','sha256'),('data','data_sha256')):
        if sha(bound[key])!=bound[digest]:raise ValueError('merged keep drift')
    if 'new_outcomes' in bound and sha(bound['new_outcomes'])!=bound['outcomes_sha256']:raise ValueError('merged keep outcomes drift')
    receipt=read(bound['receipt'])
    if receipt.get('schema')!='jit_merged_keep_coverage_v1' or receipt.get('role')!='train' or receipt.get('source_B')!=plan['source_B']:
        raise ValueError('merged keep source/role mismatch')
    if receipt['path']!=bound['data'] or receipt['sha256']!=bound['data_sha256'] or receipt.get('full_success') is not True:
        raise ValueError('merged keep binding mismatch')
    return bound['receipt']


def require_coverage_execution(path,authorization=None,*,now=None):
    """No side effects, checked before CUDA imports, services, or collection."""
    p=read(path)
    if p.get('schema')!=COVERAGE_SCHEMA:return None
    if authorization is None:raise ValueError('explicit new-stage execution authorization required')
    a=read(authorization)
    if (a.get('schema')!='jit_keep_coverage_execution_authorization_v1' or
        a.get('scope')!='collect80_and_BC2000_keep_coverage_only' or
        a.get('plan_sha256')!=sha(path) or a.get('authorized_updates')!=2000 or
        a.get('charged_upper_bound')!=958000 or a.get('original_D2_started_unix')!=p['original_D2_started_unix'] or
        a.get('source')!='explicit_user_instruction' or not a.get('instruction_reference')):
        raise ValueError('execution authorization identity/scope mismatch')
    if (time.time() if now is None else now)-p['original_D2_started_unix']>=43200:
        raise TimeoutError('original D2 twelve-hour clock expired; do not reset')
    return a


def claim_coverage_stage(root,kind,path):
    """Exclusive permanent claim; failure does not authorize an automatic retry."""
    import json
    with (Path(root)/(kind+'_claim.json')).open('x') as stream:
        json.dump(dict(pid=os.getpid(),plan=str(path),plan_sha256=sha(path),created_unix=time.time()),stream)


def _history_locks(locks,parent,code):
    import hashlib
    result={};historical={}
    for name,h in locks.items():
        f=Path(name)
        if f.is_relative_to(Path(parent['code'])) and f.suffix=='.py':
            relative=str(f.relative_to(parent['code']))
            blob=subprocess.check_output(['git','show',parent['code_revision']+':'+relative],cwd=code)
            if hashlib.sha256(blob).hexdigest()!=h:raise ValueError('historical B code drift '+name)
            historical[name]=dict(revision=parent['code_revision'],sha256=h)
        else:
            if sha(f)!=h:raise ValueError('source input drift '+name)
            result[name]=h
    return result,historical


def _keep_records(receipt):
    if receipt.get('role')!='train' or receipt.get('data_role')!='TRAIN' or receipt.get('full_success') is not True or receipt.get('DEV_used') is not False:
        raise ValueError('old keep TRAIN provenance required')
    if sha(receipt['path'])!=receipt['sha256']:raise ValueError('old keep hash drift')
    with np.load(receipt['path'],allow_pickle=False) as z:obs=np.asarray(z['actor_observation_before'])
    result=[];offset=0
    for e in receipt['episodes']:
        n=e['steps'];result.append({**e,'role':'TRAIN','ancestry':e.get('ancestry',[e['ancestor']]),
            'source_kind':'old_keep','observations':obs[offset:offset+n]});offset+=n
    if offset!=len(obs):raise ValueError('keep episode windows incomplete')
    tape=Path(receipt['path']).parent/'keep_collection/prefixes.npz'
    if tape.exists():
        with np.load(tape,allow_pickle=False) as z:
            for r in result:
                lane=r['lane'];mask=z['prefix_mask'][:,lane];contact=np.flatnonzero(z['first_valid_contact'][mask,lane])
                if int(mask.sum())!=r['steps']:raise ValueError('old keep raw window drift')
                r['post_contact_observations']=0 if not len(contact) else r['steps']-int(contact[0])
                r['first_valid_contact']=int(contact[0]) if len(contact) else None
    return result


def prepare_keep_coverage(proposal,parent,output,repository):
    """Static file preparation; no environment import or physics/optimizer call."""
    import shutil
    q=read(proposal);old=read(parent);oldroot=Path(old['output']);code=Path(repository).resolve()
    if q.get('schema')!='jit_keep_coverage_proposal_v1' or old.get('schema')!='jit_retention_B_v1':raise ValueError('coverage proposal/parent schema')
    if read(oldroot/'status.json')['phase']!='completed' or read(oldroot/'B_completed.json')['completed_updates']!=2000:
        raise ValueError('completed B2000 parent required; not a zero-retry/resume')
    prior=old['budget']['prior_charged']+sum(x['charged'] for x in read(oldroot/'costs.json'))
    if prior!=388036:raise ValueError('actual prior D2 cost differs from declaration')
    if old['ppo_updates'] or old['E_G_updates'] or old['automatic_next_stage']:raise ValueError('parent B-only contract')
    c=q['keep_collection'];student=q['student']
    if any(c.get(k)!=v for k,v in dict(seed=1010269101,role='TRAIN',capacity=80,nominal=16,random=64,horizon=400,requested_action_residual=0,source='fixed pi0').items()):raise ValueError('keep collection contract')
    if c['actor_normalizer_pair']!=old['models']['pi0']['policy']:raise ValueError('keep pi0 pairing changed')
    if any(student.get(k)!=v for k,v in dict(learning_rate=1e-5,demo=1,keep=1,batch=256,clip=1,updates=2000,checkpoints=[0,100,500,1000,2000],normalizer='frozen',critic='frozen').items()):raise ValueError('fixed BC contract changed')
    locks,history=_history_locks({**old['locks'],**q['locks']},old,code)
    # Models are inherited as a unit. Pin actual serialized source files without loading JAX/CUDA.
    for m in old['models'].values():
        pol=m['policy'];ck=Path(pol['checkpoint'])
        if sha(ck/'payload.pkl')!=pol['payload_sha256']:raise ValueError('source payload drift')
        for f in (ck/'identity.json',ck/'payload.pkl',Path(m['bank']),Path(m['frozen_policy']),Path(pol['formal_config'])):locks[str(f)]=sha(f)
        formal=read(pol['formal_config'])
        for phase in ('up','down'):
            ref=formal['inputs'][phase+'_config_path'];h=formal['inputs'][phase+'_config_sha256']
            if sha(ref)!=h:raise ValueError('frozen physical/task configuration drift')
            locks[ref]=h
        physical=read(formal['inputs']['up_config_path'])['model']
        for key in ('xml','reference'):
            asset=code/physical[key+'_path'];h=physical[key+'_sha256']
            if sha(asset)!=h:raise ValueError('frozen physics/reference asset drift')
            locks[str(asset)]=h
    demo=read(old['demo_manifest']);train=read(old['train_roots']);solver=read(old['solver_roots'])
    if demo['count']!=525 or len(set(demo['sample_roots']))!=8 or len(train)!=8 or len(solver)!=7:raise ValueError('locked G8/525 and all7 solver roots required')
    if any(x['root']['role']!='TRAIN' or x['root']['data_role'].upper()!='TRAIN' for x in train):raise ValueError('TRAIN roots role drift')
    if any(x['root']['role']!='SOLVER_DEV' for x in solver):raise ValueError('solver role drift')
    forbidden={v for x in solver for v in (x['root']['root_id'],x['root']['root_episode_id'])}
    forbidden.update('DEV:'+c['case'] for c in old['batches'][0]['cases'])
    ids=c['ancestor_ids'];expected=[f'TRAIN_KEEP-{c["seed"]}-{i}' for i in range(80)]
    if ids!=expected or set(ids)&forbidden:raise ValueError('fresh TRAIN ancestor namespace required')
    oldreceipt=read(old['keep_receipt']);balanced_keep_records(_keep_records(oldreceipt),forbidden)
    zero=read(oldroot/'B/learner_update_0000.json')
    for name,key in (('actor_sha256','actor_sha256'),('normalizer_sha256','normalizer_sha256'),('critic_sha256','critic_sha256')):
        if zero[name]!=old['models']['pi0']['policy'][key]:raise ValueError('parent original pi0 initialization mismatch')
    if set(demo['sample_roots'])!={x['root']['root_episode_id'] for x in train}:raise ValueError('demo ancestor roles differ')
    prep=Path(q['output']);spec=read(c['spec']);initial=Path(spec['initial_state_bank']);request=Path(spec['frozen_request_table']['path'])
    with np.load(initial,allow_pickle=False) as z:
        if z['qpos'].shape!=(80,12) or z['qvel'].shape!=(80,11) or not all(np.isfinite(z[k]).all() for k in ('qpos','qvel')):raise ValueError('keep input dimensions/finite')
    with np.load(request,allow_pickle=False) as z:
        if z['requested'].shape!=(400,80,4) or np.any(z['requested']) or z['onsets'].shape!=(80,) or np.any(z['onsets']):raise ValueError('no action disturbance permitted')
    _validate_keep_spec(spec,old['models']['pi0'],c['seed'])
    # mkdir is the concurrency guard; never overwrite a partially prepared stage.
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    for src,name in ((initial,'keep_initial.npz'),(request,'keep_requests.npz')):shutil.copyfile(src,root/name)
    spec.update(initial_state_bank=str(root/'keep_initial.npz'),frozen_request_table=dict(path=str(root/'keep_requests.npz'),sha256=sha(root/'keep_requests.npz')))
    write(root/'keep_spec.json',spec)
    episodes=[dict(ancestor=a,ancestry=[a],role='TRAIN',data_role='TRAIN',lane=i,group='nominal' if i<16 else 'random',condition_id='nominal_same_condition' if i<16 else a,source_initial_bank=str(root/'keep_initial.npz'),source_seed=c['seed']) for i,a in enumerate(ids)]
    write(root/'TRAIN_keep_manifest.json',dict(schema='jit_keep_episode_roles_v1',role='TRAIN',episodes=episodes,forbidden_ancestors=sorted(forbidden),TEST_namespace='unopened; new TRAIN namespace only',DEV_state_used=False))
    for f in (Path(proposal),Path(parent),oldroot/'status.json',oldroot/'B_completed.json',oldroot/'costs.json',oldroot/'B/learner_update_0000.json',Path(old['keep_receipt']),Path(demo['path']),Path(oldreceipt['path']).parent/'keep_collection/prefixes.npz'):locks[str(f)]=sha(f)
    for f in (root/'keep_initial.npz',root/'keep_requests.npz',root/'keep_spec.json',root/'TRAIN_keep_manifest.json'):locks[str(f)]=sha(f)
    # Lock all execution Python modules, not just the launcher. No compile/import.
    code_locks={str(f):sha(f) for directory in ('JIT/src/jit_dvgc','JIT/cli') for f in (code/directory).rglob('*.py')}
    locks.update(code_locks)
    layout=coverage_budget();start=old['original_D2_started_unix']
    p={**old,'schema':COVERAGE_SCHEMA,'stage':'D2_B_keep_coverage','output':str(root),'code':str(code),'code_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip(),
        'source_B':str(Path(parent).resolve()),'proposal':str(Path(proposal).resolve()),'only_changed_factor':'keep_train_coverage','locks':locks,'historical_code_locks':history,
        'budget':layout,'old_keep_receipt':old['keep_receipt'],'keep_receipt':str(root/'keep_receipt.json'),'keep_collection':dict(spec=str(root/'keep_spec.json'),manifest=str(root/'TRAIN_keep_manifest.json'),capacity=80,horizon=400),
        'reused_zero':dict(DEV=old['reused_zero']['DEV'],reason='baseline DEV only; all15 node0 snapshots physically evaluated anew'),
        'execution_permission':dict(status='not_authorized',requires_explicit_new_stage=True),'original_D2_started_unix':start,'tensorboard_port':6028,
        'learner_last':None,'stage_candidate':None,'best_dev_candidate':old['models']['R5'],'published_policy':None,'diagnostic_train_roots':[],
        'initializer':'original_pi0_fresh_Adam_RNG','previous_B_updates':2000,'BC_sampling':'unchanged fold_in(base_rng,absolute_update); split demo/keep once; same seed as source B'}
    write(root/'code_lock.json',dict(revision=p['code_revision'],files=code_locks));write(root/'input_lock.json',locks);write(root/'source_lock.json',dict(identities=read(oldroot/'source_lock.json')['identities'],locks=locks));write(root/'budget_dry_run.json',layout);write(root/'plan.json',p)
    remaining=max(0,43200-(time.time()-start))
    write(root/'status.json',dict(phase='prepared',stage=p['stage'],charged_interactions=0,D2_prior_charged=prior,completed_supervised_updates=0,planned_supervised_updates=2000,execution_authorized=False,original_D2_started_unix=start,remaining_wall_seconds=remaining,execution_blockers=['missing_new_stage_authorization']+(['original_clock_expired'] if remaining<=0 else [])))
    audit(root/'plan.json');return root/'plan.json'


def audit_keep_coverage(p):
    parent=read(p['source_B']);proposal=read(p['proposal'])
    if p['only_changed_factor']!='keep_train_coverage' or p['models']!=parent['models'] or p['BC']!=parent['BC'] or p['BC_selection']!=parent['BC_selection'] or p['seed']!=parent['seed']:
        raise ValueError('single-change BC/source/sampling contract drift')
    if p['demo_manifest']!=parent['demo_manifest'] or p['train_roots']!=parent['train_roots'] or p['solver_roots']!=parent['solver_roots'] or p['batches']!=parent['batches']:
        raise ValueError('demo/roots/DEV identity changed')
    if p['budget']!=coverage_budget() or p['original_D2_started_unix']!=parent['original_D2_started_unix']:
        raise ValueError('coverage budget/original clock drift')
    if p['initializer']!='original_pi0_fresh_Adam_RNG' or p['training_updates']!=2000 or p['execution_permission']!={'status':'not_authorized','requires_explicit_new_stage':True}:
        raise ValueError('fresh pi0/explicit-stage authorization contract')
    root=Path(p['output']);manifest=read(p['keep_collection']['manifest']);ids=[e['ancestor'] for e in manifest['episodes']]
    expected=[f'TRAIN_KEEP-{proposal["keep_collection"]["seed"]}-{i}' for i in range(80)]
    if manifest.get('role')!='TRAIN' or manifest.get('DEV_state_used') is not False or ids!=expected:raise ValueError('keep ancestor/source namespace changed')
    forbidden=set(manifest['forbidden_ancestors'])
    for e in manifest['episodes']:
        if e['role']!='TRAIN' or e['data_role']!='TRAIN' or e['ancestry']!=[e['ancestor']] or set(e['ancestry'])&forbidden:raise ValueError('keep role/ancestor leakage')
    if p['keep_receipt']!=str(root/'keep_receipt.json') or p['old_keep_receipt']!=parent['keep_receipt']:raise ValueError('keep worker input route drift')
    spec=read(p['keep_collection']['spec'])
    if spec['num_envs']!=80 or spec['horizon']!=400 or spec['role']!='TRAIN' or spec['seed']!=proposal['keep_collection']['seed'] or spec['bank']!=p['models']['pi0']['bank'] or spec['proposer']!=p['models']['pi0']['proposer']:raise ValueError('keep spec source/layout drift')
    _validate_keep_spec(spec,p['models']['pi0'],proposal['keep_collection']['seed'])
    with np.load(spec['initial_state_bank'],allow_pickle=False) as z:
        if z['qpos'].shape!=(80,12) or z['qvel'].shape!=(80,11) or not np.isfinite(z['qpos']).all() or not np.isfinite(z['qvel']).all():raise ValueError('keep initial layout')
    with np.load(spec['frozen_request_table']['path'],allow_pickle=False) as z:
        if z['requested'].shape!=(400,80,4) or np.any(z['requested']) or np.any(z['onsets']):raise ValueError('keep request must be zero')
    remaining=max(0,43200-(time.time()-p['original_D2_started_unix']))
    return dict(remaining_wall_seconds=remaining,execution_authorized=False,execution_blockers=['missing_new_stage_authorization']+(['original_clock_expired'] if remaining<=0 else []),worker_keep_input=p['keep_receipt'],keep_data_status='pending_collection',TRAIN_ancestors=80,nominal_unique_conditions=1,random_initials=64)


def _validate_keep_spec(spec,model,seed):
    expected=dict(bank=model['bank'],proposer=model['proposer'],role='TRAIN',seed=seed,num_envs=80,horizon=400,controller_mode='fixed_random',full_episode_rollout=True,frozen_explorer_evaluation=True,reward_mode='original_all_phases',success_criterion='stable_forward_recovery')
    if any(spec.get(k)!=v for k,v in expected.items()):raise ValueError('keep collection source/layout/criterion changed')
    if any(k in spec for k in ('warmup_initializer','closed_loop_prefix_initializer','bridge_action_plan','phase_policy','explorer_resume','explorer_checkpoint')):
        raise ValueError('keep collection cannot override frozen pi0 or use E/G')


def collect_coverage_keep(run):
    """Future authorized runtime only: one fixed-pi0 collection, no refill."""
    p=run.p;root=run.root;k=p['keep_collection'];name='keep_collection'
    out=run.child(name,[PY,str(Path(p['code'])/'JIT/cli/run_pulse_exploration.py'),'--mode','collect','--spec',k['spec'],'--output',str(root/name)],32000)
    from .retention_repair_execution import verify
    verify(dict(name=name,spec=k['spec'],capacity=80,horizon=400,scored=80),root)
    manifest=read(k['manifest']);new=[]
    with np.load(out/'prefixes.npz',allow_pickle=False) as z:
        if z['prefix_mask'].shape!=(400,80):raise ValueError('keep trace capacity mismatch')
        for e in manifest['episodes']:
            lane=e['lane'];m=z['prefix_mask'][:,lane].astype(bool);n=int(m.sum())
            if not n or not m[:n].all() or m[n:].any():raise ValueError('keep trace incomplete/noncontiguous')
            for key in ('actor_observation_before','qpos','qvel','normalized_action_executed'):
                if not np.isfinite(z[key][m,lane]).all():raise FloatingPointError('nonfinite keep physics/observation')
            if np.any(z['requested_delta'][m,lane]) or np.any(z['effective_delta'][m,lane]):raise ValueError('keep disturbance was applied')
            success=bool(np.any(z['success'][m,lane]));failure=bool(np.any(z['physical_failure'][m,lane]));accepted=success and not failure
            contact=np.flatnonzero(z['first_valid_contact'][m,lane]);post=0 if not len(contact) else n-int(contact[0])
            new.append({**e,'success':accepted,'reason':'success' if accepted else 'ambiguous_success_failure' if success and failure else 'physical_failure' if failure else 'no_success_within_horizon',
                'source_kind':'new_keep','observations':z['actor_observation_before'][m,lane].copy(),'post_contact_observations':post,'first_valid_contact':int(contact[0]) if len(contact) else None})
    write(root/'new_keep_outcomes.json',[{key:v for key,v in r.items() if key!='observations'} for r in new])
    if not any(r['success'] for r in new):raise ValueError('no new successful keep coverage; no refill or BC')
    old=_keep_records(read(p['old_keep_receipt']));obs,weights,episodes,summary=balanced_keep_records(old+new,set(manifest['forbidden_ancestors']))
    summary.update(new_declared=80,new_success=sum(r['success'] for r in new),new_failed_or_ambiguous=sum(not r['success'] for r in new),new_success_per_group={g:sum(r['success'] and r['group']==g for r in new) for g in ('nominal','random')},new_post_contact_observations=sum(r['post_contact_observations'] for r in new if r['success']),old_post_contact_observations=sum(r['post_contact_observations'] for r in old) if all('post_contact_observations' in r for r in old) else 'UNKNOWN',nominal_unique_conditions=1)
    summary['stage_observations']={kind:{g:dict(total=sum(e['steps'] for e in episodes if e['source_kind']==kind and e['group']==g),at_or_after_contact=sum(e['post_contact_observations'] for e in episodes if e['source_kind']==kind and e['group']==g) if all('post_contact_observations' in e for e in episodes if e['source_kind']==kind and e['group']==g) else 'UNKNOWN') for g in ('nominal','random')} for kind in ('old_keep','new_keep')}
    summary['post_contact_observations']=sum(e['post_contact_observations'] for e in episodes) if all('post_contact_observations' in e for e in episodes) else 'UNKNOWN'
    dest=root/'keep_anchors.npz';np.savez_compressed(dest,actor_observation_before=obs,weights=weights)
    write(root/'keep_receipt.json',dict(schema='jit_merged_keep_coverage_v1',role='train',data_role='TRAIN',source_B=p['source_B'],old_receipt=p['old_keep_receipt'],new_outcomes=str(root/'new_keep_outcomes.json'),path=str(dest),sha256=sha(dest),episodes=episodes,groups={g:[i for i,e in enumerate(episodes) if e['group']==g] for g in ('nominal','random')},summary=summary,full_success=True,DEV_used=False,charged_interactions=read(out/'status.json')['charged_interactions']))
    write(root/'merged_keep_lock.json',dict(receipt=str(root/'keep_receipt.json'),sha256=sha(root/'keep_receipt.json'),data=str(dest),data_sha256=sha(dest),new_outcomes=str(root/'new_keep_outcomes.json'),outcomes_sha256=sha(root/'new_keep_outcomes.json')))
