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


def prepare(output,repository):
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
    plan=dict(schema='jit_retention_B_v1',stage='D2_B',output=str(root),code=str(code),code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip(),locks=locks,
        source_plan=str(BASE/'plan.json'),source_supplement=str(BASE/'report_004/formal_preparation.json'),models=original['models'],indices=original['indices'],budget=layout,
        original_D2_started_unix=start,keep_receipt=str(BASE/'keep_receipt.json'),demo_manifest=original['demo_manifest'],reset_inventory=supp['reset_pool_inventory'],batches=batches,
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
        write(self.root/'status.json',dict(phase=phase,stage=stage,pid=os.getpid(),updated_unix=time.time(),charged_interactions=sum(c['charged'] for c in self.costs),D2_prior_charged=12900,planned_supervised_updates=2000,PPO_updates=0,E_G_updates=0,completed_supervised_updates=getattr(self,'current_update',0),**extra))
    def child(self,name,command,maximum):
        if time.time()-self.p['original_D2_started_unix']>=43200:raise TimeoutError('D2 original12h budget')
        if 12900+sum(c['charged'] for c in self.costs)+maximum>2000000:raise ValueError('D2 physics reserve before dispatch')
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
    if p['stage']!='D2_B' or p['automatic_next_stage'] or p['ppo_updates'] or p['E_G_updates']:raise ValueError('B-only authorization contract')
    for f,h in p['locks'].items():
        if sha(f)!=h:raise ValueError('input drift '+f)
    if len(read(p['train_roots']))!=8 or len(read(p['solver_roots']))!=7:raise ValueError('root count')
    return dict(phase='passed',budget=p['budget'],source='pi0',normalizer_critic_frozen=True,automatic_PPO=False)


def reward_http(root,step):
    import urllib.request,json
    last=None
    for _ in range(20):
        try:
            url='http://localhost:6026/data/plugin/scalars/scalars?run=.&tag=DEV%2FB_reward_per_transition'
            values=json.loads(urllib.request.urlopen(url,timeout=3).read())
            if any(int(v[1])==step for v in values):
                write(Path(root)/f'TensorBoard_reward_http_{step:04d}.json',dict(url=url,values=values,verified_step=step));return
        except Exception as e:last=repr(e)
        time.sleep(1)
    raise RuntimeError('current B DEV reward scalar not visible '+str(last))


def run(path):
    audit(path)
    p=read(path);root=Path(p['output'])
    for f,h in p['locks'].items():
        if sha(f)!=h:raise ValueError('input drift '+f)
    if read(root/'status.json')['phase']!='prepared':raise ValueError('no implicit rerun')
    r=Run(path);write(root/'ACTIVE_RUN.json',dict(execution=str(root/'status.json'),lineage=str(root/'status.json'),name='retention D2 finite BC+keep'))
    with (root/'watcher.log').open('x') as f:w=subprocess.Popen([PY,str(Path(p['code'])/'JIT/cli/watch_run_errors.py'),'--active-run',str(root/'ACTIVE_RUN.json'),'--state-dir',str(root/'notifications')],cwd=p['code'],env=r.env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    with (root/'tensorboard.log').open('x') as f:
        tb=subprocess.Popen([PY,'-m','tensorboard.main','--logdir',str(root/'tensorboard'),'--port','6026','--host','0.0.0.0','--reload_interval','1'],env=r.env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    write(root/'launch.json',dict(pid=os.getpid(),watcher_pid=w.pid,tensorboard_pid=tb.pid,tensorboard_url='http://localhost:6026',started_unix=time.time(),plan=str(path)))
    try:
        for b in p['batches']:
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
        r.status('completed','D2_B',result='B complete or confirmed retention stop; no automatic PPO',learner_last=state['learner_last'],B_selected_candidate=state['candidate'],best_dev_candidate=p['models']['R5'],published_policy=None)
    except BaseException as e:
        if (root/'costs.json').exists():r.costs=read(root/'costs.json')
        r.status('failed','D2_B',error=repr(e));raise


def reset_worker(path):
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
    keep=load_retention_traces(read(p['keep_receipt']));demo=read(p['demo_manifest']);baseline=full_results(p,root/'DEV_pi0')
    writer.add_scalar('DEV/pi0_reward_per_transition',baseline['reward_per_transition'],0);writer.add_scalar('DEV/R5_reward_per_transition',full_results(p,root/'DEV_R5')['reward_per_transition'],0);writer.flush()
    def metrics(row):
        r.current_update=row['update']
        for k,v in row.items():writer.add_scalar('BC/'+k,v,row['update'])
        if row['update']%10==0:writer.flush();r.status('running','BC+keep')
    def probe(step,params):
        from .student import load_optional_demo
        import jax.numpy as jp
        obs,targets,_=load_optional_demo(demo)
        predict=lambda x:np.asarray(net.parametric_action_distribution.mode(net.policy_network.apply(params[0],params[1],{'state':jp.asarray(x)})))
        prediction=predict(obs)
        roots=np.asarray(demo['sample_roots']);origins=np.asarray(demo['sample_origins'])
        result={}
        for ancestor in np.unique(roots):
            result[str(ancestor)]={}
            for origin in np.unique(origins):
                ids=(roots==ancestor)&(origins==origin)
                result[str(ancestor)][str(origin)]=dict(n=int(ids.sum()),mse_by_action=np.mean((prediction[ids]-targets[ids])**2,axis=0).tolist())
        ko,kp=keep
        reference_action=np.asarray(net.parametric_action_distribution.mode(net.policy_network.apply(payload.observation_normalizer,payload.actor_params,{'state':jp.asarray(ko)})))
        groups=np.concatenate([np.repeat(x['group'],x['steps']) for x in read(p['keep_receipt'])['episodes']])
        keep_error=(predict(ko)-reference_action)**2
        keep_rows={str(g):dict(n=int((groups==g).sum()),mse_by_action=keep_error[groups==g].mean(axis=0).tolist()) for g in np.unique(groups)}
        out=dict(description='fixed TRAIN probes; separate from actual optimizer batch contributions',demo=result,keep=keep_rows)
        write(root/f'B{step:04d}_TRAIN_probes.json',out);return out
    def checkpoint(step,params):
        r.current_update=step
        if pytree_sha256(params[0])!=p['models']['pi0']['policy']['normalizer_sha256'] or pytree_sha256(params[2])!=pytree_sha256(payload.critic_params):raise ValueError('BC frozen statistics or critic changed')
        if step==0 and pytree_sha256(params[1])!=p['models']['pi0']['policy']['actor_sha256']:raise ValueError('BC did not start from original pi0')
        if step==0:
            full=baseline;solver=solver_combinations(r,0,zero_only=True);train=solver_combinations(r,0,zero_only=True,role='TRAIN');warnings=[];confirmed=[]
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
        labels=full['labels'];old=baseline['labels'];cells={g:dict(n=len(labels[g]),success=sum(labels[g]),N01=sum(not a and b for a,b in zip(old[g],labels[g])),N10=sum(a and not b for a,b in zip(old[g],labels[g]))) for g in 'ABCD'}
        score=sum(x['combinations']['independent']['label']==1 for x in solver);scores[step]=score;eligible[step]=not bool(confirmed)
        rows[step]=dict(update=step,cells=cells,solver_success=score,solver_n=len(solver),warning_cells=warnings,confirmed_regression=confirmed,eligible=eligible[step],full=full,solver=solver,train=train,train_success=sum(x['combinations']['independent']['label']==1 for x in train),
            actor_sha256=pytree_sha256(params[1]),normalizer_sha256=pytree_sha256(params[0]),critic_sha256=pytree_sha256(params[2]))
        write(root/'B_checkpoint_results.json',rows)
        writer.add_scalar('DEV/B_reward_per_transition',full['reward_per_transition'],step)
        writer.add_scalar('SOLVER_DEV/independent_success',score,step)
        for g,c in cells.items():
            for k in ('success','N01','N10'):writer.add_scalar('DEV/'+g+'/'+k,c[k],step)
        writer.flush();reward_http(root,step);r.status('running','B_checkpoint_checked')
        return bool(confirmed)
    try:
        warmup_actor(net,(payload.observation_normalizer,payload.actor_params,payload.critic_params),(payload.observation_normalizer,payload.actor_params),demo,keep,root/'B',seed=p['seed'],updates=2000,batch_size=256,full_learner=True,actual_gradient_audit=True,metrics_callback=metrics,checkpoint_callback=checkpoint,probe_callback=probe)
        status=read(root/'B/warmup_status.json');selected=select_B(scores,eligible);completed=status['completed_updates']
        diagnostics=solver_combinations(r,selected,role='TRAIN',only_roots=p['diagnostic_train_roots'])
        confirmations=[solver_combinations(r,selected,zero_only=True,role='TRAIN',repeat=f'_confirm{i}') for i in range(2)]
        focus=[x['root']['root_id'] for x in read(p['train_roots']) if x['root']['R5_label']!=1]
        R5_confirmations=[solver_combinations(r,selected,zero_only=True,role='TRAIN',repeat=f'_confirm{i}',model='R5',only_roots=focus) for i in range(2)]
        write(root/'selected_diagnostics.json',dict(selected_update=selected,combinations=diagnostics,TRAIN_confirmations=confirmations,R5_focus_confirmations=R5_confirmations))
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
