"""One explicitly selected finite stage; no student/E/G optimizer calls."""
import os,subprocess,time
from pathlib import Path
import numpy as np
from .retention_repair import read,write,sha
from .retention_next import audit,baseline_class,choose_roots,qualify_method
from .retention_repair_execution import PY


class Stage:
    def __init__(self,path):
        audit(path);self.plan_path=Path(path);self.p=read(path);self.root=Path(self.p['output']);self.code=Path(self.p['code'])
        if read(self.root/'status.json')['phase']!='prepared':raise ValueError('no implicit resume or retry')
        self.start=time.monotonic();self.costs=[];self.active='start'
        self.env=dict(os.environ,PYTHONPATH=str(self.code/'JIT/src'),JAX_PLATFORMS='cuda,cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false',
            JIT_AUTO_PUBLISH='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',JAX_COMPILATION_CACHE_DIR=str(self.root/'jax_cache'))

    def status(self,phase,**extra):
        write(self.root/'status.json',dict(phase=phase,stage=self.p['stage'],active=self.active,pid=os.getpid(),updated_unix=time.time(),
            charged_interactions=sum(c['charged_interactions'] for c in self.costs),
            maximum_planned_interactions=self.p['budget']['total'],physical_hard_cap=self.p['budget']['hard_cap'],
            training_updates=0,student_updates=0,E_updates=0,G_updates=0,wall_seconds=time.monotonic()-self.start,**extra))

    def reserve(self,name,maximum):
        if time.monotonic()-self.start>=self.p['budget']['wall_seconds']:raise TimeoutError('stage wall budget exhausted')
        if sum(c['charged_interactions'] for c in self.costs)+maximum>self.p['budget']['hard_cap']:raise ValueError('physics budget before dispatch')
        c=dict(stage=name,charged_interactions=maximum,reservation=maximum,phase='running',optimizer_updates=0)
        self.costs.append(c);self.active=name;write(self.root/'costs.json',self.costs);self.status('running');return c

    def collect(self,b):
        c=self.reserve(b['name'],b['capacity']*b['horizon']);out=self.root/b['name']
        with (self.root/(b['name']+'.log')).open('x') as log:
            child=subprocess.Popen([PY,str(self.code/'JIT/cli/run_pulse_exploration.py'),'--mode','collect','--spec',b['spec'],'--output',str(out)],cwd=self.code,env=self.env,stdout=log,stderr=subprocess.STDOUT)
            self.status('running',child_pid=child.pid)
            try:rc=child.wait(timeout=max(.1,self.p['budget']['wall_seconds']-(time.monotonic()-self.start)))
            except subprocess.TimeoutExpired:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                raise TimeoutError('collection wall budget exhausted')
        if rc:raise RuntimeError(f'{b["name"]} exit{rc}; failed reservation retained')
        receipt=read(out/'status.json')
        if receipt['phase']!='completed' or receipt['charged_interactions']>c['reservation']:raise ValueError('collection receipt drift')
        c.update(phase='completed',**{k:receipt[k] for k in ('charged_interactions','active_interactions','padding_interactions')})
        write(self.root/'costs.json',self.costs);self.status('running');return out

    def evaluate(self,name,rows,model='pi0',actions=None,session=None):
        from .pulse_exploration_runtime import evaluate
        if not rows:return []
        c=self.reserve(name,len(rows)*400);m=self.p['models'][model];rowfile=self.root/(name+'_rows.json');write(rowfile,rows)
        spec=dict(bank=m['bank'],proposer=m['proposer'],order=[m['proposer']],horizon=400,budget=len(rows)*400,
            candidates=str(rowfile),full_matrix=True,record_actor_preobservations=True,record_retention_diagnostics=True,
            reward_mode='original_all_phases',success_criterion='stable_forward_recovery',preserve_snapshot_episode_context=True,
            suffix_rng_count=len(rows),suffix_rng_indices=list(range(len(rows))))
        if actions is not None:
            ap=self.root/(name+'_actions.npz')
            np.savez_compressed(ap,actions=actions,source_only=np.array([r['candidate_id']==0 for r in rows],bool),
                root_contexts=np.array([r['snapshot_context_sha256'] for r in rows]))
            spec['bridge_action_plan']=dict(path=str(ap),sha256=sha(ap))
        write(self.root/(name+'_spec.json'),spec);out=self.root/name
        evaluate(spec,out,session=session);receipt=read(out/'status.json');results=read(out/'results.json')
        if receipt['phase']!='completed' or any(r['label'] not in (0,1) or len(r['attempts'])!=1 for r in results):raise ValueError('unknown/incomplete evaluation')
        if receipt['charged_interactions']>c['reservation']:raise ValueError('evaluation exceeds reservation')
        c.update(phase='completed',**{k:receipt[k] for k in ('charged_interactions','active_interactions','padding_interactions')})
        write(self.root/'costs.json',self.costs);self.status('running');return results


def collection_rows(stage,b,out):
    raw=read(out/'candidates.json');ledger=[];legal=[]
    from .retention_repair_report import CachedArchive
    with CachedArchive(out/'prefixes.npz') as f:
        for r in raw:
            i=r['index'];eid=b['episode_offset']+i;role=b['role'];ancestor=f'{role}-{stage.p["config"]["TRAIN_seed" if role=="TRAIN" else "solver_dev_seed"]}-{eid}'
            row={**r,'root_id':ancestor,'root_episode_id':ancestor,'data_role':'train' if role=='TRAIN' else 'solver_dev','role':role,
                'onset':r['pulse_scheduled_start_step'],'snapshot_time':float(f['time_after'][r['snapshot_control_step']-1,i]),
                'collection':str(out),'collection_lane':i,'logical_episode_id':eid}
            if r['snapshot'] is not None:
                if r['pulse_applied_steps']!=3 or r['snapshot_control_step']!=r['pulse_scheduled_start_step']+3:raise ValueError('not immediately post-pulse')
                legal.append(row)
            ledger.append(row)
    write(stage.root/(b['name']+'_ledger.json'),ledger);return ledger,legal


def independent_checks(stage,rows,prefix):
    from .generative_bridge.teacher_runtime import TeacherEvaluationSession
    session=TeacherEvaluationSession()
    try:return stage.evaluate(prefix,rows,session=session)
    finally:session.close()


def select_with_repeats(stage,rows,first,prefix,target):
    # Repetition subset is chosen only from independent pi0 failures, before G or Noise.
    labels={r['root_id']:[r['label']] for r in first}
    negative=[r for r in rows if labels[r['root_id']]==[0]]
    chosen=choose_roots(negative,labels,min(target,len(negative)))
    from .generative_bridge.teacher_runtime import TeacherEvaluationSession
    session=TeacherEvaluationSession()
    try:
        for repeat in (1,2):
            result=stage.evaluate(f'{prefix}_pi0_repeat{repeat}',chosen,session=session)
            for r in result:labels[r['root_id']].append(r['label'])
    finally:session.close()
    selected=choose_roots(chosen,labels,target)
    # R5 is a separate bank/controller, read-only and never teacher initialization.
    if selected:
        r5=stage.evaluate(f'{prefix}_R5_control',selected,model='R5')
        by_r5={r['root_id']:r for r in r5}
    else:by_r5={}
    write(stage.root/(prefix+'_baseline_labels.json'),labels)
    for r in chosen:
        r['pi0_labels']=labels[r['root_id']];r['baseline_class']=baseline_class(r['pi0_labels'])
        r['R5_label']=by_r5[r['root_id']]['label'] if r['root_id'] in by_r5 else None
    write(stage.root/(prefix+'_selection.json'),dict(candidates=chosen,selected=[r['root_id'] for r in selected],selection_uses_teacher=False))
    first_by={r['root_id']:r for r in first}
    for r in selected:r['pi0_attempt']=first_by[r['root_id']]['attempts'][0]
    return selected,labels


def teachers(stage,selected,ordinal_offset=0,proposal_reference=None):
    import jax
    import jax.numpy as jp
    from .generative_bridge.worker import generator_template
    from .generative_bridge.diffusion import restore_state,ddim_sample
    from .generative_bridge.proposals import stable_seed
    from .generative_bridge.teacher import select_teacher
    from .generative_bridge.production import teacher_candidate,lane_arrays
    from .generative_bridge.teacher_runtime import TeacherEvaluationSession
    from .unified_envelope_snapshot import load_unified_envelope_snapshot
    net,template,identity=generator_template(stage.p['models']['pi0']['frozen_policy'],stage.p['config']['proposal_seed'])
    if identity!=read(stage.p['G']['manifest'])['identity']:raise ValueError('G normalization reference differs')
    state=restore_state(Path(stage.p['G']['manifest']).parent,template,identity)
    generation=jax.jit(lambda obs,noise:ddim_sample(lambda x,o,k:net.apply(state['ema'],x,o,k),obs,noise))
    session=TeacherEvaluationSession();output=[]
    try:
        for ordinal,base in enumerate(selected,start=ordinal_offset):
            snap=load_unified_envelope_snapshot(Path(base['snapshot']));actual=lane_arrays(base['pi0_attempt'])
            reference=actual['normalized_action_executed'][:16]
            if not len(reference):raise ValueError('no source trajectory')
            if len(reference)<16:reference=np.concatenate((reference,np.repeat(reference[-1:],16-len(reference),axis=0)))
            gnoise=np.stack([np.random.default_rng(stable_seed(stage.p['config']['proposal_seed'],base['root_id'],cid)).normal(size=(16,4)) for cid in range(1,17)]).astype('f4')
            obs=np.repeat(np.asarray(snap.observation)[None],16,axis=0)
            normalized=(jp.asarray(obs)-state['normalizer']['mean'])/state['normalizer']['std']
            generated=np.asarray(generation(normalized,jp.asarray(gnoise)),np.float32)
            noise_raw=np.stack([np.random.default_rng(stable_seed(stage.p['config']['proposal_seed'],base['root_id']+'-Noise',cid)).normal(size=(16,4)) for cid in range(1,17)]).astype('f4')
            colored=noise_raw.copy();rho=stage.p['config']['noise_rho']
            for tick in range(1,16):colored[:,tick]=rho*colored[:,tick-1]+np.sqrt(1-rho*rho)*noise_raw[:,tick]
            noise_actions=np.clip(reference[None]+stage.p['config']['noise_scale']*colored,-1,1).astype('f4')
            pools={'G':np.concatenate((reference[None],generated)),'Noise':np.concatenate((reference[None],noise_actions))}
            proposal=stage.root/f'root_{ordinal:03d}_proposals.npz'
            if proposal_reference is not None:
                if sha(proposal_reference['path'])!=proposal_reference['sha256']:raise ValueError('failed-root frozen proposals changed')
                import shutil
                shutil.copyfile(proposal_reference['path'],proposal)
                with np.load(proposal) as saved:pools={k:saved[k] for k in ('G','Noise')}
            else:np.savez_compressed(proposal,G=pools['G'],Noise=pools['Noise'],G_noise=gnoise,noise_raw=noise_raw)
            result=dict(root_id=base['root_id'],root_episode_id=base['root_episode_id'],role=base['role'],onset=base['onset'],
                snapshot=base['snapshot'],state_sha256=base['state_sha256'],snapshot_context_sha256=base['snapshot_context_sha256'],
                pi0_labels=base['pi0_labels'],R5_label=base['R5_label'],tail=stage.p['models']['pi0']['policy'],G=stage.p['G'],
                proposals=str(proposal),proposals_sha256=sha(proposal),methods={},student_absorption='NOT_RUN',independent_student_success='NOT_RUN')
            candidates=[{**base,'index':cid,'candidate_id':cid} for cid in range(17)]
            for method,actions in pools.items():
                search=stage.evaluate(f'root_{ordinal:03d}_{method}_search',candidates,actions=actions,session=session)
                scored=[teacher_candidate(r,'diffusion' if method=='G' else 'colored_noise',r['candidate_id'],snap.last_action) for r in search[1:]]
                winner=select_teacher(scored);cid=winner['candidate_id'] if winner else None
                lock=dict(root=base['root_id'],method=method,candidate_id=cid,actions_sha256=sha(proposal),layout=17,
                    logical_rng_count=17,logical_rng_indices=list(range(17)),fixed_before_replays=True)
                write(stage.root/f'root_{ordinal:03d}_{method}_winner_lock.json',lock)
                source_labels=[search[0]['label']];winner_labels=[search[cid]['label']] if cid is not None else None;attempts=[search[cid]['attempts'][0]] if cid is not None else []
                # Both repeats dispatch original whole batch even if no winner or source succeeded.
                for repeat in (1,2):
                    replay=stage.evaluate(f'root_{ordinal:03d}_{method}_repeat{repeat}',candidates,actions=actions,session=session)
                    source_labels.append(replay[0]['label'])
                    if cid is not None:winner_labels.append(replay[cid]['label']);attempts.append(replay[cid]['attempts'][0])
                status=qualify_method(source_labels,winner_labels)
                result['methods'][method]=dict(status=status,source_labels=source_labels,winner_labels=winner_labels,
                    selected_candidate_id=cid,search_successful_candidates=[r['candidate_id'] for r in search[1:] if r['label']==1],
                    verified_attempts=attempts,winner_lock=lock,teacher_solved=cid is not None,repeat_verified=status=='verified_solution')
            ambiguity=any(m['status'] in ('source_ambiguous','winner_ambiguous') for m in result['methods'].values())
            g=result['methods']['G']['repeat_verified'];n=result['methods']['Noise']['repeat_verified']
            result['conversion_cell']='ambiguous' if ambiguity else 'both' if g and n else 'G_only' if g else 'Noise_only' if n else 'neither'
            result['qualified_G_incremental_lesson']=g and not ambiguity and base['role']=='TRAIN' and base['pi0_labels']==[0,0,0]
            write(stage.root/f'root_{ordinal:03d}_result.json',result);output.append(result)
            write(stage.root/'teacher_results.json',output)
    finally:session.close()
    return output


def run_d1(stage):
    ledgers=[];train=[]
    for ordinal,b in enumerate(stage.p['batches'][:2]):
        out=stage.collect(b);ledger,legal=collection_rows(stage,b,out);ledgers.extend(ledger)
        first=independent_checks(stage,legal,b['name']+'_pi0_all')
        remaining=stage.p['config']['root_target']-len(train)
        selected,labels=select_with_repeats(stage,legal,first,'train_block'+str(ordinal),remaining)
        train.extend(selected)
        # A second fresh block is authorized only when stable failures are insufficient.
        if len(train)>=stage.p['config']['root_target']:break
    b=stage.p['batches'][2];out=stage.collect(b);ledger,legal=collection_rows(stage,b,out);ledgers.extend(ledger)
    first=independent_checks(stage,legal,'solver_dev_pi0_all')
    solver,solver_labels=select_with_repeats(stage,legal,first,'solver_dev',stage.p['config']['solver_dev_target'])
    write(stage.root/'collection_ledger.json',ledgers);write(stage.root/'selected_roots.json',train+solver)
    teachers(stage,train+solver) if train or solver else write(stage.root/'teacher_results.json',[])
    write(stage.root/'D1_completed.json',dict(phase='completed',selected_train=len(train),selected_solver_dev=len(solver),
        selected_roots_sha256=sha(stage.root/'selected_roots.json'),teacher_results_sha256=sha(stage.root/'teacher_results.json'),
        updates=dict(student=0,E=0,G=0),source_lock_sha256=sha(stage.root/'source_lock.json')))


def run(path):
    if read(path)['stage']=='D2':raise ValueError('D2 is preparation only; formal arms require a separate finite launch; engineering anchors/micro use run_retention_d2.py')
    # Set platform before audit imports JAX or deserializes device arrays.
    if read(path)['stage']=='D1' and not read(path).get('resume'):os.environ['JAX_PLATFORMS']='cuda,cpu'
    stage=Stage(path)
    write(stage.root/'ACTIVE_RUN.json',dict(execution=str(stage.root/'status.json'),lineage=str(stage.root/'status.json')))
    with (stage.root/'watcher.log').open('x') as log:
        watcher=subprocess.Popen([PY,str(stage.code/'JIT/cli/watch_run_errors.py'),'--active-run',str(stage.root/'ACTIVE_RUN.json'),'--state-dir',str(stage.root/'notifications')],cwd=stage.code,env=stage.env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(stage.root/'launch.json',dict(pid=os.getpid(),watcher_pid=watcher.pid,execute=True,plan=str(path),started_unix=time.time()))
    try:
        if stage.p['stage']=='V':
            from .retention_repair_execution import verify
            for b in stage.p['batches']:stage.collect(b);verify(b,stage.root)
        elif stage.p.get('resume'):
            run_resume(stage)
        else:
            os.environ.update(stage.env)
            run_d1(stage)
        audit(path);stage.status('completed',result='evaluation_complete; no automatic next stage')
    except BaseException as exc:stage.status('failed',error=repr(exc));raise


def worker(request_path):
    request=read(request_path)
    if request.get('execute_authorized') is not True:raise ValueError('worker lacks finite parent authorization')
    if sha(request['plan'])!=request['plan_sha256']:raise ValueError('worker plan changed')
    os.environ['JAX_PLATFORMS']='cuda,cpu'
    audit(request['plan']);p=read(request['plan']);root=Path(request['output']);root.mkdir(exist_ok=False)
    # Only this root's six original17-world batches live in the GPU process.
    stage=object.__new__(Stage);stage.plan_path=Path(request['plan']);stage.p=p;stage.root=root;stage.code=Path(p['code'])
    stage.start=time.monotonic();stage.costs=[];stage.active='worker_start';stage.env=dict(os.environ)
    try:
        results=teachers(stage,[request['root']],ordinal_offset=request['ordinal'],proposal_reference=request.get('proposal_reference'))
        stage.status('completed');write(root/'completed.json',dict(phase='completed',results=results,
            charged_interactions=sum(c['charged_interactions'] for c in stage.costs),costs=stage.costs,
            original_layout=17,search_and_repeat_counts=3,student_E_G_updates=0))
    except BaseException as exc:stage.status('failed',error=repr(exc));raise


def run_resume(stage):
    previous=Path(stage.p['resume']['previous']);all_roots=read(stage.root/'selected_roots.json')
    results=read(previous/'teacher_results.json')
    stage.costs.append(dict(stage='inherited_attempt_including_failed_reservation',charged_interactions=stage.p['resume']['inherited_charge'],
        phase='inherited_completed_and_engineering_failure',source=str(previous/'costs.json'),optimizer_updates=0))
    write(stage.root/'teacher_results.json',results)
    for ordinal in stage.p['resume']['remaining_indices']:
        out=stage.root/f'worker_root_{ordinal:03d}';request=dict(plan=str(stage.plan_path),plan_sha256=sha(stage.plan_path),root=all_roots[ordinal],
            ordinal=ordinal,output=str(out),execute_authorized=True)
        prior=previous/f'root_{ordinal:03d}_proposals.npz'
        if prior.exists():request['proposal_reference']=dict(path=str(prior),sha256=sha(prior))
        req=stage.root/f'worker_root_{ordinal:03d}_request.json';write(req,request)
        cost=stage.reserve(f'worker_root_{ordinal:03d}',2*3*17*400)
        with (stage.root/f'worker_root_{ordinal:03d}.log').open('x') as log:
            child=subprocess.Popen([PY,str(stage.code/'JIT/cli/run_retention_next.py'),'worker','--request',str(req)],cwd=stage.code,
                env=stage.env,stdout=log,stderr=subprocess.STDOUT)
            stage.status('running',child_pid=child.pid)
            try:rc=child.wait(timeout=max(.1,stage.p['budget']['wall_seconds']-(time.monotonic()-stage.start)))
            except subprocess.TimeoutExpired:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                raise TimeoutError('D1 global remaining wall budget exhausted')
        if rc:raise RuntimeError(f'root{ordinal} worker exited{rc}; no automatic retry')
        receipt=read(out/'completed.json')
        if receipt['phase']!='completed' or receipt['charged_interactions']>cost['reservation']:raise ValueError('worker receipt invalid')
        cost.update(phase='completed',charged_interactions=receipt['charged_interactions'],receipt=str(out/'completed.json'),
            receipt_sha256=sha(out/'completed.json'))
        results.extend(receipt['results']);write(stage.root/'teacher_results.json',results);write(stage.root/'costs.json',stage.costs)
    if set(r['root_id'] for r in results)!=set(r['root_id'] for r in all_roots):raise ValueError('incomplete resumed D1')
    write(stage.root/'D1_completed.json',dict(phase='completed',selected_train=sum(r['role']=='TRAIN' for r in all_roots),
        selected_solver_dev=sum(r['role']=='SOLVER_DEV' for r in all_roots),selected_roots_sha256=sha(stage.root/'selected_roots.json'),
        teacher_results_sha256=sha(stage.root/'teacher_results.json'),updates=dict(student=0,E=0,G=0),source_lock_sha256=sha(stage.root/'source_lock.json'),
        previous_failed_status=str(previous/'status.json'),failure_charges_retained=True))
