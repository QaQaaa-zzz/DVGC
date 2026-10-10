"""Conditional D2 inputs and bounded engineering micro; no formal arm launcher."""
from pathlib import Path
import os,time,subprocess,pickle
import numpy as np
from .retention_repair import read,write,sha
from .retention_repair_execution import PY


def prepare(evidence,output,repository):
    evidence=Path(evidence);m=read(evidence/'teacher_dataset_manifest.json')
    if not m['qualified_gate'] or sha(m['D1_receipt'])!=m['D1_receipt_sha256']:raise ValueError('qualified D1 receipt required')
    receipt=read(m['D1_receipt'])
    if receipt['phase']!='completed':raise ValueError('D1 incomplete')
    d1=read(Path(m['D1_receipt']).parent/'plan.json');root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    code=Path(repository).resolve();model=d1['models']['pi0'];source=read(model['policy']['formal_config']);ppo=source['ppo']
    from .retention_next import random_initials,identities
    actual=identities(d1['models']);seed=1010257101
    q,v,precheck=random_initials(model,d1['indices'],16,seed)
    q=np.concatenate((np.tile(d1['indices']['nominal_qpos'],(16,1)),q)).astype('f4')
    v=np.concatenate((np.tile(d1['indices']['nominal_qvel'],(16,1)),v)).astype('f4')
    initial=root/'keep_initial.npz';requests=root/'keep_requests.npz'
    np.savez_compressed(initial,qpos=q,qvel=v);np.savez_compressed(requests,requested=np.zeros((400,32,4),np.float32),onsets=np.zeros(32,np.int32))
    template=read('/home/qy/DVGC/JIT/runs/experiments/retention_next_20261010/V_002/pi0_main0_spec.json')
    template.update(num_envs=32,seed=seed,role='TRAIN',initial_state_bank=str(initial),frozen_request_table=dict(path=str(requests),sha256=sha(requests)))
    write(root/'keep_spec.json',template);write(root/'keep_precheck.json',precheck)
    obs=[];actions=[];weights=[];origins=[];ancestors=[]
    for entry in m['TRAIN']:
        if entry['role']!='TRAIN' or sha(entry['path'])!=entry['sha256']:raise ValueError('teacher lesson drift')
        with np.load(entry['path']) as f:o=f['observations'];a=f['actions'];origin=f['origins']
        # Action-origin enum is defined by evaluator: bridge16 prefix code=2; source=0.
        segments=np.where(np.arange(len(o))<16,'bridge_prefix','source_tail')
        w=np.zeros(len(o))
        for label in ('bridge_prefix','source_tail'):
            mask=segments==label
            if not mask.any():raise ValueError('incomplete prefix/tail teacher')
            w[mask]=.5/mask.sum()/len(m['TRAIN'])
        obs.extend(o);actions.extend(a);weights.extend(w);origins.extend(segments.tolist());ancestors.extend([entry['root_episode_id']]*len(o))
    dp=root/'demo.npz';np.savez_compressed(dp,observations=np.asarray(obs),actions=np.asarray(actions),weights=np.asarray(weights))
    dm=dict(schema='jit_bridge_demo_v1_2',path=str(dp),sha256=sha(dp),count=len(obs),sample_roots=ancestors,sample_origins=origins,role='TRAIN',teacher_manifest=str(evidence/'teacher_dataset_manifest.json'))
    write(root/'demo_manifest.json',dm)
    budget=dict(PPO_A_C=2*128000,BC_supervised_updates=2000,keep_collection=32*400,GPU_micro=100,
        new_dev_pi0_R5=2*64*400,BC_checkpoints_dev=4*64*400,AC_stop_checks=8*32*400,
        final_four_cells=3*64*400,final_teacher_student_combinations=3*len(m['SOLVER_DEV'])*4*17*400,
        hard_cap=2000000,wall_seconds=43200)
    budget['physics_total']=sum(v for k,v in budget.items() if k not in ('BC_supervised_updates','hard_cap','wall_seconds'))
    if budget['physics_total']>budget['hard_cap']:raise ValueError('D2 actual layout over budget')
    bcsource='/home/qy/DVGC/JIT/runs/experiments/generative_bridge_v1_2_20260928/recovery_0003/campaign/warmup/training/warmup_status.json'
    warm=read(bcsource)
    plan=dict(schema='jit_retention_D2_preparation_v2',stage='D2',output=str(root),code=str(code),code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip(),
        D1_receipt=m['D1_receipt'],D1_receipt_sha256=m['D1_receipt_sha256'],models=d1['models'],identities=actual,indices=d1['indices'],
        teacher_manifest=str(evidence/'teacher_dataset_manifest.json'),demo_manifest=str(root/'demo_manifest.json'),
        ppo_source=ppo,ppo_learning_rate=ppo['learning_rate']/4,learning_rate_scope='whole PPO learner; no separate actor optimizer introduced',
        BC=dict(learning_rate=warm['learning_rate'],demo_coefficient=warm['demo_coefficient'],keep_coefficient=warm['keep_coefficient'],updates=2000,checkpoints=[100,500,1000,2000],source_record=bcsource),
        auxiliary=dict(demo_start=.2,demo_end=.05,keep=.2,freeze_actor_normalizer=True,normalizer_scope='entire Actor and privileged critic',demo_clock='completed_transitions',schedule_transitions=128000),
        arms=dict(A=dict(initializer='same pi0',objective='PPO+keep',transitions=128000),B=dict(initializer='same pi0',objective='BC+keep',updates=2000),C=dict(initializer='exact selected B Actor, critic, normalizer, RNG; explicit optimizer phase handoff',objective='PPO+demo+keep',transitions=128000)),
        chunk_clock='single128000 stage; checkpoints32000/64000/96000/128000; preserve_stage_clock=True on same-stage full learner resume',
        reset_mixture=dict(nominal_complete=.25,random_complete=.25,pi0_success_snapshots=.25,teacher_recoverable=.25),
        empty_pool_behavior='fail closed; no redistribution',TRAIN_seed=seed,solver_dev_roots=m['SOLVER_DEV'],
        keep_spec=str(root/'keep_spec.json'),budget=budget,micro=dict(transitions=100,envs=2,unroll=25,minibatches=1,batch_size=2,actual_updates=2,reset='nominal complete TRAIN engineering only'),
        formal_execution_authorized=False,automatic_training=False,published_policy=None,best_dev_candidate=d1['models']['R5'],learner_last=None,
        formal_gates=['fresh TRAIN keep receipt and complete reset pool adapter','BC full Adam/RNG checkpoint and explicit BC-to-PPO optimizer handoff','new independent64-episode DEV panel locked before formal arms','choose B before authorizing PPO A/C'],
        stop_rule='B/D loss >5pp versus pi0: repeat confirm then stop affected arm; no TEST use')
    locks={str(evidence/'teacher_dataset_manifest.json'):sha(evidence/'teacher_dataset_manifest.json'),m['D1_receipt']:m['D1_receipt_sha256'],bcsource:sha(bcsource)}
    for entry in m['TRAIN']:locks[entry['path']]=entry['sha256']
    for f in (initial,requests,root/'keep_spec.json',dp,root/'demo_manifest.json'):locks[str(f)]=sha(f)
    for name in ('retention_d2.py','generative_bridge/student.py','generative_bridge/actual_update.py','generative_bridge/learner_continuation.py','generative_bridge/student_clock.py','generative_bridge/warmup.py'):
        f=code/'JIT/src/jit_dvgc'/name;locks[str(f)]=sha(f)
    for model in plan['models'].values():
        for f in (model['frozen_policy'],model['policy']['formal_config'],str(Path(model['policy']['checkpoint'])/'payload.pkl')):locks[f]=sha(f)
    plan['locks']=locks;write(root/'source_lock.json',dict(identities=actual,locks=locks));write(root/'budget_dry_run.json',budget);write(root/'plan.json',plan)
    write(root/'status.json',dict(phase='prepared',stage='D2_preparation',formal_training_started=False,charged_interactions=0))
    return root/'plan.json'


def audit(path):
    p=read(path)
    for f,h in p['locks'].items():
        if sha(f)!=h:raise ValueError('D2 input drift '+f)
    if p['formal_execution_authorized'] or p['automatic_training']:raise ValueError('no formal auto execution')
    return p


def anchors(path):
    p=audit(path);root=Path(p['output']);out=root/'keep_collection'
    if out.exists():raise ValueError('no implicit retry')
    write(root/'status.json',dict(phase='running',stage='D2_keep_acquisition',charged_interactions=12800,training_updates=0,updated_unix=time.time()))
    env=dict(os.environ,PYTHONPATH=str(Path(p['code'])/'JIT/src'),JAX_PLATFORMS='cuda,cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false')
    with (root/'keep_collection.log').open('x') as log:
        subprocess.run([PY,str(Path(p['code'])/'JIT/cli/run_pulse_exploration.py'),'--mode','collect','--spec',p['keep_spec'],'--output',str(out)],env=env,cwd=p['code'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3600)
    from .retention_repair_execution import verify
    vr=verify(dict(name='keep_collection',spec=p['keep_spec'],capacity=32,scored=32),root)
    from .retention_repair_report import CachedArchive
    observations=[];weights=[];groups={};episodes=[]
    with CachedArchive(out/'prefixes.npz') as f:
        successful=[i for i in range(32) if np.any(f['success'][:,i]&f['prefix_mask'][:,i])]
        for group,lanes in [('nominal',[i for i in successful if i<16]),('random',[i for i in successful if i>=16])]:
            if not lanes:raise ValueError('empty successful TRAIN group '+group)
            groups[group]=lanes
            for lane in lanes:
                mask=f['prefix_mask'][:,lane];o=f['actor_observation_before'][mask,lane]
                observations.extend(o);weights.extend(np.full(len(o),.5/len(lanes)/len(o)))
                episodes.append(dict(ancestor=f'TRAIN_KEEP-{p["TRAIN_seed"]}-{lane}',lane=lane,group=group,success=True,steps=len(o)))
    dest=root/'keep_anchors.npz';np.savez_compressed(dest,actor_observation_before=np.asarray(observations),weights=np.asarray(weights))
    write(root/'keep_receipt.json',dict(role='train',data_role='TRAIN',seed=p['TRAIN_seed'],groups=groups,episodes=episodes,path=str(dest),sha256=sha(dest),full_success=True,charged_interactions=vr['charged_interactions'],DEV_used=False))
    write(root/'status.json',dict(phase='prepared',stage='D2_micro_ready',charged_interactions=vr['charged_interactions'],training_updates=0))


def micro(path):
    p=audit(path);root=Path(p['output']);run=root/'gpu_micro';run.mkdir(exist_ok=False);start=time.monotonic()
    import jax
    if jax.default_backend()!='gpu':raise ValueError('real GPU required')
    from torch.utils.tensorboard import SummaryWriter
    from .generative_bridge.worker import source_payload
    from .generative_bridge.student import make_joint_student_trainer,load_retention_traces
    from .generative_bridge.learner_continuation import LearnerHooks,instrument_trainer,trainer_sha
    from .generative_bridge.actual_update import instrument_updates
    from .unified_formal import build_unified_formal_environment,build_unified_formal_trainer_kwargs
    from .handoff_bank import pytree_sha256
    from .ppo_numerics import guard_ppo_updates
    from brax.training.agents.ppo import train
    pol,payload=source_payload(p['models']['pi0']['frozen_policy']);cfg,artifact,env=build_unified_formal_environment(Path(pol['formal_config']))
    env._training_action_pulse=None;env.reset=env._reset_jump_start_unified
    keep=read(root/'keep_receipt.json');retention=load_retention_traces(keep);demo=read(p['demo_manifest'])
    writer=SummaryWriter(str(run/'tensorboard'));rows=[]
    write(root/'status.json',dict(phase='running',stage='D2_GPU_micro',planned_training_transitions=100,completed_training_transitions=0,charged_interactions=12900,training_updates=0,updated_unix=time.time()))
    def sink(row):
        rows.append(row);step=len(rows)*50
        for k,v in row.items():writer.add_scalar(k,v,step)
        writer.flush();write(run/'actual_updates.json',rows)
        write(root/'status.json',dict(phase='running',stage='D2_GPU_micro',planned_training_transitions=100,completed_training_transitions=step,charged_interactions=12900,training_updates=len(rows),updated_unix=time.time()))
    class Callbacks:
        def on_progress(self,step,metrics):pass
        def on_policy_params(self,step,make_policy,params):
            with (run/f'inference_{step}.pkl').open('wb') as f:pickle.dump(jax.device_get(params),f)
    hooks=LearnerHooks(run/'learner',contract='retention_same_stage_micro_v1',trainer_sha=trainer_sha(train.train),parent=None,preserve_stage_clock=True)
    wrapped=instrument_trainer(train.train,hooks.initialize,hooks.save)
    wrapped=instrument_updates(wrapped,sink,max_grad_norm=cfg.ppo.max_grad_norm,probe=retention[0][:32])
    wrapped=make_joint_student_trainer(wrapped,demo,retention=retention,transitions=128000,freeze_actor_normalizer=True,demo_clock='completed_transitions',actual_gradient_audit=True,demo_batch_size=16,retention_batch_size=16)
    kw=build_unified_formal_trainer_kwargs(cfg,env,Callbacks());kw.update(num_timesteps=100,num_envs=2,batch_size=2,num_minibatches=1,num_updates_per_batch=1,unroll_length=25,num_evals=3,learning_rate=p['ppo_learning_rate'],restore_params=(payload.observation_normalizer,payload.actor_params,payload.critic_params),restore_value_fn=True,log_training_metrics=False,training_metrics_steps=50)
    try:
        with guard_ppo_updates(run):inference_fn,params,metrics=wrapped(**kw)
        jax.effects_barrier()
        last=read(hooks.latest_receipt()['path'])
        with open(last['state_path'],'rb') as f:saved=pickle.load(f)
        state=saved['training_state'];source_hash=pytree_sha256(payload.observation_normalizer)
        fresh=state.replace(optimizer_state=jax.tree.map(lambda x:np.zeros_like(x),state.optimizer_state),env_steps=type(state.env_steps)(hi=0,lo=0))
        resumed=LearnerHooks(run/'resume_check',contract='retention_same_stage_micro_v1',trainer_sha=trainer_sha(train.train),parent=hooks.latest_receipt(),preserve_stage_clock=True)
        restored,rng=resumed.initialize(fresh,jax.random.PRNGKey(9))
        obs={'state':retention[0][:32],'privileged_state':np.zeros((32,106),np.float32)}
        before=np.asarray(inference_fn(params,deterministic=True)(obs,jax.random.PRNGKey(1))[0])
        after=np.asarray(inference_fn((restored.normalizer_params,restored.params.policy,restored.params.value),deterministic=True)(obs,jax.random.PRNGKey(1))[0])
        checks=dict(real_GPU=jax.default_backend()=='gpu',normalizer_unchanged=last['normalizer_sha256']==source_hash,
            actor_changed=last['actor_sha256']!=pol['actor_sha256'],critic_changed=last['critic_sha256']!=pol['critic_sha256'],
            optimizer_changed=last['optimizer_sha256']!=read(run/'learner/initialization.json')['optimizer_sha256'],
            completed_transitions=int(state.env_steps),resume_clock=int(restored.env_steps),actual_optimizer_updates=len(rows),
            lambda_values=[r['effective_lambda_demo'] for r in rows],serialization_action_max_abs=float(np.max(np.abs(before-after))),
            full_state_restore_exact=pytree_sha256((restored,rng))==pytree_sha256((state,saved['rng'])),
            actual_gradient_sum_norm_error=max(r['actual/actor_gradient_norm_sum_difference'] for r in rows))
        passed=checks['normalizer_unchanged'] and checks['actor_changed'] and checks['critic_changed'] and checks['optimizer_changed'] and checks['completed_transitions']==100 and checks['resume_clock']==100 and checks['actual_optimizer_updates']==2 and checks['serialization_action_max_abs']==0 and checks['full_state_restore_exact'] and checks['lambda_values'][1]<checks['lambda_values'][0]
        write(run/'verification.json',dict(phase='passed' if passed else 'failed',checks=checks,wall_seconds=time.monotonic()-start,scope='engineering only; not ability improvement or formal A/B/C',learner_last=hooks.latest_receipt(),published_policy=None))
        if not passed:raise ValueError('GPU micro invariant failure')
        write(root/'status.json',dict(phase='completed',stage='D2_preparation_micro',completed_training_transitions=100,planned_training_transitions=100,training_updates=2,charged_interactions=12900,formal_training_started=False,updated_unix=time.time()))
    except BaseException as e:
        write(root/'status.json',dict(phase='failed',stage='D2_GPU_micro',error=repr(e),charged_interactions=12900,training_updates=len(rows),formal_training_started=False,updated_unix=time.time()));raise
    finally:writer.close()
