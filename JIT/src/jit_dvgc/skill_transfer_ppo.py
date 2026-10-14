"""Explicit recovery PPO worker; importing this module starts no training."""
from pathlib import Path
import dataclasses
import json
import pickle
import threading
import time
import numpy as np


def _read(value):
    return json.loads(Path(value).read_text()) if isinstance(value,(str,Path)) else dict(value)


def _path(value):
    from .generative_bridge.contracts import file_sha
    if isinstance(value,dict):
        if file_sha(value['path'])!=value['sha256']:
            raise ValueError('recovery PPO input hash mismatch')
        return Path(value['path'])
    return Path(value)


def validate_recovery_ppo_config(config):
    """Validate budgets without loading JAX or constructing physical environments."""
    config=dict(config)
    for key in ('pi0_descriptor','initializer_inference','curriculum_inventory','teacher_manifest','runtime_config'):
        if not config.get(key):raise ValueError('missing recovery PPO input: '+key)
    ppo=dict(num_envs=128,unroll_length=25,batch_size=16,num_minibatches=8,num_updates_per_batch=1)
    rename={'num_parallel_envs':'num_envs','requested_transitions':'num_timesteps','episode_horizon':'episode_length'}
    ppo.update({rename.get(k,k):v for k,v in config.get('ppo',{}).items()})
    block=int(config.get('transitions',128000))
    if any(type(ppo[k]) is not int or ppo[k]<=0 for k in ('num_envs','unroll_length','batch_size','num_minibatches','num_updates_per_batch')):
        raise ValueError('positive integer PPO dimensions required')
    # Installed Brax rollout = batch_size * minibatches * unroll, not num_envs alone.
    rollout=ppo['batch_size']*ppo['num_minibatches']*ppo['unroll_length']
    if ppo['batch_size']*ppo['num_minibatches']%ppo['num_envs'] or block<=0 or block%rollout:
        raise ValueError('recovery PPO budget/dimensions must align to complete Brax rollouts')
    from .recovery_curriculum import STAGES
    weights=config.get('curriculum_weights')
    if weights is None:
        prefix=int(config.get('prefix_steps',16))
        if prefix not in STAGES:raise ValueError('unsupported recovery prefix stage')
        weights=[float(s==prefix) for s in STAGES]
    weights=np.asarray(weights,float)
    if weights.shape!=(5,) or not np.isfinite(weights).all() or (weights<0).any() or weights.sum()<=0:
        raise ValueError('invalid curriculum stage weights')
    if config.get('keep_coefficient',0)!=0 or config.get('retention') is not None:
        raise ValueError('R PPO forbids keep constraints')
    if config.get('run_evals',False) is not False:
        raise ValueError('review worker does not run DEV evaluations')
    if config.get('preserve_snapshot_episode_context',True) is not True:
        raise ValueError('R requires original snapshot episode context')
    config.update(ppo=ppo,transitions=block,curriculum_weights=weights.tolist())
    return config


def run_recovery_ppo(config, output):
    """Run one explicitly budgeted reviewblock, returning its final receipt.

    Config accepts a JSON path or dict. Each input path also accepts {path,sha256}.
    initializer_inference is a pickled (normalizer,Actor,Critic) triple, including
    after BC: the source Critic is identity evidence and is replaced on fresh PPO.
    Resume requires parent_learner and the preceding matching inference triple.
    Physical episodes restart; full learner/RNG resume uses a declared lifetime
    offset with a new local block clock. No DEV or retention-bank access occurs.
    """
    config=validate_recovery_ppo_config(_read(config))
    import jax
    from tensorboardX import SummaryWriter
    from brax.training.agents.ppo import train
    from .generative_bridge.worker import source_payload
    from .generative_bridge.contracts import digest,file_sha
    from .generative_bridge.protocol import atomic_json
    from .generative_bridge.learner_continuation import LearnerHooks,trainer_sha
    from .generative_bridge.split_optimizer import make_recovery_trainer,PINNED_BRAX_TRAIN_SHA256
    from .generative_bridge.actual_update import adam_update_counts
    from .handoff_bank import pytree_sha256
    from .ppo_numerics import guard_ppo_updates
    from .unified_formal import build_unified_formal_environment,build_unified_formal_trainer_kwargs
    from .recovery_curriculum import RecoveryCurriculumReset,wrap_recovery_for_training,validate_teacher_entries,curriculum_probabilities
    from .checkpoint import save_checkpoint
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    paths={key:_path(config[key]) for key in ('pi0_descriptor','initializer_inference','curriculum_inventory','teacher_manifest','runtime_config')}
    policy,payload=source_payload(paths['pi0_descriptor'])
    teacher=_read(paths['teacher_manifest']);teachers=validate_teacher_entries(teacher)
    inventory=_read(paths['curriculum_inventory'])
    if inventory['teacher_manifest_sha256']!=file_sha(paths['teacher_manifest']):
        raise ValueError('curriculum and teacher manifest differ')
    curriculum_probabilities(inventory['entries'],config['curriculum_weights'])
    cfg,artifact,env=build_unified_formal_environment(paths['runtime_config'])
    if cfg.raw.get('reward_mode')!='original_all_phases' or cfg.raw.get('success_criterion')!='stable_forward_recovery':
        raise ValueError('R requires original_all_phases and stable_forward_recovery')
    env._training_action_pulse=None
    adapter=RecoveryCurriculumReset(env,inventory,weights=config['curriculum_weights']);adapter.install()
    with paths['initializer_inference'].open('rb') as stream:initializer=pickle.load(stream)
    if not isinstance(initializer,(tuple,list)) or len(initializer)!=3:
        raise ValueError('initializer must contain normalizer,Actor,Critic')
    if pytree_sha256(initializer[0])!=policy['normalizer_sha256']:
        raise ValueError('initializer normalizer differs from frozen pi0')
    parent=config.get('parent_learner')
    # Identity excludes stage weights, block length and initializer; curriculum
    # promotion may change these while retaining the complete learned state.
    contract=dict(schema='jit_recovery_split_ppo_v1',ppo=config['ppo'],
        pi0_descriptor_sha256=file_sha(paths['pi0_descriptor']),runtime_sha256=file_sha(paths['runtime_config']),
        teacher_sha256=file_sha(paths['teacher_manifest']),inventory_sha256=file_sha(paths['curriculum_inventory']),
        split_optimizer=config.get('split_optimizer',{}),demo_manifest=config.get('demo_manifest'),
        seed=int(config.get('seed',cfg.ppo.seed)),normalizer_frozen=True,keep_coefficient=0,
        demo_schedule_transitions=int(config.get('demo_schedule_transitions',config['transitions'])))
    hooks=LearnerHooks(root/'learner',contract=digest(contract),trainer_sha=trainer_sha(train.train),
        parent=parent,preserve_stage_clock=False)
    demo=_read(_path(config['demo_manifest'])) if config.get('demo_manifest') else None
    if demo:
        from .generative_bridge.student import load_optional_demo
        demo_data=load_optional_demo(demo);probe=demo_data[0][:32]
    else:
        trace=teachers[0]['source_trace']
        if file_sha(trace['path'])!=trace['sha256']:raise ValueError('probe teacher trace changed')
        with np.load(trace['path']) as data:probe=data['actor_observation_before'][:32].copy()
    if probe.ndim!=2 or probe.shape[1]!=76 or not np.isfinite(probe).all():
        raise ValueError('fixed TRAIN recovery probe required')
    started=time.monotonic();writer=SummaryWriter(str(root/'tensorboard'));lock=threading.RLock();rows=[];seen=[];usage={}
    atomic_json(root/'training_contract.json',{**contract,'block_config':config,'input_files':{str(p):file_sha(p) for p in paths.values()}})
    def status(phase,step,**extra):
        with lock:
            atomic_json(root/'status.json',dict(phase=phase,planned_training_transitions=config['transitions'],
                completed_training_transitions=int(step),lifetime_transitions=hooks.offset+int(step),
                actor_adam_count=max((int(r['actual/actor_adam_count']) for r in rows),default=0),
                critic_adam_count=max((int(r['actual/critic_adam_count']) for r in rows),default=0),
                updated_unix=time.time(),run_evals=False)|extra)
    def sink(row):
        if not all(np.isfinite(v) for v in row.values()):raise FloatingPointError('nonfinite actual R telemetry')
        if row['actual/actor_adam_count']!=row['actual/critic_adam_count']:
            raise ValueError('Actor/Critic clocks diverged')
        with lock:
            rows.append(row)
            with (root/'actual_updates.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
            clock=int(row['actual/completed_training_transitions']);update=int(row['actual/actor_adam_count'])
            for key,value in row.items():writer.add_scalar(key,value,update)
            writer.add_scalar('training/reward_per_transition',row['actual/reward_per_transition'],hooks.offset+clock)
            writer.flush();status('running',max(int(r['actual/completed_training_transitions']) for r in rows))
    class Callbacks:
        def on_progress(self,step,metrics):
            for key,value in metrics.items():
                if np.ndim(value)==0 and np.isfinite(float(value)):writer.add_scalar('Brax/'+key,float(value),hooks.offset+int(step))
            writer.flush()
        def on_policy_params(self,step,make_policy,params):
            jax.effects_barrier();step=int(step)
            if step in seen:raise ValueError('duplicate recovery PPO checkpoint')
            if pytree_sha256(params[0])!=policy['normalizer_sha256']:
                raise ValueError('frozen normalizer drift')
            if step==0 and pytree_sha256(params[1])!=pytree_sha256(initializer[1]):
                raise ValueError('initial R Actor identity mismatch')
            if step==0:
                atomic_json(root/'initialization_evidence.json',usage)
            target=root/f'inference_{step:06d}.pkl'
            with target.open('xb') as stream:pickle.dump(jax.device_get(params),stream,pickle.HIGHEST_PROTOCOL)
            checkpoint=root/'checkpoints'/f'transition_{step}'
            save_checkpoint(checkpoint,dataclasses.replace(payload,training_transitions=hooks.offset+step,
                observation_normalizer=params[0],actor_params=params[1],critic_params=params[2]))
            receipt=dict(schema='jit_recovery_ppo_checkpoint_v1',local_transitions=step,
                lifetime_transitions=hooks.offset+step,inference=str(target.resolve()),inference_sha256=file_sha(target),
                checkpoint=str(checkpoint.resolve()),learner=hooks.latest_receipt(),
                actor_sha256=pytree_sha256(params[1]),critic_sha256=pytree_sha256(params[2]),normalizer_sha256=pytree_sha256(params[0]),
                curriculum_weights=config['curriculum_weights'],simulator_state_restored=False)
            atomic_json(root/f'checkpoint_{step:06d}.json',receipt);seen.append(step);status('running',step)
    status('initializing',0)
    callbacks=Callbacks();kwargs=build_unified_formal_trainer_kwargs(cfg,env,callbacks)
    kwargs.update(config['ppo']);kwargs.update(num_timesteps=config['transitions'],num_evals=2,run_evals=False,
        num_eval_envs=8,seed=contract['seed'],restore_params=tuple(initializer),restore_value_fn=parent is not None,
        wrap_env_fn=wrap_recovery_for_training,training_metrics_steps=config['transitions'],log_training_metrics=True)
    kwargs.pop('eval_env',None)
    for key in ('learning_rate_schedule','desired_kl','learning_rate_schedule_min_lr','learning_rate_schedule_max_lr'):
        kwargs.pop(key,None)
    split=config.get('split_optimizer',{})
    offset=0
    if parent:
        manifest=_read(_path(parent));offset=int(manifest['lifetime_transitions'])
    budget=contract['demo_schedule_transitions'];start=float(config.get('demo_coefficient_start',.2));end=float(config.get('demo_coefficient_end',.05))
    coefficient=lambda clock:start+(end-start)*min(max(clock/budget,0),1)
    try:
        with guard_ppo_updates(root):
            wrapped=make_recovery_trainer(train.train,hooks,demo,PINNED_BRAX_TRAIN_SHA256,probe,sink,
                transitions=config['transitions'],run_dir=root,usage_sink=usage,
                demo_coefficient_start=coefficient(offset),demo_coefficient_end=coefficient(offset+config['transitions']),
                demo_batch_size=int(config.get('demo_batch_size',256)),**split)
            wrapped(**kwargs)
        jax.effects_barrier()
        if seen!=[0,config['transitions']]:raise ValueError('missing exact reviewblock checkpoints')
        latest=hooks.latest_receipt();manifest=_read(_path(latest))
        with Path(manifest['state_path']).open('rb') as stream:saved=pickle.load(stream)
        counts={k:int(v) for k,v in adam_update_counts(saved['training_state'].optimizer_state).items()}
        if counts['actor_adam_count']!=counts['critic_adam_count'] or not rows:
            raise ValueError('invalid final recovery learner clocks')
        atomic_json(root/'initialization_evidence.json',usage)
        result=_read(root/f"checkpoint_{config['transitions']:06d}.json")
        status('completed',config['transitions'],wall_seconds=time.monotonic()-started,learner=latest,**counts)
        return result
    except BaseException as error:
        status('failed',max(seen or [0]),error=repr(error),wall_seconds=time.monotonic()-started)
        raise
    finally:
        # Preserve step-zero identities and observed clocks even when a later
        # compiler/update fails; successful final evidence uses the same file.
        if usage:
            atomic_json(root/'initialization_evidence.json',usage)
        writer.close()
