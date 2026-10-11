"""Worker-only physics/PPO for T1. Importing this module launches nothing."""
from pathlib import Path
import dataclasses
import json
import pickle
import time

import numpy as np

from .retention_repair import read, write, sha


def compose_trainer(trainer, hooks, contract, demo, keep, reference, sink):
    """Keep upstream PPO/Adam math, adding only locked auxiliaries and telemetry."""
    from .generative_bridge.learner_continuation import instrument_trainer
    from .generative_bridge.actual_update import instrument_updates
    from .generative_bridge.student import make_joint_student_trainer
    wrapped = instrument_trainer(trainer,hooks.initialize,hooks.save)
    wrapped = instrument_updates(wrapped,sink,max_grad_norm=contract['ppo']['max_grad_norm'],
                                 probe=keep[0][:32],record_optimizer_count=True)
    return make_joint_student_trainer(wrapped,demo,retention=keep,retention_reference=reference,
        transitions=contract['demo_schedule_transitions'],
        demo_coefficient_start=contract['demo_coefficient_start'],
        demo_coefficient_end=contract['demo_coefficient_end'],
        keep_coefficient=contract['keep_coefficient'],demo_batch_size=contract['demo_batch_size'],
        retention_batch_size=contract['retention_batch_size'],freeze_actor_normalizer=True,
        demo_clock='completed_transitions',actual_gradient_audit=True,record_update_clock=True,
        audit_enabled=True,max_first_behavior_kl=contract['first_behavior_KL_limit'],
        audit_callback_enabled=False)


def _environment(plan):
    import jax.numpy as jp
    from .generative_bridge.worker import source_payload
    from .unified_formal import build_unified_formal_environment
    from .retention_reset import FourPoolReset
    policy,payload = source_payload(plan['models']['pi0']['frozen_policy'])
    cfg,artifact,env = build_unified_formal_environment(Path(policy['formal_config']))
    env._training_action_pulse = None
    adapter = FourPoolReset(env,read(plan['reset_inventory']))
    # Reuse the existing per-physics-substep capacity sentinel, without resizing
    # scratch or changing physics; expose terminal overflow in metrics before reset.
    if hasattr(getattr(adapter.pool.data,'_impl',None),'naconmax'):
        zeros=jp.zeros(adapter.pool.obs['state'].shape[0],bool)
        adapter.pool=adapter.pool.replace(info={**adapter.pool.info,'parallel_capacity_exceeded':zeros},
            metrics={**adapter.pool.metrics,'T1/capacity_exceeded':zeros.astype(jp.float32)})
        original=adapter.original_step
        def checked(state,action):
            result=original(state,action)
            return result.replace(metrics={**result.metrics,'T1/capacity_exceeded':result.info['parallel_capacity_exceeded'].astype(jp.float32)})
        adapter.original_step=checked
    adapter.install()
    return policy,payload,cfg,env,adapter


def capacity_receipt(env,adapter):
    impl=getattr(adapter.pool.data,'_impl',None)
    field=lambda name:int(getattr(impl,name)) if hasattr(impl,name) else None
    return dict(naconmax=field('naconmax'),naccdmax=field('naccdmax'),
        njmax=field('njmax'),source_config=dict(env.resolved_config.model),
        check='every physics substep via existing checked_physics_step; callback rejects any saturation',
        scratch_resized=False)


def wrap_t1_for_training(env,episode_length=400,action_repeat=1,randomization_fn=None):
    import jax
    import jax.numpy as jp
    from mujoco_playground._src import wrapper
    from .ppo import wrap_for_jit_training
    class CapacityGuard(wrapper.Wrapper):
        def step(self,state,action):
            result=self.env.step(state,action)
            def check(value):
                if bool(value):raise FloatingPointError('T1 collision/constraint scratch saturation')
            jax.debug.callback(check,jp.any(result.metrics.get('T1/capacity_exceeded',jp.asarray(False))))
            return result
    return CapacityGuard(wrap_for_jit_training(env,episode_length,action_repeat,randomization_fn))


def _tree_finite(tree):
    import jax
    return all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(tree))


def _save_inference(directory, step, params, source_payload, contract, learner):
    import jax
    from .checkpoint import save_checkpoint
    from .handoff_bank import pytree_sha256
    from .generative_bridge.contracts import digest
    params = jax.device_get(params)
    path = directory/f'inference_{step:06d}.pkl'
    with path.open('xb') as stream:
        pickle.dump(params,stream,pickle.HIGHEST_PROTOCOL)
    checkpoint = directory/'checkpoints'/f'transition_{step}'
    # CheckpointIdentity names the unchanged physical/observation/action contract.
    # The separately hashed T1 learner contract names reset/loss/optimizer/seed.
    save_checkpoint(checkpoint,dataclasses.replace(source_payload,training_transitions=int(step),
        observation_normalizer=params[0],actor_params=params[1],critic_params=params[2]))
    receipt = dict(schema='jit_retention_T1_checkpoint_v1',transition=int(step),
        inference=str(path),inference_sha256=sha(path),checkpoint=str(checkpoint),
        checkpoint_payload_sha256=sha(checkpoint/'payload.pkl'),
        actor_sha256=pytree_sha256(params[1]),critic_sha256=pytree_sha256(params[2]),
        normalizer_sha256=pytree_sha256(params[0]),training_contract_sha256=digest(contract),
        checkpoint_identity_scope='unchanged source environment/observation/action identity; separate T1 training contract',
        learner=learner,source_frozen_policy=contract['source_frozen_policy'])
    write(checkpoint/'checkpoint_receipt.json',receipt)
    write(directory/f'checkpoint_{step:06d}.json',receipt)
    return receipt


def verify_saved_training(directory, policy, expected_transitions, expected_updates):
    """Audit serialized full state, independently of the training callback claims."""
    import jax
    from .handoff_bank import pytree_sha256
    from .generative_bridge.actual_update import adam_update_count
    from .generative_bridge.learner_continuation import LearnerHooks
    from .retention_t1 import ordered_updates,validate_actual_update
    directory = Path(directory)
    latest = read(directory/'learner/latest.json')
    if sha(latest['path']) != latest['sha256']:
        raise ValueError('learner manifest drift')
    manifest = read(latest['path'])
    if sha(manifest['state_path']) != manifest['state_sha256']:
        raise ValueError('learner serialized state drift')
    with open(manifest['state_path'],'rb') as stream:
        saved = pickle.load(stream)
    state = saved['training_state']
    rows = ordered_updates(read(directory/'actual_updates.json'))
    contract = read(directory/'training_contract.json')
    for row in rows:
        validate_actual_update(row,contract)
    if not rows or len(rows) != expected_updates:
        raise ValueError('missing actual accepted optimizer telemetry')
    counts = [int(r['actual/optimizer_updates']) for r in rows]
    if counts != list(range(1,expected_updates+1)):
        raise ValueError('actual Adam update clock gap or duplicate')
    checks = dict(normalizer_frozen=manifest['normalizer_sha256'] == policy['normalizer_sha256'],
        full_state_finite=_tree_finite((state,saved['rng'])),
        transition_clock=int(state.env_steps) == expected_transitions == manifest['local_transitions'],
        optimizer_clock=int(adam_update_count(state.optimizer_state)) == expected_updates,
        loss_clock=max(r['actual/completed_training_transitions'] for r in rows) == expected_transitions,
        actor_changed=manifest['actor_sha256'] != policy['actor_sha256'],
        critic_changed=manifest['critic_sha256'] != policy['critic_sha256'],
        actual_actor_delta_positive=any(r['actual/actor_adam_delta_norm'] > 0 for r in rows),
        actual_critic_delta_positive=any(r['actual/critic_adam_delta_norm'] > 0 for r in rows),
        first_loss_passed=read(directory/'first_loss_gate.json')['status'] == 'passed')
    for step in (0,expected_transitions):
        node = read(directory/f'checkpoint_{step:06d}.json')
        if sha(node['inference']) != node['inference_sha256']:
            raise ValueError('inference checkpoint drift')
        with open(node['inference'],'rb') as stream:
            params = pickle.load(stream)
        if step == 0:
            checks['pi0_actor_at_zero'] = pytree_sha256(params[1]) == policy['actor_sha256']
            checks['pi0_critic_at_zero'] = pytree_sha256(params[2]) == policy['critic_sha256']
            zero_manifest = read(node['learner']['path'])
            with open(zero_manifest['state_path'],'rb') as stream:
                zero = pickle.load(stream)['training_state']
            checks['fresh_Adam_at_zero'] = all(np.all(np.asarray(x) == 0) for x in jax.tree.leaves(zero.optimizer_state))
        else:
            checks['inference_matches_learner'] = pytree_sha256(params) == pytree_sha256((state.normalizer_params,state.params.policy,state.params.value))
    # Restore through the production hook, without taking another physics step.
    template = state.replace(optimizer_state=jax.tree.map(np.zeros_like,state.optimizer_state))
    hooks = LearnerHooks(directory/'restore_verification',contract=manifest['contract_sha256'],
        trainer_sha=manifest['trainer_sha256'],parent=latest,preserve_stage_clock=True)
    restored,rng = hooks.initialize(template,jax.random.PRNGKey(1))
    checks['full_state_restore_exact'] = pytree_sha256((restored,rng)) == pytree_sha256((state,saved['rng']))
    result = dict(phase='passed' if all(checks.values()) else 'failed',checks=checks,
                  actual_optimizer_updates=expected_updates,completed_training_transitions=expected_transitions,
                  learner=latest,first_update=rows[0],last_update=rows[-1],
                  gradient_sum_max_abs_error=max(r['actual/actor_gradient_norm_sum_difference'] for r in rows),
                  claim='serialized learner engineering checks, not recovery improvement')
    write(directory/'verification.json',result)
    if not all(checks.values()):
        raise ValueError('full PPO learner verification failed')
    return result


def train_worker(plan, arm, output, *, engineering=False):
    import jax
    if jax.default_backend() != 'gpu':
        raise ValueError('T1 physics worker requires real GPU')
    from torch.utils.tensorboard import SummaryWriter
    from brax.training.agents.ppo import train
    from .generative_bridge.student import load_retention_traces
    from .generative_bridge.learner_continuation import LearnerHooks,trainer_sha
    from .generative_bridge.contracts import digest
    from .handoff_bank import pytree_sha256
    from .ppo_numerics import guard_ppo_updates
    from .unified_formal import build_unified_formal_trainer_kwargs
    from .retention_t1 import ordered_updates,validate_actual_update
    directory = Path(output); directory.mkdir(parents=True,exist_ok=False)
    started = time.monotonic(); contract = plan['contracts'][arm]
    policy,payload,cfg,env,adapter = _environment(plan)
    write(directory/'physics_capacity.json',capacity_receipt(env,adapter))
    keep = load_retention_traces(read(plan['keep_receipt']))
    demo = read(plan['demo_manifest']) if arm == 'PPO_demo_keep' else None
    transitions = 3200 if engineering else 32000
    expected_steps = [0,transitions] if engineering else [0,16000,32000]
    hooks = LearnerHooks(directory/'learner',contract=digest(contract),trainer_sha=trainer_sha(train.train),
                        parent=None,preserve_stage_clock=True)
    writer = SummaryWriter(str(Path(plan['output'])/'tensorboard'/('engineering_'+arm if engineering else arm)))
    import threading
    status_lock=threading.RLock()
    rows = []; seen = []
    write(directory/'training_contract.json',contract)
    def status(phase,step,**extra):
        with status_lock:
            write(directory/'status.json',dict(phase=phase,arm=arm,engineering=engineering,
                planned_training_transitions=transitions,completed_training_transitions=int(step),
                charged_interactions=transitions if phase != 'completed' else int(step),
                actual_optimizer_updates=max((int(r['actual/optimizer_updates']) for r in rows),default=0),
                E_updates=0,G_updates=0,updated_unix=time.time(),**extra))
    status('initializing',0)
    def sink(row):
        if int(row['actual/optimizer_updates']) == 1:
            first_loss_gate(row,contract,directory/'first_loss_gate.json')
        validate_actual_update(row,contract)
        rows.append(row)
        with (directory/'actual_updates.jsonl').open('a') as stream:
            stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
        update = int(row['actual/optimizer_updates']); clock = int(row['actual/completed_training_transitions'])
        for key,value in row.items():
            writer.add_scalar(key,value,update)
        writer.add_scalar('training/reward_per_transition',row['actual/reward_per_transition'],clock)
        writer.add_scalar('training/completed_transitions',clock,update)
        writer.flush()
        status('running',max(int(r['actual/completed_training_transitions']) for r in rows))
    class Callbacks:
        def on_progress(self,step,metrics):
            for key,value in metrics.items():
                if np.ndim(value) == 0 and np.isfinite(float(value)):
                    writer.add_scalar('Brax/'+key,float(value),int(step))
            writer.flush()
        def on_policy_params(self,step,make_policy,params):
            jax.effects_barrier(); step = int(step)
            if step not in expected_steps or step in seen:
                raise ValueError('unexpected or duplicate PPO checkpoint clock')
            if pytree_sha256(params[0]) != policy['normalizer_sha256'] or not _tree_finite(params):
                raise ValueError('full normalizer drift or nonfinite checkpoint')
            if step == 0 and (pytree_sha256(params[1]) != policy['actor_sha256'] or pytree_sha256(params[2]) != policy['critic_sha256']):
                raise ValueError('PPO did not restore original pi0 Actor and critic')
            _save_inference(directory,step,params,payload,contract,hooks.latest_receipt())
            seen.append(step); write(directory/'actual_updates.json',ordered_updates(rows)); status('running',step)
    callbacks = Callbacks()
    wrapped = compose_trainer(train.train,hooks,contract,demo,keep,
                              (payload.observation_normalizer,payload.actor_params),sink)
    kwargs = build_unified_formal_trainer_kwargs(cfg,env,callbacks)
    kwargs['wrap_env_fn']=wrap_t1_for_training
    rename = {'requested_transitions':'num_timesteps','num_parallel_envs':'num_envs','episode_horizon':'episode_length'}
    kwargs.update({rename.get(k,k):v for k,v in contract['ppo'].items()})
    kwargs.update(num_timesteps=transitions,num_evals=2 if engineering else 3,
        seed=contract['seed'],restore_params=(payload.observation_normalizer,payload.actor_params,payload.critic_params),
        restore_value_fn=True,run_evals=False,log_training_metrics=True,training_metrics_steps=3200)
    # No adaptive LR or inherited action pulse may silently change this pilot.
    for key in ('learning_rate_schedule','desired_kl','min_learning_rate','max_learning_rate',
                'learning_rate_schedule_min_lr','learning_rate_schedule_max_lr'):
        kwargs.pop(key,None)
    try:
        with guard_ppo_updates(directory):
            wrapped(**kwargs)
        jax.effects_barrier()
        if seen != expected_steps:
            raise ValueError('missing fixed endpoint checkpoint')
        write(directory/'actual_updates.json',ordered_updates(rows))
        verified = verify_saved_training(directory,policy,transitions,transitions//3200*8)
        status('completed',transitions,wall_seconds=time.monotonic()-started,
               learner=verified['learner'],reset_inventory=plan['reset_inventory'],
               normalizer_frozen=True,critic_trainable=True,optimizer='fresh_once',
               selection='fixed32000' if not engineering else 'discard_engineering_state')
    except BaseException as error:
        write(directory/'actual_updates.json',ordered_updates(rows))
        status('failed',max(seen or [0]),error=repr(error),wall_seconds=time.monotonic()-started)
        raise
    finally:
        writer.close()


def first_loss_gate(row,contract,path):
    """Observe loss associated with real accepted Adam update1, never arrival order."""
    if int(row['actual/optimizer_updates']) != 1:
        return None
    finite = all(np.isfinite(v) for v in row.values())
    excess = row['audit/installed_behavior_kl']-row['audit/installed_self_kl']
    passed = finite and excess <= contract['first_behavior_KL_limit']
    result = dict(schema='jit_retention_T1_first_loss_v1',status='passed' if passed else 'diagnostic_stop',
        actual_optimizer_updates=1,maximum_excess_behavior_kl=contract['first_behavior_KL_limit'],
        excess_behavior_kl=float(excess) if np.isfinite(excess) else None,
        finite=bool(finite),metrics={k:float(v) if np.isfinite(v) else None for k,v in row.items()},
        source='generative_bridge/campaign.py v1.2 max_first_behavior_kl=0.05',
        accepted_optimizer_updates='asynchronous callback stop is not a zero-update guarantee',
        installed_normalizer_timing='entire original pi0 normalizer frozen before every SGD')
    write(path,result)
    if not passed:raise FloatingPointError('T1 actual first-loss numerical contract failed')
    return result


def reset_worker(plan,output):
    """200 charged steps, including a real training-wrapper terminal reset."""
    import jax
    import jax.numpy as jp
    if jax.default_backend() != 'gpu':raise ValueError('real GPU reset check required')
    from .ppo import wrap_for_jit_training,make_checkpoint_policy
    from .continuation.device_rollout import stack_worlds,take_world
    directory=Path(output);directory.mkdir(parents=True,exist_ok=False)
    _,payload,_,env,adapter=_environment(plan)
    write(directory/'physics_capacity.json',capacity_receipt(env,adapter))
    key=jax.random.PRNGKey(plan['seed']);keys=jax.random.split(key,8)
    pool=stack_worlds([adapter.reset_group(keys[i],jp.asarray(i//2,jp.int32)) for i in range(8)])
    wrapped=wrap_t1_for_training(env,episode_length=400)
    administrative=jax.jit(wrapped.reset)(keys)
    initial=administrative.replace(data=pool.data,obs=pool.obs,reward=pool.reward,done=pool.done,
        metrics=pool.metrics,info={**administrative.info,**pool.info,'steps':jp.full(8,399.)})
    policy=make_checkpoint_policy(env,payload,deterministic=True)
    def step(state,t):
        action=jax.vmap(policy)(state.obs,jax.random.split(jax.random.fold_in(key,t),8))[0]
        next_state=wrapped.step(state,action)
        return next_state,dict(qpos=next_state.data.qpos,qvel=next_state.data.qvel,
            observation=next_state.obs['state'],time=next_state.data.time,group=next_state.info['retention_reset_group'],
            index=next_state.info['retention_reset_index'],done=next_state.done,reward=next_state.reward,
            finite=jp.all(jp.isfinite(next_state.data.qpos),axis=-1)&jp.all(jp.isfinite(next_state.obs['state']),axis=-1))
    first,_=jax.jit(step)(initial,jp.asarray(0))
    if not np.all(np.asarray(first.done)==1):raise ValueError('forced engineering wrapper timeout failed')
    # Verify all original task state, FIFO, events and snapshot RNG after terminal reset.
    for lane in range(8):
        actual=take_world(first,lane);saved=take_world(adapter.pool,int(first.info['retention_reset_index'][lane]))
        for a,b in zip(jax.tree.leaves((actual.obs,actual.data.qpos,actual.data.qvel,actual.data.time)),
                       jax.tree.leaves((saved.obs,saved.data.qpos,saved.data.qvel,saved.data.time))):
            np.testing.assert_array_equal(a,b)
        for name,value in saved.info.items():
            if name=='rng' and int(actual.info['retention_reset_group'])<2:continue
            if name=='rng':
                np.testing.assert_array_equal(jax.random.key_data(actual.info[name]),jax.random.key_data(value));continue
            for a,b in zip(jax.tree.leaves(actual.info[name]),jax.tree.leaves(value)):
                np.testing.assert_array_equal(a,b)
    # First transition is counted exactly once; continue24 more, without another reset pass.
    _,tail=jax.device_get(jax.jit(lambda state:jax.lax.scan(step,state,jp.arange(1,25)))(first))
    _,head=jax.device_get((first,dict(qpos=first.data.qpos,qvel=first.data.qvel,observation=first.obs['state'],
        time=first.data.time,group=first.info['retention_reset_group'],index=first.info['retention_reset_index'],
        done=first.done,reward=first.reward,finite=jp.all(jp.isfinite(first.obs['state']),axis=-1))))
    tape={k:np.concatenate([np.asarray(head[k])[None],np.asarray(tail[k])]) for k in head}
    if not tape['finite'].all():raise ValueError('nonfinite reset GPU engineering trace')
    np.savez_compressed(directory/'reset_trace.npz',**tape)
    write(directory/'verification.json',dict(phase='passed',charged_interactions=200,
        actual_training_wrapper=True,initial_forced_groups=[0,0,1,1,2,2,3,3],
        first_transition_forced_wrapper_timeout=True,full_task_context_after_reset_exact=True,
        snapshot_rng_preserved=True,trace_sha256=sha(directory/'reset_trace.npz'),
        claim='engineering reset check; not policy recovery evidence'))
    write(directory/'status.json',dict(phase='completed',charged_interactions=200,training_updates=0))


def panel_worker(plan,arm,output):
    import jax
    import jax.numpy as jp
    if jax.default_backend() != 'gpu':raise ValueError('real GPU TRAIN panel required')
    from .ppo import make_checkpoint_policy
    from .continuation.device_rollout import stack_worlds,prepare_parallel_worlds
    from .pulse_exploration_runtime import freeze_inactive_worlds
    directory=Path(output);directory.mkdir(parents=True,exist_ok=False)
    _,payload,_,env,adapter=_environment(plan)
    write(directory/'physics_capacity.json',capacity_receipt(env,adapter))
    node=read(Path(plan['output'])/'training'/arm/'checkpoint_032000.json')
    if sha(node['inference'])!=node['inference_sha256']:raise ValueError('panel checkpoint drift')
    with open(node['inference'],'rb') as stream:params=pickle.load(stream)
    payload=dataclasses.replace(payload,observation_normalizer=params[0],actor_params=params[1],critic_params=params[2])
    key=jax.random.PRNGKey(plan['seed']);keys=jax.random.split(key,4)
    initial=prepare_parallel_worlds(stack_worlds([adapter.reset_group(keys[i],jp.asarray(i,jp.int32)) for i in range(4)]),env,4)
    policy=make_checkpoint_policy(env,payload,deterministic=True)
    def step(carry,t):
        state,alive=carry;action=jax.vmap(policy)(state.obs,jax.random.split(jax.random.fold_in(key,t),4))[0]
        advanced=jax.vmap(env.step)(state,jp.where(alive[:,None],action,0))
        advanced=freeze_inactive_worlds(advanced,state,alive)
        record=dict(qpos=advanced.data.qpos,qvel=advanced.data.qvel,observation=advanced.obs['state'],
            reward=advanced.reward,action=action,active=alive,time=advanced.data.time,done=advanced.done,
            finite=jp.all(jp.isfinite(advanced.data.qpos),axis=-1)&jp.all(jp.isfinite(advanced.data.qvel),axis=-1)&jp.all(jp.isfinite(advanced.obs['state']),axis=-1),
            capacity_exceeded=advanced.info.get('parallel_capacity_exceeded',jp.zeros(4,bool)))
        return (advanced,alive&~advanced.done.astype(bool)),record
    _,tape=jax.device_get(jax.jit(lambda s:jax.lax.scan(step,(s,jp.ones(4,bool)),jp.arange(400)))(initial))
    if not tape['finite'][tape['active']].all() or tape['capacity_exceeded'].any():raise ValueError('nonfinite/overflow TRAIN panel')
    np.savez_compressed(directory/'trace.npz',**tape)
    write(directory/'status.json',dict(phase='completed',charged_interactions=1600,
        active_interactions=int(tape['active'].sum()),padding_interactions=int(1600-tape['active'].sum()),
        checkpoint=node,role='TRAIN',claim='four reset-pool diagnostic panels, excluded from evaluation success counts'))


def worker(request_path):
    """Workers cannot bypass the coordinator's locked one-shot dispatch reservation."""
    from .retention_t1 import audit
    request_path=Path(request_path).resolve();request=read(request_path);plan_path=Path(request['plan'])
    audit(plan_path);plan=read(plan_path);root=Path(plan['output'])
    claim=read(root/'execution.claim')
    dispatch=read(root/'active_dispatch.json')
    if (request['plan_sha256']!=sha(plan_path) or claim['plan_sha256']!=sha(plan_path) or
        dispatch['request']!=str(request_path) or dispatch['request_sha256']!=sha(request_path) or
        dispatch['output']!=request['output'] or dispatch['phase']!='running'):
        raise ValueError('T1 worker lacks matching frozen execution claim and reservation')
    output=Path(request['output']).resolve()
    if not output.is_relative_to(root) or output.exists():raise ValueError('fresh run-owned worker output required')
    kind=request['kind']
    if kind=='reset':return reset_worker(plan,output)
    if kind in ('engineering','train'):
        return train_worker(plan,request['arm'],output,engineering=kind=='engineering')
    if kind=='panel':return panel_worker(plan,request['arm'],output)
    raise ValueError('unknown T1 worker kind')
