"""GPU short pulses and batched same-context suffixes; CPU delayed PPO updates."""
from pathlib import Path
import json
import time
import numpy as np
from .exploration_loop import read, write
from .probe_bank import load_probe_bank, _file_sha
from .evidence_integrity import canonical_sha256


def suffix_label(valid, failure, timeout, done, horizon_reached):
    from .unified_continuation_labels import classify_first_valid_landing_outcome
    if valid and failure:return None,'simultaneous_landing_failure_unresolved'
    positive,outcome=classify_first_valid_landing_outcome(valid_contact_seen=valid,physical_failure_before_landing=failure,timeout=timeout,done=done,reached_rollout_horizon=horizon_reached)
    return int(positive),outcome


def aggregate_labels(attempts,order):
    if any(a['label']==1 for a in attempts):return 1
    by_policy={a['policy']:a['label'] for a in attempts}
    return 0 if all(name in by_policy and by_policy[name]==0 for name in order) else None


def networks(spec):
    import jax
    import optax
    from brax.training.acme import running_statistics
    from .checkpoint import load_checkpoint
    from .unified_training import checkpoint_identity
    from .unified_formal import build_unified_formal_environment
    from .exploration_network import make_exploration_network_factory, initialize_residual_actor
    from .handoff_bank import pytree_sha256
    bank=load_probe_bank(Path(spec['bank']))
    member=next(m for m in bank['members'] if m['name']==spec['proposer'])
    config,_,env=build_unified_formal_environment(Path(member['policy']['formal_config']))
    payload=load_checkpoint(Path(member['policy']['checkpoint']),expected=checkpoint_identity(config,env))
    for k,v in [('actor_sha256',payload.actor_params),('normalizer_sha256',payload.observation_normalizer),('critic_sha256',payload.critic_params)]:
        if pytree_sha256(v)!=member['policy'][k]:raise ValueError('base payload drift: '+k)
    net=make_exploration_network_factory()({'state':76,'privileged_state':106},4,preprocess_observations_fn=running_statistics.normalize)
    rng=jax.random.PRNGKey(spec['seed']);rng,a,v=jax.random.split(rng,3)
    params=dict(policy=initialize_residual_actor(net,a),value=net.value_network.init(v))
    optimizer=optax.chain(optax.clip_by_global_norm(spec['max_grad_norm']),optax.adam(spec['learning_rate']))
    return env,payload,member,net,optimizer,dict(params=params,optimizer=optimizer.init(params),rng=rng)


def collect(spec, output):
    import jax
    import jax.numpy as jp
    from flax import serialization
    from .ppo import make_checkpoint_policy
    from .exploration_network import compose_residual_action
    from .exploration_continuation import snapshot_arrays, snapshot_from_arrays
    from .unified_envelope_snapshot import save_unified_envelope_snapshot, snapshot_context_sha256, physical_state_sha256
    from .continuation.device_rollout import prepare_parallel_worlds, _shared_warp
    from .iterative_probe_training import first_landing_state
    from .analysis.capability_tube import physical_coordinates_from_arrays, quantize_coordinates, ROOT_GEOMETRY_FIELDS, _cell_id
    if jax.default_backend()!='gpu':raise RuntimeError('GPU pulse collection required')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    env,payload,member,net,opt,state=networks(spec)
    if spec.get('explorer_checkpoint'):
        state=serialization.from_bytes(state,Path(spec['explorer_checkpoint']).read_bytes())
    (output/'behavior.msgpack').write_bytes(serialization.to_bytes(state))
    normalizer=payload.observation_normalizer;params=state['params'];count=spec['num_envs']
    base=make_checkpoint_policy(env,payload,deterministic=True);dist=net.parametric_action_distribution
    reset=jax.vmap(env._reset_jump_start_unified)
    step=jax.vmap(lambda s,a:first_landing_state(env.step(s,a)))
    def run(rng):
        initial=prepare_parallel_worlds(reset(jax.random.split(rng,count)),env,count)
        def advance(carry,_):
            s,key,alive=carry;key,k=jax.random.split(key)
            logits=net.policy_network.apply(normalizer,params['policy'],s.obs)
            raw=dist.sample_no_postprocessing(logits,k);delta=dist.postprocess(raw)
            b=base(s.obs,k)[0]
            action,requested,effective=compose_residual_action(b,delta,jp.asarray(spec['delta_limit']))
            nxt=step(s,action)
            finite=jp.all(jp.isfinite(nxt.data.qpos),axis=-1)&jp.all(jp.isfinite(nxt.data.qvel),axis=-1)
            terminal=nxt.done.astype(bool)|~finite
            tape=dict(observation=s.obs['privileged_state'],raw_action=raw,log_prob=dist.log_prob(logits,raw),value=net.value_network.apply(normalizer,params['value'],s.obs),mask=alive,terminal=terminal,finite=finite,action=action,base_action=b,delta=delta,effective_delta=effective,qpos=nxt.data.qpos,qvel=nxt.data.qvel,phase=nxt.info['active_phase'])
            tape.update({'snap/'+k:v for k,v in snapshot_arrays(nxt).items()})
            def choose(path,n,o):
                return n if _shared_warp(path) else jp.where(alive.reshape((count,)+(1,)*(n.ndim-1)),n,o)
            nxt=jax.tree_util.tree_map_with_path(choose,nxt,s)
            return (nxt,key,alive&~terminal),tape
        (_,key,_),tape=jax.lax.scan(advance,(initial,rng,jp.ones(count,bool)),None,length=spec['pulse_steps'])
        return key,tape
    start=time.monotonic();rng,key=jax.random.split(state['rng'])
    write(output/'status.json',dict(phase='running',charged_interactions=count*spec['pulse_steps']))
    next_rng,tape=jax.device_get(jax.jit(run)(key));np.savez_compressed(output/'prefixes.npz',**tape)
    if np.any(tape['mask']&~tape['finite']):raise ValueError('nonfinite pulse prefix')
    state['rng']=next_rng;(output/'update_state.msgpack').write_bytes(serialization.to_bytes(state))
    generator={**member['policy'],'actor_sha256':canonical_sha256(dict(base=member['policy']['actor_sha256'],residual=_file_sha(output/'behavior.msgpack'),delta=spec['delta_limit'],pulse_steps=spec['pulse_steps'])),'payload_sha256':_file_sha(output/'behavior.msgpack')}
    rows=[]
    for e in range(count):
        t=int(np.flatnonzero(tape['mask'][:,e])[-1])
        arrays={k[5:]:v[t,e] for k,v in tape.items() if k.startswith('snap/')}
        snap=snapshot_from_arrays(arrays,env=env,record=generator,parent_trajectory=str(output/'prefixes.npz')+'::'+str(e),parent_state_sha256=_file_sha(output/'prefixes.npz'))
        path=output/'snapshots'/f'{e:05d}';save_unified_envelope_snapshot(path,snap)
        coords=physical_coordinates_from_arrays(tape['qpos'][t,e],tape['qvel'][t,e],bundle=env._bundle)
        phase='upstream' if snap.active_phase==0 else 'downstream'
        rows.append(dict(index=e,cell=_cell_id(phase,'root_geometry_v1',quantize_coordinates(coords,ROOT_GEOMETRY_FIELDS)),coordinates=coords,phase=phase,snapshot=str(path),state_sha256=physical_state_sha256(snap),snapshot_context_sha256=snapshot_context_sha256(snap),prefix_file=str(output/'prefixes.npz'),prefix_sha256=_file_sha(output/'prefixes.npz'),behavior_sha256=_file_sha(output/'behavior.msgpack'),prefix_terminal=bool(tape['terminal'][t,e]),label=None,learning_attempted=False))
    write(output/'candidates.json',rows)
    write(output/'network_inventory.json',dict(actor_parameters=sum(x.size for x in jax.tree.leaves(params['policy'])),critic_parameters=sum(x.size for x in jax.tree.leaves(params['value'])),base_actor_frozen=True,base_critic_frozen=True,inputs=106,history_frames=3,hidden=[256,256,256],output_actions=4))
    write(output/'hyperparameters.json',spec)
    write(output/'status.json',dict(phase='completed',charged_interactions=count*spec['pulse_steps'],active_interactions=int(tape['mask'].sum()),wall_seconds=time.monotonic()-start))


def evaluate(spec, output):
    """Each policy handles all unresolved candidates in one compiled batch."""
    import jax
    import jax.numpy as jp
    from .exploration_continuation import FrozenSuffixEvaluator
    from .unified_envelope_snapshot import load_unified_envelope_snapshot, snapshot_context_sha256
    from .unified_continuation_labels import fresh_unified_continuation_start, classify_first_valid_landing_outcome
    from .continuation.device_rollout import stack_worlds,prepare_parallel_worlds,_shared_warp
    from .iterative_probe_training import first_landing_state
    if jax.default_backend()!='gpu':raise RuntimeError('GPU suffix evaluation required')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rows=read(spec['candidates']);horizon=spec['horizon'];charged=0;active_count=0
    bank=load_probe_bank(Path(spec['bank']));all_names=[m['name'] for m in bank['members'] if 'evaluator' in m['roles']]
    suffix=FrozenSuffixEvaluator(spec['bank'],all_names,horizon,output/'runtime',spec['budget'])
    for r in rows:r.update(attempts=[],label=None,witness=None)
    if spec.get('reuse_results'):
        reused=read(spec['reuse_results'])
        if len(reused)!=len(rows):raise ValueError('reused candidate count differs')
        for row,old in zip(rows,reused):
            if row['snapshot_context_sha256']!=old['snapshot_context_sha256']:raise ValueError('reused context differs')
            for a in old['attempts']:
                if a['actor_sha256']!=suffix.members[a['policy']]['policy']['actor_sha256'] or _file_sha(Path(a['trace']))!=a['trace_sha256']:raise ValueError('reused evaluator/trace drift')
            row.update(attempts=old['attempts'],label=old['label'],witness=old['witness'])
    start=time.monotonic()
    for name in spec['order']:
        subset=[r for r in rows if (spec.get('full_matrix',False) or r['label']!=1) and not r.get('prefix_terminal',False) and not any(a['policy']==name for a in r['attempts'])]
        if not subset:continue
        env,policy,_=suffix._runtime(name)
        restored=[]
        for r in subset:
            snap=load_unified_envelope_snapshot(Path(r['snapshot']))
            if snapshot_context_sha256(snap)!=r['snapshot_context_sha256']:raise ValueError('suffix snapshot identity changed')
            restored.append(fresh_unified_continuation_start(snap,env))
        count=len(restored);initial=prepare_parallel_worlds(stack_worlds(restored),env,count)
        if charged+count*horizon>spec['budget']:raise RuntimeError('insufficient declared suffix reservation')
        write(output/'status.json',dict(phase='running',policy=name,charged_interactions=charged,reserved_attempt_interactions=count*horizon))
        step=jax.vmap(lambda s,a:first_landing_state(env.step(s,a)))
        def rollout(initial):
            def frame(s,action,mask):
                return dict(qpos=s.data.qpos,qvel=s.data.qvel,action=action,reward=s.reward,mask=mask,done=s.done,success=s.info['success'],physical_failure=s.info['physical_failure'],timeout=s.info['timeout'],end_code=s.info['end_code'],valid_contact=s.info['down_events'].valid_contact_seen)
            blank=frame(initial,jp.zeros((count,4)),jp.zeros(count,bool))
            traces={k:jp.zeros((horizon,)+v.shape,v.dtype) for k,v in blank.items()}
            def condition(c):return (c[0]<horizon)&jp.any(c[2])
            def advance(c):
                t,s,alive,tr=c
                keys=jax.random.split(jax.random.fold_in(jax.random.PRNGKey(0),t),count)
                action=jax.vmap(policy)(s.obs,keys)[0]
                nxt=step(s,jp.where(alive[:,None],action,0))
                finite=jp.all(jp.isfinite(nxt.data.qpos),axis=-1)&jp.all(jp.isfinite(nxt.data.qvel),axis=-1)
                f=frame(nxt,action,alive);tr={k:v.at[t].set(f[k]) for k,v in tr.items()}
                def choose(path,n,o):return n if _shared_warp(path) else jp.where(alive.reshape((count,)+(1,)*(n.ndim-1)),n,o)
                nxt=jax.tree_util.tree_map_with_path(choose,nxt,s)
                alive=alive&~nxt.done.astype(bool)&~nxt.info['down_events'].valid_contact_seen&finite
                return t+1,nxt,alive,tr
            tick,final,_,tr=jax.lax.while_loop(condition,advance,(jp.array(0),initial,jp.ones(count,bool),traces))
            return tick,tr
        tick,tape=jax.device_get(jax.jit(rollout)(initial));tick=int(tick);tape={k:v[:tick] for k,v in tape.items()}
        cost=count*tick;charged+=cost;active_count+=int(tape['mask'].sum())
        trace_path=output/(name+'_traces.npz');np.savez_compressed(trace_path,**tape)
        for e,r in enumerate(subset):
            indices=np.flatnonzero(tape['mask'][:,e]);last=int(indices[-1]);valid=bool(tape['valid_contact'][last,e]);failure=bool(tape['physical_failure'][last,e])
            if not np.isfinite(tape['qpos'][indices,e]).all() or not np.isfinite(tape['qvel'][indices,e]).all():raise ValueError('nonfinite suffix')
            label,outcome=suffix_label(valid,failure,bool(tape['timeout'][last,e]),bool(tape['done'][last,e]),len(indices)>=horizon)
            r['attempts'].append(dict(policy=name,actor_sha256=suffix.members[name]['policy']['actor_sha256'],label=label,outcome=outcome,steps=len(indices),task_return=float(tape['reward'][indices,e].sum()),trace=str(trace_path),trace_lane=e,trace_sha256=_file_sha(trace_path),snapshot_context_sha256=r['snapshot_context_sha256']))
            if label:r.update(label=1,witness=name)
        del initial,restored
        jax.clear_caches()
    for r in rows:
        if r.get('prefix_terminal'):r.update(label=0,terminal_reason='pulse_terminated_before_handoff')
        elif r['label']!=1:r['label']=aggregate_labels(r['attempts'],spec['order'])
    write(output/'results.json',rows)
    write(output/'status.json',dict(phase='completed',charged_interactions=charged,active_interactions=active_count,padding_interactions=charged-active_count,successes=sum(r['label']==1 for r in rows),wall_seconds=time.monotonic()-start))


def update(spec, output):
    import jax
    import jax.numpy as jp
    import optax
    from flax import serialization
    from .exploration_training import episode_advantages
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    env,payload,member,net,opt,state=networks(spec)
    source=Path(spec['collection']);state=serialization.from_bytes(state,(source/'update_state.msgpack').read_bytes())
    tape=np.load(source/'prefixes.npz');feedback=read(spec['feedback'])
    eligible=np.array(feedback['eligible'],bool);mask=tape['mask'].astype(bool)&eligible[None,:]
    reward=np.zeros(mask.shape,np.float32)
    for e in range(mask.shape[1]):
        ix=np.flatnonzero(mask[:,e])
        if len(ix):reward[ix[-1],e]=feedback['rewards'][e]
    adv,ret=episode_advantages(reward,tape['value'],mask,gamma=1.,lam=1.)
    if not mask.any():
        (output/'state.msgpack').write_bytes(serialization.to_bytes(state))
        write(output/'optimizer_updates.json',[]);write(output/'hyperparameters.json',spec)
        write(output/'metrics.json',dict(optimizer_updates=0,post_update_kl=0.,post_update_clip_fraction=0.,value_explained_variance=None,total_loss=0.,actor_loss=0.,critic_loss=0.,entropy=0.,gradient_norm=0.,reward_components=feedback['component_sums'],reward=0.,eligible_episodes=0,effective_training_samples=0,update_skipped=True))
        write(output/'status.json',dict(phase='completed',charged_interactions=0));return
    a=adv[mask];a=(a-a.mean())/(a.std()+1e-8)
    batch=dict(obs=tape['observation'][mask],raw=tape['raw_action'][mask],old=tape['log_prob'][mask],adv=a,target=ret[mask])
    dist=net.parametric_action_distribution;normalizer=payload.observation_normalizer
    @jax.jit
    def learn(params,optstate,b,key):
        def loss(p):
            obs={'privileged_state':b['obs']};logits=net.policy_network.apply(normalizer,p['policy'],obs)
            lr=dist.log_prob(logits,b['raw'])-b['old'];ratio=jp.exp(lr);w=b['weight'];den=jp.maximum(w.sum(),1)
            policy=-jp.sum(w*jp.minimum(ratio*b['adv'],jp.clip(ratio,1-spec['clip'],1+spec['clip'])*b['adv']))/den
            value=net.value_network.apply(normalizer,p['value'],obs);vl=.5*jp.sum(w*(value-b['target'])**2)/den
            ent=jp.sum(w*dist.entropy(logits,key))/den
            return policy+spec['value_coefficient']*vl-spec['entropy_coefficient']*ent,jp.array([policy,vl,ent])
        (total,parts),grads=jax.value_and_grad(loss,has_aux=True)(params);delta,optstate=opt.update(grads,optstate,params)
        return optax.apply_updates(params,delta),optstate,jp.concatenate([jp.array([total]),parts,jp.array([optax.global_norm(grads)])])
    params=state['params'];optstate=state['optimizer'];rng=state['rng'];size=len(a);mb=spec['minibatch_size'];logs=[]
    host=np.random.default_rng(spec['seed']+spec['round_index'])
    for epoch in range(spec['epochs']):
        order=host.permutation(size)
        for lo in range(0,size,mb):
            ix=order[lo:lo+mb];n=len(ix);ix=np.pad(ix,(0,mb-n),mode='wrap');mini={k:jp.asarray(v[ix]) for k,v in batch.items()};mini['weight']=jp.asarray(np.arange(mb)<n,dtype=jp.float32)
            rng,key=jax.random.split(rng);params,optstate,values=learn(params,optstate,mini,key)
            values=np.asarray(values)
            if not np.isfinite(values).all():raise ValueError('nonfinite explorer update')
            logs.append(dict(epoch=epoch,valid_samples=n,**dict(zip(['total_loss','actor_loss','critic_loss','entropy','gradient_norm'],map(float,values)))))
        logits=net.policy_network.apply(normalizer,params['policy'],{'privileged_state':jp.asarray(batch['obs'])})
        lr=np.asarray(dist.log_prob(logits,jp.asarray(batch['raw'])))-batch['old']
        kl=float(np.mean(np.expm1(lr)-lr))
        if not np.isfinite(kl):raise ValueError('nonfinite post-update KL')
        if kl>spec['target_kl']:break
    values=np.asarray(net.value_network.apply(normalizer,params['value'],{'privileged_state':jp.asarray(batch['obs'])}))
    variance=float(np.var(batch['target']));explained=float(1-np.var(batch['target']-values)/variance) if variance>1e-12 else None
    (output/'state.msgpack').write_bytes(serialization.to_bytes(dict(params=params,optimizer=optstate,rng=rng)))
    np.savez_compressed(output/'learning.npz',reward=reward,advantages=adv,returns=ret,mask=mask)
    write(output/'optimizer_updates.json',logs)
    write(output/'hyperparameters.json',spec)
    write(output/'metrics.json',dict(optimizer_updates=len(logs),post_update_kl=kl,post_update_clip_fraction=float(np.mean(np.abs(np.expm1(lr))>spec['clip'])),value_explained_variance=explained,**{k:float(np.mean([r[k] for r in logs])) for k in ['total_loss','actor_loss','critic_loss','entropy','gradient_norm']},reward_components=feedback['component_sums'],reward=float(np.sum(feedback['rewards'])),eligible_episodes=int(eligible.sum()),effective_training_samples=size))
    write(output/'status.json',dict(phase='completed',charged_interactions=0))


def baseline(spec,output):
    import jax
    from .exploration_continuation import snapshot_arrays,snapshot_from_arrays
    from .unified_envelope_snapshot import save_unified_envelope_snapshot,snapshot_context_sha256,physical_state_sha256
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    env,payload,member,net,opt,state=networks(spec)
    start=jax.jit(env._reset_jump_start_unified)(jax.random.PRNGKey(spec['seed']))
    snap=snapshot_from_arrays(jax.device_get(snapshot_arrays(start)),env=env,record=member['policy'],parent_trajectory='fixed_jump_start_zero_actions',parent_state_sha256=spec['jump_start_state_sha256'])
    if physical_state_sha256(snap)!=spec['jump_start_state_sha256']:raise ValueError('baseline fixed start drift')
    path=output/'snapshot';save_unified_envelope_snapshot(path,snap)
    rows=[dict(index=0,snapshot=str(path),snapshot_context_sha256=snapshot_context_sha256(snap),prefix_terminal=False)]
    write(output/'candidates.json',rows)
    evaluate({**spec,'candidates':str(output/'candidates.json'),'budget':len(spec['order'])*spec['horizon'],'full_matrix':True},output/'evaluation')
    result=read(output/'evaluation/results.json')[0]
    source=next(a for a in result['attempts'] if a['policy']==spec['proposer'])
    if source['label']!=1:raise ValueError('source pi cannot complete fixed-start baseline')
    write(output/'status.json',read(output/'evaluation/status.json'))
