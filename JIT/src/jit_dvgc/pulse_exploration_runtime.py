"""GPU short pulses and batched same-context suffixes; CPU delayed PPO updates."""
from pathlib import Path
import json
import time
import numpy as np
from .exploration_loop import read, write
from .probe_bank import load_probe_bank, _file_sha
from .evidence_integrity import canonical_sha256
from .pulse_schedule import (controller_mode, selected_event, event_ready,
    pulse_activity, descent_clearance, lane_onsets, collection_steps,
    initial_velocity_randomization, sample_initial_velocity_noise)


def freeze_inactive_worlds(following, previous, alive):
    """Freeze complete per-world state while retaining shared Warp allocator leaves."""
    import jax
    import jax.numpy as jp
    from .continuation.device_rollout import _shared_warp
    count=alive.shape[0]
    def choose(path,new,old):
        return new if _shared_warp(path) else jp.where(alive.reshape((count,)+(1,)*(new.ndim-1)),new,old)
    return jax.tree_util.tree_map_with_path(choose,following,previous)


def scheduled_pulse_summary(spec,tape,rows):
    """Small receipt over all declared episodes; raw tapes remain authoritative."""
    from .generative_bridge.pulse_protocol import normalize_pulse_contract
    from .pulse_schedule import collection_budget
    contract=normalize_pulse_contract(spec)
    if contract is None:return {}
    delays=lane_onsets(spec,spec['round_index']);groups={}
    mask=np.asarray(tape['mask'],bool)
    for onset in np.unique(delays):
        lanes=np.flatnonzero(delays==onset)
        actual=mask[:,lanes]
        request=np.asarray(tape['requested_delta'])[:,lanes][actual]
        effective=np.asarray(tape['effective_delta'])[:,lanes][actual]
        groups[str(int(onset))]=dict(episodes=len(lanes),applied_actions=int(actual.sum()),
            snapshot_control_steps=[rows[i]['snapshot_control_step'] for i in lanes
                                    if rows[i]['pulse_outcome']=='valid_post_pulse'],
            requested_absolute_max=float(np.max(np.abs(request))) if request.size else 0.,
            effective_absolute_max=float(np.max(np.abs(effective))) if effective.size else 0.,
            clipped_channels=int(np.asarray(tape['action_clipped'])[:,lanes][actual].sum()))
    return dict(pulse_contract=contract,collection_budget=collection_budget(spec,spec['round_index']),
        episode_performance_denominator=len(rows),pulse_onset_summary=groups,
        pulse_outcome_counts={outcome:sum(r['pulse_outcome']==outcome for r in rows) for outcome in
            ('pre_pulse_terminal','during_pulse_terminal','valid_post_pulse')})


def suffix_label(valid, failure, timeout, done, horizon_reached):
    from .unified_continuation_labels import classify_first_valid_landing_outcome
    if valid and failure:return None,'simultaneous_landing_failure_unresolved'
    positive,outcome=classify_first_valid_landing_outcome(valid_contact_seen=valid,physical_failure_before_landing=failure,timeout=timeout,done=done,reached_rollout_horizon=horizon_reached)
    return int(positive),outcome


def terminal_prefix_label(valid, failure):
    """Old tapes lack failure flags: a landing then cannot establish success."""
    if valid and failure is None:return None,'legacy_terminal_landing_unresolved'
    return suffix_label(bool(valid),bool(failure),False,True,False)


def completed_trace_endpoint(*, arrays, terminal, stage_reached, full_episode,
                             valid, failure, prefix_sha, lane, tick):
    """Return evaluation cutoff evidence without inventing physical termination."""
    if not (terminal or not stage_reached or full_episode):
        return None
    label, reason = terminal_prefix_label(valid, failure)
    if not stage_reached:
        label, reason = None, 'stage_not_reached'
    elif full_episode and not terminal:
        label, reason = None, 'horizon_exhausted'
    return dict(snapshot=None,
                state_sha256=canonical_sha256(dict(qpos=arrays['data/qpos'].tolist(), qvel=arrays['data/qvel'].tolist())),
                snapshot_context_sha256=canonical_sha256(dict(terminal_prefix_sha256=prefix_sha, lane=lane, tick=tick)),
                prefix_label=label, terminal_reason=reason,
                endpoint_kind='terminal_trace' if terminal else 'horizon_trace' if stage_reached else 'stage_not_reached_trace',
                terminal_tick=tick)


def aggregate_labels(attempts,order):
    if any(a['label']==1 for a in attempts):return 1
    by_policy={a['policy']:a['label'] for a in attempts}
    return 0 if all(name in by_policy and by_policy[name]==0 for name in order) else None


def recovery_mode(spec):
    criterion = spec.get('success_criterion', 'first_valid_landing')
    if criterion not in ('first_valid_landing', 'stable_forward_recovery'):
        raise ValueError('unsupported pulse success criterion')
    return criterion == 'stable_forward_recovery'


def endpoint_state(state, spec):
    from .iterative_probe_training import first_landing_state
    return state if recovery_mode(spec) else first_landing_state(state)


def endpoint_success(state, spec):
    return state.info['success'] if recovery_mode(spec) else state.info['down_events'].valid_contact_seen


def networks(spec):
    controller_mode(spec)
    selected_event(spec)
    descent_clearance(spec)
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
    config_path = Path(member['policy']['formal_config'])
    raw_config = read(config_path)
    if spec.get('frozen_explorer_evaluation') and raw_config.get('continuous_learner') is not None:
        from .iterative_probe_training import load_config, build_environment
        config = load_config(config_path, evaluation_only=True)
        _, env = build_environment(config)
    else:
        config,_,env=build_unified_formal_environment(config_path)
    # This runtime owns its declared pulse schedule. Never apply the training
    # augmentation a second time, including nominal and snapshot continuations.
    env._training_action_pulse = None
    if config.raw.get('success_criterion', 'first_valid_landing') != spec.get('success_criterion', 'first_valid_landing'):
        raise ValueError('source and exploration endpoint differ')
    env._reward_mode=spec.get('reward_mode',config.raw.get('reward_mode','phase_recovery'))
    if spec.get('phase_policy'):
        from .rsi_comparison import load_phase_evaluation_policy
        payload,member=load_phase_evaluation_policy(spec,config,member)
    else:
        payload=load_checkpoint(Path(member['policy']['checkpoint']),expected=checkpoint_identity(config,env))
    for k,v in [('actor_sha256',payload.actor_params),('normalizer_sha256',payload.observation_normalizer),('critic_sha256',payload.critic_params)]:
        if pytree_sha256(v)!=member['policy'][k]:raise ValueError('base payload drift: '+k)
    if spec.get('explorer_backend')=='rsl_rl':
        from .rsl_pulse import initialize,restore,validate_neighborhood_identity
        state=(restore(spec['explorer_checkpoint']) if spec.get('explorer_checkpoint') else
               initialize(spec,payload.observation_normalizer.mean['privileged_state'],
                          payload.observation_normalizer.std['privileged_state']))
        validate_neighborhood_identity({'neighborhood':spec.get('neighborhood')},state)
        return env,payload,member,None,None,state
    net=make_exploration_network_factory()({'state':76,'privileged_state':106},4,preprocess_observations_fn=running_statistics.normalize)
    rng=jax.random.PRNGKey(spec['seed']);rng,a,v=jax.random.split(rng,3)
    params=dict(policy=initialize_residual_actor(net,a),value=net.value_network.init(v))
    optimizer=optax.chain(optax.clip_by_global_norm(spec['max_grad_norm']),optax.adam(spec['learning_rate']))
    return env,payload,member,net,optimizer,dict(params=params,optimizer=optimizer.init(params),rng=rng)


def explorer_inventory(spec, params):
    import jax
    rsl=spec.get('explorer_backend')=='rsl_rl'
    actor,critic=('actor','critic') if rsl else ('policy','value')
    return dict(actor_parameters=sum(x.size for name in [actor,actor+'_encoder'] if name in params for x in jax.tree.leaves(params[name])),
                critic_parameters=sum(x.size for name in [critic,critic+'_encoder'] if name in params for x in jax.tree.leaves(params[name])),
                hidden=[128]*3 if rsl else [256]*3,activation='elu' if rsl else 'swish',
                backend='rsl_rl_3.2.0' if rsl else 'jax_custom')


def physical_trace(env, state):
    """Truthful post/pre-state fields at control resolution, independent of endpoint."""
    import jax
    from types import SimpleNamespace
    from .geometry import extract_geometry
    # Geometry only needs these per-world leaves. DataWarp also carries shared
    # contact/capacity arrays, which must never be naively vmapped with the worlds.
    def extract(qpos, qvel, geom_xpos, geom_xmat):
        data=SimpleNamespace(qpos=qpos,qvel=qvel,geom_xpos=geom_xpos,geom_xmat=geom_xmat)
        return extract_geometry(data,env._geometry)
    geometry=jax.vmap(extract)(state.data.qpos,state.data.qvel,
                              state.data.geom_xpos,state.data.geom_xmat)
    down = state.info['down_events']
    return dict(valid_contact_seen=down.valid_contact_seen, contact_x=down.contact_x,
                recovery_ticks=down.post_contact_ticks, time=state.data.time,
                front_wheel_clearance=geometry.front_wheel_terrain_clearance,
                rear_wheel_clearance=geometry.rear_wheel_terrain_clearance,
                maximum_wheel_penetration=geometry.maximum_wheel_penetration,
                prohibited_contact=geometry.prohibited_contact)


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
    from .generative_bridge.pulse_protocol import collection_draws, pulse_delta_for_step, normalize_pulse_contract
    from .generative_bridge.rollout import observation_fields
    from .generative_bridge.explorer_admission import (collection_mixture, episode_policy_keys,
        collection_receipt, mix_sample)
    mixture_learned, mixture_uniform = collection_mixture(spec)
    pulse_contract=normalize_pulse_contract(spec)
    mixture_keys = episode_policy_keys(spec)
    logical_draws, pulse_receipt = collection_draws(spec)
    record_preobs = bool(spec.get('record_actor_preobservations', False))
    if jax.default_backend()!='gpu':raise RuntimeError('GPU pulse collection required')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    env,payload,member,net,opt,state=networks(spec)
    if spec.get('explorer_checkpoint') and spec.get('explorer_backend')!='rsl_rl':
        state=serialization.from_bytes(state,Path(spec['explorer_checkpoint']).read_bytes())
    (output/'behavior.msgpack').write_bytes(serialization.to_bytes(state))
    normalizer=payload.observation_normalizer;params=state['params'];count=spec['num_envs']
    admission_receipt = None
    if mixture_learned is not None:
        admission_receipt = collection_receipt(spec,state,mixture_learned)
        mixture_learned,mixture_uniform,mixture_keys=map(jp.asarray,
            (mixture_learned,mixture_uniform,mixture_keys))
    if pulse_receipt is not None:
        write(output/'pulse_protocol_manifest.json', pulse_receipt)
        logical_draws = jp.asarray(logical_draws)
    from .pulse_exploration import pulse_delay
    mode=controller_mode(spec);event=selected_event(spec)
    descent_limit=descent_clearance(spec)
    delay=0 if event else pulse_delay(spec,spec['round_index'])
    mixed=spec.get('pulse_batch_mode')=='mixed' and not spec.get('nominal_source_rollout')
    delays=lane_onsets(spec,spec['round_index']) if not event else np.zeros(count,np.int32)
    prefix_steps=collection_steps(spec,delays,event)
    full_episode=bool(spec.get('full_episode_rollout'))
    neighbor_query=None
    if spec.get('neighborhood'):
        from .neighborhood_sampling import prepare_neighbor_query
        neighbor_query=prepare_neighbor_query(spec,env,member,output)
    if event and (spec['horizon']!=400 or not 1<=spec['pulse_steps']<=5):
        raise ValueError('event pulse requires horizon400 and one to five pulse steps')
    base=make_checkpoint_policy(env,payload,deterministic=True);dist=net.parametric_action_distribution if net else None
    reset=jax.vmap(env._reset_jump_start_unified)
    initial_arrays=None
    if spec.get('initial_state_bank'):
        if not spec.get('frozen_explorer_evaluation'):
            raise ValueError('explicit initial state bank requires frozen evaluation')
        from .evaluation_initial_state import load_initial_state_bank
        initial_arrays=load_initial_state_bank(spec['initial_state_bank'],count=count,
            nq=env.mj_model.nq,nv=env.mj_model.nv,root_qpos=env._bundle.model_index.root_qpos_address)
    fixed_requests=None
    if spec.get('frozen_request_table'):
        from .retention_repair import load_requests
        fixed_requests,delays=load_requests(spec)
        fixed_requests=jp.asarray(fixed_requests)
    step=jax.vmap(lambda s,a:endpoint_state(env.step(s,a),spec))
    def run(rng):
        reset_key, noise_key, pulse_key = jax.random.split(rng, 3)
        initial_velocity_noise = sample_initial_velocity_noise(
            jax.random.split(noise_key, count), env._bundle.model_index,
            env.mj_model.nv, spec)
        if initial_arrays is None:
            initial=prepare_parallel_worlds(
                reset(jax.random.split(reset_key, count), initial_velocity_noise), env, count)
        else:
            q,v=initial_arrays
            initial=prepare_parallel_worlds(reset(jax.random.split(reset_key,count),
                initial_velocity_noise,jp.asarray(q),jp.asarray(v)),env,count)
        def advance(carry,tick):
            s,key,alive,trigger_tick,applied_steps,previous_vz=carry;key,k=jax.random.split(key)
            before=physical_trace(env,s)
            vz=s.data.qvel[:,env._bundle.model_index.root_dof_address+2]
            if event:
                ready=event_ready(event,tick=tick,front_clearance=before['front_wheel_clearance'],
                    rear_clearance=before['rear_wheel_clearance'],previous_vertical_velocity=previous_vz,
                    vertical_velocity=vz,valid_contact_seen=before['valid_contact_seen'],
                    min_airborne_clearance=env._down_config.descent.min_airborne_clearance,
                    min_descent_velocity=env._resolved_config.events.min_descent_velocity,
                    descent_clearance=descent_limit)
            else:
                ready=tick>=jp.asarray(delays)
            trigger_tick,pulse_mask=pulse_activity(trigger_tick,applied_steps,ready,alive,tick,spec['pulse_steps'])
            explorer_obs=s.obs['privileged_state']
            if neighbor_query is not None:
                explorer_obs=jp.concatenate([explorer_obs,neighbor_query(s,pulse_mask)],axis=-1)
            if mode=='fixed_random':
                delta=(pulse_delta_for_step(logical_draws, applied_steps, pulse_mask)
                       if logical_draws is not None else
                       jax.random.uniform(k,(count,4),minval=-1.,maxval=1.))
                raw=delta
                log_prob=jp.full((count,),-4.*jp.log(2.))
                value=jp.zeros(count)
            elif spec.get('explorer_backend')=='rsl_rl':
                from .rsl_pulse import sample
                if mixture_keys is None:
                    raw,delta,log_prob,value=sample(state,explorer_obs,k)
                else:
                    episode_keys=jax.vmap(lambda key:jax.random.fold_in(key,tick))(mixture_keys)
                    raw,delta,log_prob,value=jax.vmap(lambda obs,key:sample(state,obs,key))(explorer_obs,episode_keys)
            else:
                logits=net.policy_network.apply(normalizer,params['policy'],s.obs)
                raw=dist.sample_no_postprocessing(logits,k);delta=dist.postprocess(raw)
                log_prob=dist.log_prob(logits,raw)
                value=net.value_network.apply(normalizer,params['value'],s.obs)
            if mixture_learned is not None:
                uniform=pulse_delta_for_step(mixture_uniform,applied_steps,pulse_mask)
                raw,delta,log_prob,value,on_policy=mix_sample(raw,delta,log_prob,value,
                    mixture_learned,uniform,pulse_mask & alive)
            b=base(s.obs,k)[0]
            apply_pulse=pulse_mask[:,None] if (event or mixed or full_episode or logical_draws is not None or mixture_learned is not None) else tick>=delay
            if fixed_requests is not None:
                requested=jp.where(alive[:,None],fixed_requests[tick],0.)
                action=jp.clip(b+requested,-1.,1.)
                effective=action-b
            else:
                action,requested,effective=compose_residual_action(b,jp.where(apply_pulse,delta,0.),jp.asarray(spec['delta_limit']))
            nxt=step(s,action)
            if pulse_contract is not None:
                nxt=freeze_inactive_worlds(nxt,s,alive)
            finite=jp.all(jp.isfinite(nxt.data.qpos),axis=-1)&jp.all(jp.isfinite(nxt.data.qvel),axis=-1)
            terminal=nxt.done.astype(bool)|~finite
            after=physical_trace(env,nxt)
            tape=dict(observation=explorer_obs,raw_action=raw,log_prob=log_prob,value=value,mask=pulse_mask,prefix_mask=alive,terminal=terminal,finite=finite,action=action,base_action=b,delta=delta,requested_delta=requested,effective_delta=effective,qpos=nxt.data.qpos,qvel=nxt.data.qvel,phase=nxt.info['active_phase'],phase_before=s.info['active_phase'],phase_after=nxt.info['active_phase'],pulse_triggered=trigger_tick>=0,pulse_trigger_tick=trigger_tick,pulse_step_index=jp.where(pulse_mask,applied_steps,-1),pulse_event_ready=ready,action_clipped=jp.abs(requested-effective)>1e-7)
            if mixture_learned is not None:
                tape.update(explorer_learned=mixture_learned,on_policy_mask=on_policy,
                    log_prob_valid=on_policy)
            tape.update(observation_fields(s,nxt,action,enabled=record_preobs))
            if spec.get('record_retention_diagnostics'):
                tape['reward']=nxt.reward
                tape.update({'metric/'+k:v for k,v in nxt.metrics.items() if k.startswith(('reward/','terminal/','event/'))})
                tape.update(roll=nxt.info['reward_state'].roll,roll_rate=nxt.info['reward_state'].roll_rate,
                    airborne_seen=nxt.info['down_events'].airborne_seen,roll_limit=nxt.info['roll_limit'])
            tape.update(after)
            tape.update({k+'_before':v for k,v in before.items()})
            tape['first_valid_contact']=~before['valid_contact_seen']&after['valid_contact_seen']&alive
            tape['time_after']=after['time'];tape['time_before']=before['time']
            tape.update(physical_failure=nxt.info['physical_failure'],end_code=nxt.info['end_code'])
            if full_episode:
                tape['success']=endpoint_success(nxt,spec)
            tape.update({'snap/'+k:v for k,v in snapshot_arrays(nxt).items()})
            nxt=freeze_inactive_worlds(nxt,s,alive)
            applied_steps=applied_steps+pulse_mask.astype(jp.int32)
            active=alive&~terminal
            if (event or mixed) and not full_episode:active=active&(applied_steps<spec['pulse_steps'])
            return (nxt,key,active,trigger_tick,applied_steps,vz),tape
        carry=(initial,pulse_key,jp.ones(count,bool),jp.full(count,-1,jp.int32),jp.zeros(count,jp.int32),
               initial.data.qvel[:,env._bundle.model_index.root_dof_address+2])
        if event:
            shape=jax.eval_shape(lambda c,t:advance(c,t)[1],carry,jp.asarray(0,jp.int32))
            traces=jax.tree.map(lambda x:jp.zeros((prefix_steps,)+x.shape,x.dtype),shape)
            def condition(c):return (c[0]<prefix_steps)&jp.any(c[1][2])
            def body(c):
                tick,prior,traces=c
                following,frame=advance(prior,tick)
                traces=jax.tree.map(lambda trace,value:trace.at[tick].set(value),traces,frame)
                return tick+1,following,traces
            ticks,carry,tape=jax.lax.while_loop(condition,body,(jp.asarray(0),carry,traces))
        else:
            carry,tape=jax.lax.scan(advance,carry,jp.arange(prefix_steps),length=prefix_steps)
            ticks=jp.asarray(prefix_steps)
        tape['initial_velocity_noise'] = jp.broadcast_to(
            initial_velocity_noise, (prefix_steps, count, env.mj_model.nv)
        )
        if initial_arrays is not None:
            tape['initial_qpos']=initial.data.qpos
            tape['initial_qvel']=initial.data.qvel
        return carry[1],ticks,tape
    start=time.monotonic();rng,key=jax.random.split(state['rng'])
    if spec.get('frozen_explorer_evaluation'):
        key=jax.random.PRNGKey(spec['seed'])
    write(output/'status.json',dict(phase='running',charged_interactions=count*prefix_steps))
    if spec.get('reuse_prefix_collection'):
        import shutil
        previous=Path(spec['reuse_prefix_collection']);old=read(previous.parent/'collection_spec.json')
        for field in ['round_index','explorer_checkpoint','pulse_steps','num_envs','delta_limit','bank','seed','pulse_start_schedule','pulse_event_schedule','controller_mode','pulse_descent_clearance','pulse_batch_mode','neighborhood','neighborhood_map_sha256','pulse_protocol_v1_2','record_actor_preobservations','pulse_contract_version','single_pulse_per_episode','maximum_collection_steps']:
            if old.get(field)!=spec.get(field):raise ValueError('reused prefix contract differs: '+field)
        if _file_sha(previous/'behavior.msgpack')!=_file_sha(output/'behavior.msgpack'):raise ValueError('reused behavior differs')
        for filename in ['prefixes.npz','update_state.msgpack']:
            if spec['input_files'].get(str(previous/filename))!=_file_sha(previous/filename):raise ValueError('unlocked reused prefix state')
        tape=dict(np.load(previous/'prefixes.npz'))
        executed_ticks=len(tape['prefix_mask'])
        state=serialization.from_bytes(state,(previous/'update_state.msgpack').read_bytes());next_rng=state['rng']
        shutil.copyfile(previous/'prefixes.npz',output/'prefixes.npz')
        write(output/'reuse.json',dict(previous=str(previous),physics_replayed=False,charged_interactions=0))
    else:
        next_rng,executed_ticks,tape=jax.device_get(jax.jit(run)(key))
        executed_ticks=int(executed_ticks);tape={k:(v if k in ('initial_qpos','initial_qvel') else v[:executed_ticks]) for k,v in tape.items()}
        np.savez_compressed(output/'prefixes.npz',**tape)
    if np.any(tape['prefix_mask']&~tape['finite']):raise ValueError('nonfinite pulse prefix')
    state['rng']=next_rng;(output/'update_state.msgpack').write_bytes(serialization.to_bytes(state))
    if admission_receipt is not None:
        admission_receipt['files']={name:_file_sha(output/name) for name in
            ('prefixes.npz','behavior.msgpack','update_state.msgpack')}
        write(output/'explorer_admission.json',admission_receipt)
    prefix_sha = _file_sha(output/'prefixes.npz')
    behavior_sha = _file_sha(output/'behavior.msgpack')
    generator={**member['policy'],'actor_sha256':canonical_sha256(dict(base=member['policy']['actor_sha256'],residual=behavior_sha,delta=spec['delta_limit'],pulse_steps=spec['pulse_steps'],pulse_start_step=delay,pulse_batch_mode=spec.get('pulse_batch_mode','single'),neighborhood_map_sha256=spec.get('neighborhood_map_sha256'),controller_mode=mode,pulse_event=event,pulse_descent_clearance=descent_limit)),'payload_sha256':behavior_sha}
    if pulse_contract is not None:
        generator['actor_sha256']=canonical_sha256(dict(legacy_identity=generator['actor_sha256'],pulse_contract=pulse_contract))
    rows=[]
    for e in range(count):
        t=int(np.flatnonzero(tape['prefix_mask'][:,e])[-1])
        arrays={k[5:]:v[t,e] for k,v in tape.items() if k.startswith('snap/')}
        terminal=bool(tape['terminal'][t,e])
        trigger_step=int(tape['pulse_trigger_tick'][t,e]) if 'pulse_trigger_tick' in tape else (delay if tape['mask'][:,e].any() else -1)
        stage_reached=trigger_step>=0 if pulse_contract is not None else event is None or trigger_step>=0
        phase='upstream' if int(arrays['info/active_phase'])==0 else 'downstream'
        coords=physical_coordinates_from_arrays(tape['qpos'][t,e],tape['qvel'][t,e],bundle=env._bundle)
        endpoint=completed_trace_endpoint(arrays=arrays, terminal=terminal, stage_reached=stage_reached,
            full_episode=full_episode, valid=bool(arrays['down/recovery_success'] if recovery_mode(spec) else arrays['down/valid_contact_seen']),
            failure=bool(tape['physical_failure'][t,e]) if 'physical_failure' in tape else None,
            prefix_sha=prefix_sha, lane=e, tick=t)
        if endpoint is None:
            snap=snapshot_from_arrays(arrays,env=env,record=generator,parent_trajectory=str(output/'prefixes.npz')+'::'+str(e),parent_state_sha256=prefix_sha)
            path=output/'snapshots'/f'{e:05d}';save_unified_envelope_snapshot(path,snap)
            endpoint=dict(snapshot=str(path),state_sha256=physical_state_sha256(snap),snapshot_context_sha256=snapshot_context_sha256(snap),endpoint_kind='continuation_snapshot')
        rows.append(dict(index=e,cell=_cell_id(phase,'root_geometry_v1',quantize_coordinates(coords,ROOT_GEOMETRY_FIELDS)),coordinates=coords,phase=phase,**endpoint,prefix_file=str(output/'prefixes.npz'),prefix_sha256=prefix_sha,behavior_sha256=behavior_sha,prefix_terminal=terminal or not stage_reached or full_episode,physical_prefix_terminal=terminal,rollout_horizon_exhausted=full_episode and not terminal,prefix_physical_failure=bool(tape['physical_failure'][t,e]) if 'physical_failure' in tape else False,pulse_start_step=trigger_step if (event or mixed) else delay,pulse_trigger_step=trigger_step,pulse_event=event,stage_reached=stage_reached,pulse_applied_steps=int(tape['mask'][:,e].sum()),label=None,learning_attempted=False))
        if pulse_contract is not None:
            from .pulse_schedule import pulse_outcome
            rows[-1].update(pulse_outcome=pulse_outcome(rows[-1]['pulse_applied_steps'],spec['pulse_steps'],terminal),
                pulse_scheduled_start_step=int(delays[e]),pulse_contract=pulse_contract,
                snapshot_control_step=t+1,episode_performance_denominator=True)
    if admission_receipt is not None:
        for row,episode_id,mode_name in zip(rows,admission_receipt['episode_ids'],admission_receipt['episode_modes']):
            row.update(explorer_episode_mode=mode_name,explorer_on_policy=mode_name=='learned',
                explorer_lineage={k:admission_receipt[k] for k in ('run_id','round','collection_id',
                    'behavior_actor_sha256','behavior_normalizer_sha256')},logical_episode_id=episode_id)
    if pulse_receipt is not None:
        for row, receipt in zip(rows, pulse_receipt['episodes']):
            row['logical_episode'] = {**receipt, 'master_seed': pulse_receipt['master_seed'],
                'role': pulse_receipt['role'], 'round': pulse_receipt['round']}
    if spec.get('quality_mode')=='discovery_conversion':
        for e,row in enumerate(rows):
            cells=[]
            for tick in np.flatnonzero(tape['mask'][:,e]):
                coords=physical_coordinates_from_arrays(tape['qpos'][tick,e],tape['qvel'][tick,e],bundle=env._bundle)
                phase='upstream' if int(tape['phase_after'][tick,e])==0 else 'downstream'
                cells.append(_cell_id(phase,'root_geometry_v1',quantize_coordinates(coords,ROOT_GEOMETRY_FIELDS)))
            row['pulse_cells']=cells
    write(output/'candidates.json',rows)
    write(output/'network_inventory.json',dict(**explorer_inventory(spec,params),base_actor_frozen=True,base_critic_frozen=True,inputs=int(np.asarray(state['normalizer_mean']).size) if spec.get('explorer_backend')=='rsl_rl' else 106,history_frames=3,output_actions=4,controller_mode=mode,explorer_actor_used=mode=='learned_residual',explorer_trainable=mode=='learned_residual' and not spec.get('frozen_explorer_evaluation',False),exploration_critic_used=mode=='learned_residual',random_distribution='uniform[-1,1]' if mode=='fixed_random' else None,trace_schema='jit_pulse_physical_trace_v2',event_time_resolution_seconds=.02))
    write(output/'hyperparameters.json',{**spec,'pulse_descent_clearance':descent_limit,
        **({'inference_precision':'highest'} if spec.get('explorer_backend')=='rsl_rl' else {})})
    active_count=int(tape['prefix_mask'].sum());physical_count=count*executed_ticks
    write(output/'status.json',dict(phase='completed',charged_interactions=0 if spec.get('reuse_prefix_collection') else physical_count,active_interactions=active_count,padding_interactions=physical_count-active_count,waiting_interactions=active_count-int(tape['mask'].sum()),pulse_training_steps=int(tape['mask'].sum()) if mode=='learned_residual' and not spec.get('frozen_explorer_evaluation') else 0,pulse_applied_steps=int(tape['mask'].sum()),pulse_start_step=delay if event is None and not mixed else None,pulse_batch_mode=spec.get('pulse_batch_mode','single'),onset_counts={str(int(t)):int((delays==t).sum()) for t in np.unique(delays)},pulse_event=event,stage_not_reached=sum(not r['stage_reached'] for r in rows),executed_ticks=executed_ticks,wall_seconds=time.monotonic()-start,**scheduled_pulse_summary(spec,tape,rows)))


def evaluate(spec, output, *, session=None):
    """Evaluate with canonical restoration, optionally in bounded processes."""
    if spec.get('bridge_action_plan') and spec.get('evaluation_batch_size') is not None:
        raise ValueError('bridge candidates must be explicitly sharded with identity-bound plans')
    if spec.get('evaluation_batch_size') is not None:
        from .pulse_evaluation_batches import evaluation_shards, evaluate_batched
        rows=read(spec['candidates'])
        batches=evaluation_shards(rows,spec['evaluation_batch_size'])
        if len(batches)>1:
            return evaluate_batched(spec,output)
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
    from .generative_bridge.rollout import (load_prefix_plan, prefix_action, observation_fields,
        closed_loop_action, validate_evaluation_options, evaluation_controller_provenance,
        warmup_evaluation_policy)
    bridge_plan=load_prefix_plan(spec,rows)
    record_preobs=(spec.get('record_actor_preobservations',False) or bridge_plan is not None
                   or spec.get('closed_loop_prefix_policy') is not None
                   or spec.get('warmup_initializer') is not None)
    bank=load_probe_bank(Path(spec['bank']));all_names=[m['name'] for m in bank['members'] if 'evaluator' in m['roles']]
    suffix=(FrozenSuffixEvaluator(spec['bank'],all_names,horizon,output/'runtime',spec['budget'])
            if session is None else session.suffix(spec,all_names,output/'runtime'))
    prefix_name=validate_evaluation_options(spec,suffix.members)
    if session is not None and (prefix_name is not None or spec.get('warmup_initializer') is not None):
        raise ValueError('teacher session forbids unpinned alternate controllers')
    for r in rows:r.update(attempts=[],label=None,witness=None)
    if spec.get('reuse_results'):
        reused=read(spec['reuse_results'])
        if len(reused)!=len(rows):raise ValueError('reused candidate count differs')
        for row,old in zip(rows,reused):
            if row['snapshot_context_sha256']!=old['snapshot_context_sha256']:raise ValueError('reused context differs')
            for a in old['attempts']:
                if a['actor_sha256']!=suffix.members[a['policy']]['policy']['actor_sha256'] or _file_sha(Path(a['trace']))!=a['trace_sha256']:raise ValueError('reused evaluator/trace drift')
            row.update(attempts=old['attempts'],label=old['label'],witness=old['witness'])
    start=time.monotonic(); timings=[]
    for name in spec['order']:
        subset=[r for r in rows if (spec.get('full_matrix',False) or r['label']!=1) and not r.get('prefix_terminal',False) and not any(a['policy']==name for a in r['attempts'])]
        if not subset:continue
        stage_start=time.monotonic()
        from .generative_bridge.performance import measure,record_event
        import os
        with measure('model_actor_generator_load',backend='gpu',stage=name):
            env,policy,_=suffix._runtime(name)
        actual_policy=suffix.members[name]['policy']
        if spec.get('warmup_initializer') is not None:
            policy,actual_policy=warmup_evaluation_policy(env,actual_policy,spec['warmup_initializer'])
        closed_prefix_policy=None
        if prefix_name is not None:
            _,closed_prefix_policy,_=suffix._runtime(prefix_name)
            if suffix.members[prefix_name]['policy']['xml_sha256'] != actual_policy['xml_sha256']:
                raise ValueError('closed-loop prefix/tail physics differ')
        controller_provenance=evaluation_controller_provenance(spec,actual_policy,suffix.members)
        runtime_ready=time.monotonic()
        if record_preobs:
            env._training_action_pulse = None
        env._reward_mode=spec.get('reward_mode',getattr(env,'_reward_mode','phase_recovery'))
        restore_wall=time.time()
        restored=[]
        for r in subset:
            snap=load_unified_envelope_snapshot(Path(r['snapshot']))
            if snapshot_context_sha256(snap)!=r['snapshot_context_sha256']:raise ValueError('suffix snapshot identity changed')
            if spec.get('preserve_snapshot_episode_context'):
                from .unified_envelope_snapshot import restore_unified_envelope_snapshot
                state=restore_unified_envelope_snapshot(snap,env)
                if not np.array_equal(np.asarray(state.obs['state']),np.asarray(snap.observation)):
                    raise ValueError('restored Actor observation/history differs')
                # Snapshot schema keeps event/episode clocks but omits MuJoCo time.
                # New retention roots bind the actual recorded post-pulse time.
                if 'snapshot_time' not in r:raise ValueError('retention snapshot requires recorded time')
                state=state.replace(data=state.data.replace(time=jp.asarray(r['snapshot_time'],jp.float32)))
            else:
                state=fresh_unified_continuation_start(snap,env)
            if recovery_mode(spec) and not spec.get('preserve_snapshot_episode_context'):
                state=state.replace(info={**state.info,'down_events':state.info['down_events'].replace(post_contact_ticks=jp.asarray(0,jp.int32),recovery_success=jp.asarray(False))})
            restored.append(state)
        count=len(restored);initial=prepare_parallel_worlds(stack_worlds(restored),env,count)
        # The stacked world now owns every required leaf; release individual worlds
        # before compiling/stepping to avoid retaining two full state representations.
        jax.block_until_ready(initial)
        del restored,state,snap
        rng_count=spec.get('suffix_rng_count',count)
        rng_indices=spec.get('suffix_rng_indices',list(range(count)))
        if (type(rng_count) is not int or rng_count<count or len(rng_indices)!=count
                or len(set(rng_indices))!=count
                or any(type(i) is not int or not 0<=i<rng_count for i in rng_indices)):
            raise ValueError('invalid original suffix RNG lane mapping')
        if charged+count*horizon>spec['budget']:raise RuntimeError('insufficient declared suffix reservation')
        write(output/'status.json',dict(phase='running',policy=name,charged_interactions=charged,reserved_attempt_interactions=count*horizon))
        restore_ready=time.monotonic()
        record_event(os.environ.get('JIT_PERFORMANCE_FILE'),'snapshot_read_and_restore',restore_wall,
            restore_ready-runtime_ready,backend='gpu',batch_size=count,stage=name)
        if bridge_plan is not None:
            bridge_prefixes,bridge_source_only=map(jp.asarray,bridge_plan)
            if len(subset)!=len(rows):raise ValueError('bridge requires all candidates in one frozen-policy batch')
        step=jax.vmap(lambda s,a:endpoint_state(env.step(s,a),spec))
        if bridge_plan is None:
            bridge_prefixes=jp.zeros((count,16,4),jp.float32)
            bridge_source_only=jp.ones(count,bool)
        def rollout(initial,bridge_prefixes=bridge_prefixes,bridge_source_only=bridge_source_only):
            def frame(s,action,mask,previous):
                result=dict(qpos=s.data.qpos,qvel=s.data.qvel,action=action,reward=s.reward,mask=mask,done=s.done,success=s.info['success'],physical_failure=s.info['physical_failure'],timeout=s.info['timeout'],end_code=s.info['end_code'],valid_contact=endpoint_success(s,spec))
                before=physical_trace(env,previous);after=physical_trace(env,s)
                result.update(after)
                result.update({k+'_before':v for k,v in before.items()})
                result.update(phase_before=previous.info['active_phase'],phase_after=s.info['active_phase'],
                    first_valid_contact=~before['valid_contact_seen']&after['valid_contact_seen']&mask,
                    time_before=before['time'],time_after=after['time'],base_action=action,
                    requested_delta=jp.zeros_like(action),effective_delta=jp.zeros_like(action))
                result.update(observation_fields(previous,s,action,enabled=record_preobs))
                if spec.get('record_retention_diagnostics'):
                    result.update(roll=s.info['reward_state'].roll,roll_rate=s.info['reward_state'].roll_rate,
                        roll_limit=s.info['roll_limit'],phase_episode_step=s.info['phase_episode_step'],episode_step=s.info['episode_step'])
                    result.update({'metric/'+k:v for k,v in s.metrics.items() if k.startswith(('reward/','terminal/','event/'))})
                return result
            blank=frame(initial,jp.zeros((count,4)),jp.zeros(count,bool),initial)
            if bridge_plan is not None or prefix_name is not None:blank['action_origin_code']=jp.zeros(count,jp.int32)
            traces={k:jp.zeros((horizon,)+v.shape,v.dtype) for k,v in blank.items()}
            def condition(c):return (c[0]<horizon)&jp.any(c[2])
            def advance(c):
                t,s,alive,tr=c
                keys=jax.random.split(jax.random.fold_in(jax.random.PRNGKey(0),t),rng_count)[jp.asarray(rng_indices)]
                action=closed_loop_action(t,s.obs,keys,policy,closed_prefix_policy)
                if bridge_plan is not None:
                    action=prefix_action(t,action,bridge_prefixes,bridge_source_only)
                nxt=step(s,jp.where(alive[:,None],action,0))
                finite=jp.all(jp.isfinite(nxt.data.qpos),axis=-1)&jp.all(jp.isfinite(nxt.data.qvel),axis=-1)
                f=frame(nxt,action,alive,s)
                if bridge_plan is not None:
                    f['action_origin_code']=jp.where(bridge_source_only,0,jp.where(t<16,1,2))
                elif prefix_name is not None:
                    f['action_origin_code']=jp.full(count,jp.where(t<16,3,2),jp.int32)
                tr={k:v.at[t].set(f[k]) for k,v in tr.items()}
                def choose(path,n,o):return n if _shared_warp(path) else jp.where(alive.reshape((count,)+(1,)*(n.ndim-1)),n,o)
                nxt=jax.tree_util.tree_map_with_path(choose,nxt,s)
                alive=alive&~nxt.done.astype(bool)&~endpoint_success(nxt,spec)&finite
                return t+1,nxt,alive,tr
            tick,final,_,tr=jax.lax.while_loop(condition,advance,(jp.array(0),initial,jp.ones(count,bool),traces))
            return tick,tr
        if bridge_plan is None:
            bridge_prefixes=jp.zeros((count,16,4),jp.float32)
            bridge_source_only=jp.ones(count,bool)
        if session is None:
            # Default path keeps original closed-over constant prefix semantics.
            tick,tape=jax.device_get(jax.jit(rollout)(initial))
        else:
            from .generative_bridge.teacher_runtime import kernel_identity
            key=kernel_identity(spec,actual_policy,count,record_preobs,prefix_name,bridge_plan is not None)
            tick,tape=session.execute(key,rollout,initial,bridge_prefixes,bridge_source_only)
        tick=int(tick);tape={k:v[:tick] for k,v in tape.items()}
        rollout_ready=time.monotonic()
        cost=count*tick;charged+=cost;active_count+=int(tape['mask'].sum())
        trace_path=output/(name+'_traces.npz');np.savez_compressed(trace_path,**tape)
        trace_sha = _file_sha(trace_path)
        for e,r in enumerate(subset):
            indices=np.flatnonzero(tape['mask'][:,e]);last=int(indices[-1]);valid=bool(tape['valid_contact'][last,e]);failure=bool(tape['physical_failure'][last,e])
            if not np.isfinite(tape['qpos'][indices,e]).all() or not np.isfinite(tape['qvel'][indices,e]).all():raise ValueError('nonfinite suffix')
            label,outcome=suffix_label(valid,failure,bool(tape['timeout'][last,e]),bool(tape['done'][last,e]),len(indices)>=horizon)
            if recovery_mode(spec) and label == 1: outcome='stable_forward_recovery'
            r['attempts'].append(dict(policy=name,actor_sha256=actual_policy['actor_sha256'],label=label,outcome=outcome,steps=len(indices),task_return=float(tape['reward'][indices,e].sum()),trace=str(trace_path),trace_lane=e,trace_sha256=trace_sha,snapshot_context_sha256=r['snapshot_context_sha256']))
            r['attempts'][-1].update(controller_provenance)
            if record_preobs:
                r['attempts'][-1].update(recording_schema='jit_actor_success_trace_v1_1' if bridge_plan is None else 'jit_bridge_trace_v1',
                    action_origin=('closed_loop_prefix_then_tail' if prefix_name is not None else
                                   'actor_only' if bridge_plan is None else 'bridge_prefix_then_source_tail'),
                    normalizer_sha256=actual_policy['normalizer_sha256'],
                    model_sha256=actual_policy['xml_sha256'])
            if bridge_plan is not None:
                # Composite-controller success is never a source Actor witness.
                r['attempts'][-1]['controller_kind']='composite_teacher'
                r['attempts'][-1]['source_only']=bool(bridge_plan[1][e])
            if label:r.update(label=1,witness=name if controller_provenance['actor_witness_eligible'] else None)
        timings.append(dict(policy=name,candidates=count,runtime_seconds=runtime_ready-stage_start,restore_seconds=restore_ready-runtime_ready,compile_and_rollout_seconds=rollout_ready-restore_ready,export_seconds=time.monotonic()-rollout_ready))
        write(output/'timings.json',timings)
        del initial
        if session is None:jax.clear_caches()
    for r in rows:
        if r.get('prefix_terminal'):r.update(label=r.get('prefix_label'),witness=spec['proposer'] if r.get('prefix_label')==1 and not (bridge_plan is not None or prefix_name is not None or spec.get('warmup_initializer') is not None) else None)
        elif r['label']!=1:r['label']=aggregate_labels(r['attempts'],spec['order'])
    write(output/'results.json',rows)
    import resource
    write(output/'status.json',dict(phase='completed',charged_interactions=charged,active_interactions=active_count,padding_interactions=charged-active_count,successes=sum(r['label']==1 for r in rows),wall_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def update(spec, output):
    mode=controller_mode(spec)
    if spec.get('explorer_backend')=='rsl_rl' and mode!='fixed_random':
        from .rsl_pulse import update_runtime
        return update_runtime(spec,output)
    if mode=='fixed_random':
        # Uniform random actions have no trainable exploration distribution.
        # Carry bytes forward exactly, including the collection's advanced RNG.
        source=Path(spec['collection']);output=Path(output)
        output.mkdir(parents=True,exist_ok=False)
        feedback=read(spec['feedback'])
        with np.load(source/'prefixes.npz') as tape:
            pulse_mask=tape['mask'].astype(bool)
        eligible=np.asarray(feedback['eligible'],bool)
        reward=np.zeros(pulse_mask.shape,np.float32)
        for lane in range(pulse_mask.shape[1]):
            ticks=np.flatnonzero(pulse_mask[:,lane]&eligible[lane])
            if len(ticks):reward[ticks[-1],lane]=feedback['rewards'][lane]
        (output/'state.msgpack').write_bytes((source/'update_state.msgpack').read_bytes())
        np.savez_compressed(output/'learning.npz',reward=reward,advantages=np.zeros_like(reward),
                            returns=np.zeros_like(reward),mask=np.zeros_like(pulse_mask),pulse_mask=pulse_mask)
        write(output/'optimizer_updates.json',[]);write(output/'hyperparameters.json',spec)
        write(output/'metrics.json',dict(optimizer_updates=0,post_update_kl=0.,
            post_update_clip_fraction=0.,value_explained_variance=None,total_loss=0.,actor_loss=0.,
            critic_loss=0.,entropy=0.,gradient_norm=0.,reward_components=feedback['component_sums'],
            reward=float(np.sum(feedback['rewards'])),eligible_episodes=int(eligible.sum()),
            effective_training_samples=0,update_skipped=True,skip_reason='fixed_random_controller',
            controller_mode=mode,losses_evaluated=False))
        write(output/'status.json',dict(phase='completed',charged_interactions=0));return
    import jax
    import jax.numpy as jp
    import optax
    from flax import serialization
    from .exploration_training import episode_advantages
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    env,payload,member,net,opt,state=networks(spec)
    source=Path(spec['collection']);state=serialization.from_bytes(state,(source/'update_state.msgpack').read_bytes())
    tape=np.load(source/'prefixes.npz');feedback=read(spec['feedback'])
    from .generative_bridge.explorer_admission import admit_update,training_mask
    spec=admit_update(spec,source,state,tape,payload.observation_normalizer)
    eligible=np.array(feedback['eligible'],bool);mask=training_mask(spec,tape,feedback)
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
