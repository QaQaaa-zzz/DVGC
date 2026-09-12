"""Frozen-base residual PPO with reached-state continuation credit.

This privileged exploration policy is a simulation instrument, not a deployable
replacement Actor. No mutable novelty callbacks occur inside transformed JAX.
"""
from pathlib import Path
import json,time,hashlib,sys
import numpy as np


def select_candidate_ticks(data, episode_index, cells, ledger_cells, spec):
    """Select existing eligible arrival frames; never synthesize a state.

    Legacy mode keeps first unique x bins. Opt-in stratification allocates the
    bounded quota round-robin over present phases and spans each phase's bins.
    """
    mode = spec.get('candidate_selection', 'first_bins')
    if mode not in ('first_bins', 'phase_stratified_v1'):
        raise ValueError('unknown candidate selection')
    limit = spec['max_candidates_per_episode']
    if type(limit) is not int or limit <= 0:
        raise ValueError('positive candidate limit required')
    spacing = spec['candidate_spacing']
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError('positive finite candidate spacing required')
    e = episode_index
    seen = set(ledger_cells)
    used_bins = set()
    eligible = {0: [], 1: []}
    ordered = []
    for tick in np.flatnonzero(data['mask'][:, e]):
        x = float(data['qpos'][tick, e, 0])
        phase = int(data['phase'][tick, e])
        semantic = ((phase == 0 and not bool(data['snap/up/apex_seen'][tick, e])) or
                    (phase == 1 and data['qvel'][tick, e, 2] < 0 and not bool(data['snap/down/valid_contact_seen'][tick, e])))
        if (data['terminal'][tick, e] or not data['finite'][tick, e] or not semantic
            or not spec['candidate_min_x'] <= x <= spec['candidate_max_x'] or cells[tick] in seen):
            continue
        x_bin = int(np.floor(x / spacing))
        key = x_bin if mode == 'first_bins' else (phase, x_bin)
        if key in used_bins:
            continue
        used_bins.add(key)
        eligible[phase].append(int(tick))
        ordered.append(int(tick))
    if mode == 'first_bins':
        return ordered[:limit]
    quotas = {phase: 0 for phase in eligible}
    for _ in range(min(limit, len(ordered))):
        available = [phase for phase in eligible if quotas[phase] < len(eligible[phase])]
        phase = min(available, key=lambda phase: (quotas[phase], phase))
        quotas[phase] += 1
    selected = []
    for phase, count in quotas.items():
        if count:
            indices = np.linspace(0, len(eligible[phase])-1, count, dtype=int)
            selected.extend(eligible[phase][index] for index in indices)
    return sorted(selected)


def episode_advantages(rewards, values, mask, *, gamma, lam):
    rewards,values,mask=map(np.asarray,(rewards,values,mask))
    if rewards.shape!=values.shape or rewards.shape!=mask.shape or rewards.ndim!=2:
        raise ValueError('time x episode arrays required')
    advantage=np.zeros_like(rewards,dtype=np.float32);carry=np.zeros(rewards.shape[1],np.float32)
    for t in range(len(rewards)-1,-1,-1):
        next_mask=mask[t+1] if t+1<len(rewards) else 0.
        next_value=values[t+1]*next_mask if t+1<len(rewards) else 0.
        delta=rewards[t]+gamma*next_value-values[t]
        carry=(delta+gamma*lam*next_mask*carry)*mask[t]
        advantage[t]=carry
    return advantage,(advantage+values)*mask


def _write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initial_ledger(baseline, episodes, *, arrival_mode, baseline_mode='collect'):
    """A matched arm can reuse the first arm's exact per-pi baseline artifact."""
    if baseline_mode not in ('collect','locked'):
        raise ValueError('unknown baseline mode')
    cells=set(baseline['cells'])
    if baseline_mode=='collect':
        for episode in episodes:
            if arrival_mode or (episode['success'] and not episode['physical_failure'] and episode['completed']):
                cells.update(episode['cells'])
    return dict(policy_sha256=baseline['policy_sha256'],cells=sorted(cells))


def run(spec_path,output):
    import jax,jax.numpy as jp,optax
    from brax.training.acme import running_statistics
    import brax,flax,mujoco_playground
    from .handoff_bank import pytree_sha256
    from flax import serialization
    from .exploration_network import make_exploration_network_factory,initialize_residual_actor,compose_residual_action,validate_residual_delta_limit
    from .exploration_reward import continuation_reward_batch,arrival_reward_batch
    from .exploration_continuation import snapshot_arrays,snapshot_from_arrays,FrozenSuffixEvaluator
    from .unified_envelope_snapshot import physical_state_sha256,snapshot_context_sha256
    from .evidence_integrity import canonical_sha256
    from .ppo import make_checkpoint_policy
    from .probe_bank import load_probe_bank
    from .unified_formal import build_unified_formal_environment
    from .checkpoint import load_checkpoint
    from .unified_training import checkpoint_identity
    from .iterative_probe_training import first_landing_state
    from .continuation.device_rollout import _shared_warp,prepare_parallel_worlds
    from .analysis.capability_tube import physical_coordinates_from_arrays,quantize_coordinates,ROOT_GEOMETRY_FIELDS,_cell_id,resolution_contract
    from .acquisition.causal_jump import physical_state_sha256_from_state
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    spec=json.loads(Path(spec_path).read_text());start=time.monotonic()
    required=('bank','proposer','num_envs','horizon','batches','epochs','minibatch_size','seed','learning_rate','clip','gamma','gae_lambda','value_coefficient','entropy_coefficient','max_grad_norm','evaluation_episodes','baseline_cells','jump_start_state_sha256')
    if spec.get('schema')!='jit_frozen_policy_residual_ppo_v1':raise ValueError('explicit frozen-policy residual PPO schema required; legacy full-action trial is not residual exploration')
    for k in ['delta_limit','max_candidates_per_episode','candidate_min_x','candidate_max_x','candidate_spacing','evaluator_order','suffix_horizon','suffix_budget']:
        if k not in spec:raise ValueError('missing residual/suffix contract: '+k)
    limits=validate_residual_delta_limit(spec['delta_limit'])
    arrival_mode=spec.get('reward_mode','witnessed_novelty_v1')=='arrival_novelty_v1'
    random_control=spec.get('controller_mode','learned_residual')=='fixed_random'
    if spec.get('reward_mode','witnessed_novelty_v1') not in ('witnessed_novelty_v1','arrival_novelty_v1'):raise ValueError('unknown reward mode')
    if spec.get('controller_mode','learned_residual') not in ('learned_residual','fixed_random'):raise ValueError('unknown controller mode')
    if spec.get('candidate_selection','first_bins') not in ('first_bins','phase_stratified_v1'):raise ValueError('unknown candidate selection')
    if arrival_mode and spec['suffix_budget']!=0:raise ValueError('arrival mode defers suffix evaluation to separately budgeted stage')
    if type(spec['max_candidates_per_episode']) is not int or spec['max_candidates_per_episode']<=0 or not 0.01<=spec['candidate_spacing']<=0.1 or not 2.5<=spec['candidate_min_x']<spec['candidate_max_x']<=8.:raise ValueError('invalid candidate sampling budget')
    if any(k not in spec for k in required):raise ValueError('incomplete declared training configuration')
    if spec['horizon']!=400 or any(type(spec[k]) is not int or spec[k]<=0 for k in ['num_envs','batches','epochs','minibatch_size','evaluation_episodes']):raise ValueError('invalid budget/full episode horizon')
    if spec['evaluation_episodes']!=spec['num_envs']:raise ValueError('matched batch evaluation required for shared compiled collector')
    bank=load_probe_bank(Path(spec['bank']));member=next(m for m in bank['members'] if m['name']==spec['proposer']);record=member['policy']
    if 'proposer' not in member['roles']:raise ValueError('proposer role required')
    baseline=json.loads(Path(spec['baseline_cells']).read_text())
    if baseline['policy_sha256']!=record['actor_sha256']:raise ValueError('per-policy baseline identity mismatch')
    sources={str(p.resolve()):_sha(p) for base in [Path(__file__).parent,Path(__file__).parents[2]/'cli'] for p in base.rglob('*.py')}
    for package in [brax,flax,mujoco_playground]:
        for file in Path(package.__file__).parent.rglob('*.py'):sources[str(file.resolve())]=_sha(file)
    repo=Path(__file__).resolve().parents[3]
    for folder in [repo/'JIT/configs',repo/'assets']:
        for file in folder.rglob('*'):
            if file.is_file():sources[str(file.resolve())]=_sha(file)
    inputs={str(Path(p).resolve()):_sha(p) for p in [spec_path,spec['bank'],spec['baseline_cells'],member['frozen_policy'],record['formal_config'],Path(record['checkpoint'])/'payload.pkl',Path(record['checkpoint'])/'identity.json']}
    for bank_member in bank['members']:
        policy_record=bank_member['policy']
        for file in [bank_member['frozen_policy'],policy_record['formal_config'],str(Path(policy_record['checkpoint'])/'payload.pkl'),str(Path(policy_record['checkpoint'])/'identity.json')]:inputs[str(Path(file).resolve())]=_sha(file)
    scheduled=spec['num_envs']*spec['horizon']*(spec['batches']+2)
    _write(output/'declaration.json',dict(spec=spec,input_files=inputs,source_files=sources,maximum_interactions=scheduled+spec['suffix_budget'],maximum_forward_interactions=scheduled,maximum_suffix_interactions=spec['suffix_budget'],maximum_attempts=1,role='train',final_test_used=False,resolution=resolution_contract(),normalizer='frozen_for_pilot',padding_charged=True,coverage_scope=('per frozen pi provisional arrivals; not verified envelope' if arrival_mode else 'per frozen pi; reached candidate with same-context frozen-bank suffix witness'),reward=('one per new arrival root cell; delayed evaluation kept separate' if arrival_mode else 'one per newly witnessed root cell at causal action tick; parent trajectory landing not required'),controller=('frozen pi plus fixed uniform random residual' if random_control else 'frozen pi plus bounded learned residual'),base_actor_sha256=record['actor_sha256']))
    def verify():
        for p,sha in {**sources,**inputs}.items():
            if _sha(p)!=sha:raise ValueError('training source/input drift: '+p)
    def status(phase,**kw):_write(output/'status.json',dict(phase=phase,wall_seconds=time.monotonic()-start,**kw))
    charged=0;active_total=0;history=[];suffix=None
    try:
        status('building_runtime')
        if jax.default_backend()!='gpu':raise RuntimeError('GPU backend required for declared simulation stage')
        config,_,env=build_unified_formal_environment(Path(record['formal_config']))
        payload=load_checkpoint(Path(record['checkpoint']),expected=checkpoint_identity(config,env))
        for name,value in [('actor_sha256',payload.actor_params),('normalizer_sha256',payload.observation_normalizer),('critic_sha256',payload.critic_params)]:
            if pytree_sha256(value)!=record[name]:raise ValueError('frozen policy payload identity drift: '+name)
        if env._bundle.xml_sha256!=record['xml_sha256']:raise ValueError('model identity drift')
        normalizer=payload.observation_normalizer
        base_policy=make_checkpoint_policy(env,payload,deterministic=True)
        net=make_exploration_network_factory()({'state':76,'privileged_state':106},4,preprocess_observations_fn=running_statistics.normalize)
        rng=jax.random.PRNGKey(spec['seed']);rng,vkey=jax.random.split(rng)
        rng,akey=jax.random.split(rng)
        params={'policy':initialize_residual_actor(net,akey),'value':net.value_network.init(vkey)}
        frozen_base_sha=pytree_sha256(payload.actor_params)
        suffix=FrozenSuffixEvaluator(spec['bank'],spec['evaluator_order'],spec['suffix_horizon'],output/'suffixes',spec['suffix_budget'])
        suffix_cache={}
        current_residual_checkpoint=None
        optimizer=optax.chain(optax.clip_by_global_norm(spec['max_grad_norm']),optax.adam(spec['learning_rate']));optstate=optimizer.init(params)
        distribution=net.parametric_action_distribution
        reset=jax.vmap(env._reset_jump_start_unified)
        step=jax.vmap(lambda s,a:first_landing_state(env.step(s,a)))
        count=spec['num_envs'];horizon=spec['horizon']
        @jax.jit
        def collect(params,key,deterministic):
            key,rkey=jax.random.split(key);initial=prepare_parallel_worlds(reset(jax.random.split(rkey,count)),env,count)
            def advance(carry,_):
                state,key,alive=carry;key,akey=jax.random.split(key)
                logits=net.policy_network.apply(normalizer,params['policy'],state.obs)
                raw=distribution.sample_no_postprocessing(logits,akey)
                sampled=distribution.postprocess(raw)
                normalized_delta=jp.where(deterministic,distribution.mode(logits),sampled)
                if random_control:
                    normalized_delta=jp.where(deterministic,jp.zeros_like(sampled),jax.random.uniform(akey,sampled.shape,minval=-1.,maxval=1.))
                base_action=base_policy(state.obs,akey)[0]
                action,requested_delta,effective_delta=compose_residual_action(base_action,normalized_delta,jp.asarray(limits))
                value=net.value_network.apply(normalizer,params['value'],state.obs)
                base_critic_value=net.value_network.apply(normalizer,payload.critic_params,state.obs)
                nxt=step(state,action)
                finite=jp.all(jp.isfinite(nxt.data.qpos),-1)&jp.all(jp.isfinite(nxt.data.qvel),-1)&jp.all(jp.isfinite(nxt.obs['privileged_state']),-1)&jp.all(jp.isfinite(action),-1)
                finite=finite&~nxt.info['parallel_capacity_exceeded']
                terminal=nxt.done.astype(bool)|~finite
                observed=dict(base_critic_value=base_critic_value,observation=state.obs['privileged_state'],base_action=base_action,normalized_delta=normalized_delta,requested_delta=requested_delta,effective_delta=effective_delta,action=action,raw_action=raw,log_prob=distribution.log_prob(logits,raw),value=value,mask=alive,qpos=nxt.data.qpos,qvel=nxt.data.qvel,phase=nxt.info['active_phase'],success=nxt.info['success'],physical_failure=nxt.info['physical_failure'],finite=finite,terminal=terminal,end_code=nxt.info['end_code'])
                observed.update({'snap/'+k:v for k,v in snapshot_arrays(nxt).items()})
                # Reset inactive worlds to the fixed start before their next padded
                # simulator slot; padding never contributes gradients or coverage.
                keep=alive&~terminal
                nxt=jax.tree_util.tree_map_with_path(lambda path,x,y:x if _shared_warp(path) else jp.where(keep.reshape((count,)+(1,)*(x.ndim-1)),x,y),nxt,initial)
                return (nxt,key,keep),observed
            _,data=jax.lax.scan(advance,(initial,key,jp.ones(count,dtype=bool)),None,length=horizon)
            return data,initial.data.qpos,initial.data.qvel
        @jax.jit
        def update(params,optstate,batch,key):
            def loss(p):
                obs={'privileged_state':batch['observation']}
                logits=net.policy_network.apply(normalizer,p['policy'],obs)
                logp=distribution.log_prob(logits,batch['raw_action'])
                ratio=jp.exp(logp-batch['log_prob']);adv=batch['advantage'];mask=batch['weight'];denom=jp.maximum(mask.sum(),1)
                policy=-jp.sum(mask*jp.minimum(ratio*adv,jp.clip(ratio,1-spec['clip'],1+spec['clip'])*adv))/denom
                value=net.value_network.apply(normalizer,p['value'],obs)
                vloss=.5*jp.sum(mask*(value-batch['return'])**2)/denom
                entropy=jp.sum(mask*distribution.entropy(logits,key))/denom
                total=policy+spec['value_coefficient']*vloss-spec['entropy_coefficient']*entropy
                return total,jp.array([policy,vloss,entropy])
            (lossval,parts),grads=jax.value_and_grad(loss,has_aux=True)(params)
            updates,optstate=optimizer.update(grads,optstate,params)
            return optax.apply_updates(params,updates),optstate,jp.concatenate([jp.array([lossval]),parts])
        def rollout(label,deterministic):
            nonlocal rng,charged,active_total
            verify();status('rollout_'+label,charged_interactions=charged+suffix.charged_interactions,forward_interactions=charged,suffix_interactions=suffix.charged_interactions)
            reserve=count*horizon;charged+=reserve
            _write(output/(label+'_reservation.json'),dict(maximum_interactions=reserve,charged_interactions=reserve))
            rng,key=jax.random.split(rng);t=time.monotonic()
            data,q0,v0=jax.device_get(collect(params,key,jp.asarray(deterministic)))
            from types import SimpleNamespace
            for q,v in zip(q0,v0):
                if physical_state_sha256_from_state(SimpleNamespace(data=SimpleNamespace(qpos=q,qvel=v)))!=spec['jump_start_state_sha256']:raise ValueError('fixed start identity drift')
            np.savez_compressed(output/(label+'_trajectories.npz'),**data,initial_qpos=q0,initial_qvel=v0)
            if np.any(data['mask']&~data['finite']):raise ValueError('nonfinite active trajectory; retained failure evidence')
            episodes=[]
            for e in range(count):
                indices=np.flatnonzero(data['mask'][:,e]);cells=[]
                for tick in indices:
                    co=physical_coordinates_from_arrays(data['qpos'][tick,e],data['qvel'][tick,e],bundle=env._bundle)
                    cells.append(_cell_id('upstream' if int(data['phase'][tick,e])==0 else 'downstream','root_geometry_v1',quantize_coordinates(co,ROOT_GEOMETRY_FIELDS)))
                last=indices[-1]
                episodes.append(dict(cells=cells,success=bool(data['success'][last,e]),physical_failure=bool(data['physical_failure'][last,e]),completed=bool(data['terminal'][last,e]),finite=True))
            active=int(data['mask'].sum());active_total+=active
            receipt=dict(episodes=episodes,active_interactions=active,scheduled_interactions=reserve,padding_interactions=reserve-active,wall_seconds=time.monotonic()-t)
            _write(output/(label+'_episodes.json'),receipt)
            return data,episodes,receipt
        def candidate_rewards(label,data,episodes,ledger):
            verify()
            candidates=[]
            controller=dict(base_actor_sha256=frozen_base_sha,residual_actor_sha256=pytree_sha256(params['policy']),normalizer_sha256=pytree_sha256(normalizer),action_composition=('clip(base+delta_limit*uniform[-1,1],-1,1)' if random_control else 'clip(base+delta_limit*tanh_sample,-1,1)'),delta_limit=list(limits))
            controller['controller_mode']=spec.get('controller_mode','learned_residual')
            controller['seed']=spec['seed']
            controller['residual_checkpoint']=str(current_residual_checkpoint)
            controller['residual_checkpoint_sha256']=_sha(current_residual_checkpoint)
            controller['controller_sha256']=canonical_sha256(controller)
            generator_record={**record,'actor_sha256':controller['controller_sha256'],'payload_sha256':controller['residual_checkpoint_sha256']}
            prefix_path=output/(label+'_trajectories.npz');prefix_sha=_sha(prefix_path)
            for e,episode in enumerate(episodes):
                for tick in select_candidate_ticks(data,e,episode['cells'],ledger['cells'],spec):
                    cell=episode['cells'][tick]
                    arrays={k[5:]:v[tick,e] for k,v in data.items() if k.startswith('snap/')}
                    snapshot=snapshot_from_arrays(arrays,env=env,record=generator_record,parent_trajectory=f'{label}/episode_{e}',parent_state_sha256=spec['jump_start_state_sha256'])
                    context=snapshot_context_sha256(snapshot)
                    status('suffix_'+label,charged_interactions=charged+suffix.charged_interactions,candidate_episode=e,candidate_tick=int(tick))
                    result=suffix_cache.get(context)
                    if result is None:
                        result=(suffix.defer(snapshot) if arrival_mode else suffix.evaluate(snapshot));suffix_cache[context]=result
                    if result['snapshot_context_sha256']!=context or result['state_sha256']!=physical_state_sha256(snapshot) or result['bank_sha256']!=bank['bank_sha256']:raise ValueError('suffix receipt candidate or bank identity mismatch')
                    if canonical_sha256({k:v for k,v in result.items() if k!='receipt_sha256'})!=result['receipt_sha256']:raise ValueError('suffix receipt hash mismatch')
                    provenance=dict(episode_index=e,tick=int(tick),cell=cell,state_sha256=physical_state_sha256(snapshot),context_sha256=context,label=result['label'],witness=result['witness'],suffix_receipt_sha256=result['receipt_sha256'],prefix_file=str(prefix_path),prefix_file_sha256=prefix_sha,controller=controller,generated_by_env_step_only=True,jump_start_state_sha256=spec['jump_start_state_sha256'])
                    provenance['snapshot_dir']=str(suffix.output/context/'snapshot')
                    provenance['base_critic_value_before_action']=float(data['base_critic_value'][tick,e])
                    provenance['reward_mode']=spec.get('reward_mode','witnessed_novelty_v1')
                    provenance['receipt_sha256']=canonical_sha256(provenance)
                    candidates.append(provenance)
                    _write(output/(label+'_candidates.json'),candidates)
                    if result['label'] is None and not arrival_mode:raise RuntimeError('unknown suffix result retained; no negative conversion or PPO update')
            _write(output/(label+'_candidates.json'),candidates)
            return (arrival_reward_batch if arrival_mode else continuation_reward_batch)(ledger,candidates,shape=data['mask'].shape,expected_policy_sha256=record['actor_sha256'])
        def checkpoint(label):
            nonlocal current_residual_checkpoint
            verify()
            if pytree_sha256(payload.actor_params)!=frozen_base_sha or pytree_sha256(normalizer)!=record['normalizer_sha256']:raise ValueError('frozen base policy/normalizer changed')
            p=output/'checkpoints'/label;p.mkdir(parents=True,exist_ok=False)
            state=dict(params=params,optimizer=optstate,normalizer=normalizer,rng=rng)
            (p/'state.msgpack').write_bytes(serialization.to_bytes(state))
            current_residual_checkpoint=p/'state.msgpack'
            _write(p/'identity.json',dict(schema='jit_frozen_policy_residual_ppo_checkpoint_v1',state_sha256=_sha(p/'state.msgpack'),base_actor_sha256=record['actor_sha256'],actor_input='privileged_state106',output_kind='bounded_delta_added_to_frozen_pi',delta_limit=list(limits),action_order=['steer','rear_wheel_drive','hip','knee'],ledger=ledger,charged_interactions=charged+suffix.charged_interactions,suffix_interactions=suffix.charged_interactions,active_interactions=active_total,spec=spec))
        data,episodes,receipt=rollout('initial',True)
        ledger=initial_ledger(baseline,episodes,arrival_mode=arrival_mode,baseline_mode=spec.get('baseline_mode','collect'))
        if not np.array_equal(data['action'],data['base_action']):raise ValueError('initial zero-mean residual does not reproduce frozen base action')
        _write(output/'baseline.json',ledger);checkpoint('initial')
        for iteration in range(1,spec['batches']+1):
            label=f'batch_{iteration:04d}';data,episodes,receipt=rollout(label,False)
            rewards,ledger,evidence=candidate_rewards(label,data,episodes,ledger)
            _write(output/(label+'_reward.json'),evidence)
            rewards=rewards.astype(np.float32)
            adv,returns=episode_advantages(rewards,data['value'],data['mask'],gamma=spec['gamma'],lam=spec['gae_lambda'])
            mask=data['mask'].astype(bool);a=adv[mask];a=(a-a.mean())/(a.std()+1e-8)
            batch=dict(observation=data['observation'][mask],raw_action=data['raw_action'][mask],log_prob=data['log_prob'][mask],advantage=a,**{'return':returns[mask]})
            np.savez_compressed(output/(label+'_learning.npz'),rewards=rewards,advantages=adv,returns=returns)
            size=len(a);mb=spec['minibatch_size'];losses=[]
            host_rng=np.random.default_rng(np.random.SeedSequence([spec['seed'],iteration]))
            for epoch in range(0 if random_control else spec['epochs']):
                order=host_rng.permutation(size)
                for lo in range(0,size,mb):
                    ix=order[lo:lo+mb];actual=len(ix);ix=np.pad(ix,(0,mb-actual),mode='wrap')
                    mini={k:jp.asarray(v[ix]) for k,v in batch.items()};mini['weight']=jp.asarray(np.arange(mb)<actual,dtype=jp.float32)
                    rng,key=jax.random.split(rng);params,optstate,loss=update(params,optstate,mini,key)
                    losses.append(np.asarray(loss))
            if not np.isfinite(losses).all() or not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(params)):raise ValueError('nonfinite PPO update')
            row=dict(batch=iteration,new_cells=float(np.sum(rewards)),cumulative_cells=len(ledger['cells']),successes=sum(e['success'] and not e['physical_failure'] and e['completed'] and e['finite'] for e in episodes),episodes=count,active_interactions=receipt['active_interactions'],charged_interactions=charged+suffix.charged_interactions,suffix_interactions=suffix.charged_interactions,mean_losses=np.mean(losses,axis=0).tolist() if losses else [],controller_mode=spec.get('controller_mode','learned_residual'),reward_mode=spec.get('reward_mode','witnessed_novelty_v1'),optimizer_updates=len(losses),wall_seconds=time.monotonic()-start)
            history.append(row);_write(output/'training_metrics.json',history);print(json.dumps(row),flush=True)
            checkpoint(label)
        data,episodes,receipt=rollout('final',not random_control)
        diagnostic_credits,_,evidence=candidate_rewards('final',data,episodes,ledger)
        _write(output/'final_diagnostic_reward.json',evidence)
        verify();status('completed',charged_interactions=charged+suffix.charged_interactions,forward_interactions=charged,suffix_interactions=suffix.charged_interactions,base_actor_unchanged=pytree_sha256(payload.actor_params)==frozen_base_sha,active_interactions=active_total,padding_interactions=charged-active_total,training_scheduled_interactions=count*horizon*spec['batches'],final_successes=sum(e['success'] and not e['physical_failure'] and e['completed'] and e['finite'] for e in episodes),initial_baseline_cells=len(json.loads((output/'baseline.json').read_text())['cells']),training_new_cells=sum(x['new_cells'] for x in history),final_diagnostic_new_cells=float(np.sum(diagnostic_credits)),independent_repetitions=False,final_test_used=False)
    except BaseException as exc:
        status('error',error=f'{type(exc).__name__}: {exc}',charged_interactions=charged+(suffix.charged_interactions if suffix is not None else 0),forward_interactions=charged,suffix_interactions=suffix.charged_interactions if suffix is not None else 0,active_interactions=active_total,no_automatic_retry=True)
        raise
