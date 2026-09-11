"""Per-frozen-policy coverage PPO with complete episodes and auditable host credit.

This privileged exploration policy is a simulation instrument, not a deployable
replacement Actor. No mutable novelty callbacks occur inside transformed JAX.
"""
from pathlib import Path
import json,time,hashlib,sys
import numpy as np


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


def run(spec_path,output):
    import jax,jax.numpy as jp,optax
    from brax.training.acme import running_statistics
    import brax,flax,mujoco_playground
    from .handoff_bank import pytree_sha256
    from flax import serialization
    from .exploration_network import make_exploration_network_factory,widen_actor_warm_start
    from .exploration_reward import reward_batch
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
    scheduled=spec['num_envs']*spec['horizon']*(spec['batches']+2)
    _write(output/'declaration.json',dict(spec=spec,input_files=inputs,source_files=sources,maximum_interactions=scheduled,maximum_attempts=1,role='train',final_test_used=False,resolution=resolution_contract(),normalizer='frozen_for_pilot',padding_charged=True,coverage_scope='per frozen pi; successful complete explorer trajectories',reward='one per newly credited root cell, batch duplicates shared',base_actor_sha256=record['actor_sha256']))
    def verify():
        for p,sha in {**sources,**inputs}.items():
            if _sha(p)!=sha:raise ValueError('training source/input drift: '+p)
    def status(phase,**kw):_write(output/'status.json',dict(phase=phase,wall_seconds=time.monotonic()-start,**kw))
    charged=0;active_total=0;history=[]
    try:
        status('building_runtime')
        if jax.default_backend()!='gpu':raise RuntimeError('GPU backend required for declared simulation stage')
        config,_,env=build_unified_formal_environment(Path(record['formal_config']))
        payload=load_checkpoint(Path(record['checkpoint']),expected=checkpoint_identity(config,env))
        for name,value in [('actor_sha256',payload.actor_params),('normalizer_sha256',payload.observation_normalizer),('critic_sha256',payload.critic_params)]:
            if pytree_sha256(value)!=record[name]:raise ValueError('frozen policy payload identity drift: '+name)
        if env._bundle.xml_sha256!=record['xml_sha256']:raise ValueError('model identity drift')
        normalizer,actor=widen_actor_warm_start(payload.observation_normalizer,payload.actor_params)
        net=make_exploration_network_factory()({'state':76,'privileged_state':106},4,preprocess_observations_fn=running_statistics.normalize)
        rng=jax.random.PRNGKey(spec['seed']);rng,vkey=jax.random.split(rng)
        params={'policy':actor,'value':net.value_network.init(vkey)}
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
                action=jp.where(deterministic,distribution.mode(logits),sampled)
                value=net.value_network.apply(normalizer,params['value'],state.obs)
                nxt=step(state,action)
                finite=jp.all(jp.isfinite(nxt.data.qpos),-1)&jp.all(jp.isfinite(nxt.data.qvel),-1)&jp.all(jp.isfinite(nxt.obs['privileged_state']),-1)&jp.all(jp.isfinite(action),-1)
                finite=finite&~nxt.info['parallel_capacity_exceeded']
                terminal=nxt.done.astype(bool)|~finite
                observed=dict(observation=state.obs['privileged_state'],action=action,raw_action=raw,log_prob=distribution.log_prob(logits,raw),value=value,mask=alive,qpos=nxt.data.qpos,qvel=nxt.data.qvel,phase=nxt.info['active_phase'],success=nxt.info['success'],physical_failure=nxt.info['physical_failure'],finite=finite,terminal=terminal,end_code=nxt.info['end_code'])
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
            verify();status('rollout_'+label,charged_interactions=charged)
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
        def checkpoint(label):
            verify();p=output/'checkpoints'/label;p.mkdir(parents=True,exist_ok=False)
            state=dict(params=params,optimizer=optstate,normalizer=normalizer,rng=rng)
            (p/'state.msgpack').write_bytes(serialization.to_bytes(state))
            _write(p/'identity.json',dict(schema='jit_privileged_coverage_ppo_checkpoint_v1',state_sha256=_sha(p/'state.msgpack'),base_actor_sha256=record['actor_sha256'],actor_input='privileged_state106',action_order=['steer','rear_wheel_drive','hip','knee'],ledger=ledger,charged_interactions=charged,active_interactions=active_total,spec=spec))
        data,episodes,receipt=rollout('initial',True)
        ledger={'policy_sha256':record['actor_sha256'],'cells':sorted(set(baseline['cells']).union(*(set(e['cells']) for e in episodes if e['success'] and not e['physical_failure'] and e['completed'])))}
        _write(output/'baseline.json',ledger);checkpoint('initial')
        for iteration in range(1,spec['batches']+1):
            label=f'batch_{iteration:04d}';data,episodes,receipt=rollout(label,False)
            credits,ledger,evidence=reward_batch(ledger,episodes,expected_policy_sha256=record['actor_sha256'])
            _write(output/(label+'_reward.json'),evidence)
            rewards=np.zeros_like(data['value'])
            for e,credit in enumerate(credits):rewards[np.flatnonzero(data['mask'][:,e])[-1],e]=credit
            adv,returns=episode_advantages(rewards,data['value'],data['mask'],gamma=spec['gamma'],lam=spec['gae_lambda'])
            mask=data['mask'].astype(bool);a=adv[mask];a=(a-a.mean())/(a.std()+1e-8)
            batch=dict(observation=data['observation'][mask],raw_action=data['raw_action'][mask],log_prob=data['log_prob'][mask],advantage=a,**{'return':returns[mask]})
            np.savez_compressed(output/(label+'_learning.npz'),rewards=rewards,advantages=adv,returns=returns)
            size=len(a);mb=spec['minibatch_size'];losses=[]
            host_rng=np.random.default_rng(np.random.SeedSequence([spec['seed'],iteration]))
            for epoch in range(spec['epochs']):
                order=host_rng.permutation(size)
                for lo in range(0,size,mb):
                    ix=order[lo:lo+mb];actual=len(ix);ix=np.pad(ix,(0,mb-actual),mode='wrap')
                    mini={k:jp.asarray(v[ix]) for k,v in batch.items()};mini['weight']=jp.asarray(np.arange(mb)<actual,dtype=jp.float32)
                    rng,key=jax.random.split(rng);params,optstate,loss=update(params,optstate,mini,key)
                    losses.append(np.asarray(loss))
            if not np.isfinite(losses).all() or not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(params)):raise ValueError('nonfinite PPO update')
            row=dict(batch=iteration,new_cells=float(np.sum(credits)),cumulative_cells=len(ledger['cells']),successes=sum(e['success'] and not e['physical_failure'] and e['completed'] and e['finite'] for e in episodes),episodes=count,active_interactions=receipt['active_interactions'],charged_interactions=charged,mean_losses=np.mean(losses,axis=0).tolist(),optimizer_updates=len(losses),wall_seconds=time.monotonic()-start)
            history.append(row);_write(output/'training_metrics.json',history);print(json.dumps(row),flush=True)
            checkpoint(label)
        data,episodes,receipt=rollout('final',True)
        diagnostic_credits,_,evidence=reward_batch(ledger,episodes,expected_policy_sha256=record['actor_sha256'])
        _write(output/'final_diagnostic_reward.json',evidence)
        verify();status('completed',charged_interactions=charged,active_interactions=active_total,padding_interactions=charged-active_total,training_scheduled_interactions=count*horizon*spec['batches'],final_successes=sum(e['success'] and not e['physical_failure'] and e['completed'] and e['finite'] for e in episodes),initial_baseline_cells=len(json.loads((output/'baseline.json').read_text())['cells']),training_new_cells=sum(x['new_cells'] for x in history),final_diagnostic_new_cells=float(sum(diagnostic_credits)),independent_repetitions=False,final_test_used=False)
    except BaseException as exc:
        status('error',error=f'{type(exc).__name__}: {exc}',charged_interactions=charged,active_interactions=active_total,no_automatic_retry=True)
        raise
