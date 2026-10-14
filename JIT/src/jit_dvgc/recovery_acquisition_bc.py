"""Recovery expert Actor-only BC, with no old-skill data or proposal veto.

Updates is an absolute target count, including on resume. Checkpoint callbacks
receive (absolute_update, (normalizer, actor, inference_critic)); True stops.
BC continuation restores Adam and the sampling RNG. PPO must start new Adam.
"""
import hashlib
import json
import pickle
import time
from pathlib import Path
import numpy as np
from .generative_bridge.protocol import atomic_json


def recovery_sample_weights(manifest, root_ids=None):
    """Equal root mass, half prefix/tail mass; empty segments renormalize."""
    roots=np.asarray(manifest.get('sample_roots', []))
    origins=np.asarray(manifest.get('sample_origins', []))
    count=manifest.get('count')
    if roots.shape!=(count,) or origins.shape!=(count,) or not count:
        raise ValueError('nonempty aligned sample_roots/sample_origins required')
    if any(not isinstance(x,str) or not x for x in roots.tolist()):
        raise ValueError('nonempty root identities required')
    if not set(origins.tolist()) <= {'bridge_prefix','source_tail'}:
        raise ValueError('unknown demonstration segment')
    selected=set(roots.tolist()) if root_ids is None else set(root_ids)
    if not selected or not selected<=set(roots.tolist()):
        raise ValueError('root filter contains missing or no roots')
    weights=np.zeros(count,np.float64)
    for root in sorted(selected):
        segments=[s for s in ('bridge_prefix','source_tail') if np.any((roots==root)&(origins==s))]
        for segment in segments:
            mask=(roots==root)&(origins==segment)
            weights[mask]=1./(len(selected)*len(segments)*mask.sum())
    return weights


def warmup_recovery_actor(network, initializer, demo_manifest, output, *, seed=0,
        updates=2000, batch_size=256, learning_rate=1e-4, checkpoint_steps=None,
        resume_checkpoint=None, checkpoint_callback=None, metrics_callback=None,
        root_ids=None):
    import jax
    import jax.numpy as jp
    import optax
    from .generative_bridge.student import load_optional_demo
    from .generative_bridge.contracts import file_sha
    from .handoff_bank import pytree_sha256
    if type(updates) is not int or updates<0:raise ValueError('nonnegative update target required')
    if type(batch_size) is not int or batch_size<=0:raise ValueError('positive batch size required')
    if not np.isfinite(learning_rate) or learning_rate<=0:raise ValueError('positive finite learning rate required')
    if initializer is None or len(initializer)!=3:raise ValueError('explicit inference initializer tuple required')
    schedule=([s for s in (0,100,500,1000,2000) if s<=updates] if checkpoint_steps is None else list(checkpoint_steps))
    if any(type(s) is not int or s<0 or s>updates for s in schedule) or len(set(schedule))!=len(schedule):
        raise ValueError('checkpoint schedule must contain unique absolute steps within requested budget')
    def finite(tree):
        return all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(tree))
    if not finite(initializer):raise FloatingPointError('nonfinite initializer')
    online=demo_manifest.get('schema')=='frontierbridge_online_labels_v1'
    if online:
        from .policy_distillation import load_dataset
        if file_sha(demo_manifest['path'])!=demo_manifest['sha256']:raise ValueError('online data hash mismatch')
        data=load_dataset(demo_manifest['path'])
        if len(data[0])!=demo_manifest['count'] or data[0].shape[1:]!=(76,):raise ValueError('online data shape mismatch')
    else:data=load_optional_demo(demo_manifest)
    if data is None:raise ValueError('recovery BC requires nonempty demonstration data')
    if online:
        if root_ids is not None:raise ValueError('online labels use explicitly declared sampling weights')
        weights=np.asarray(demo_manifest['sampling_weights'],np.float64)
        if weights.shape!=(demo_manifest['count'],) or not np.isfinite(weights).all() or np.any(weights<0) or not np.isclose(weights.sum(),1):raise ValueError('invalid online sampling weights')
        if set(demo_manifest['sample_origins'])!={'student_visited_query'}:raise ValueError('online observations must be real student queries')
    else:weights=recovery_sample_weights(demo_manifest,root_ids)
    normalizer,actor,critic=initializer
    fixed_hash=pytree_sha256(normalizer);critic_hash=pytree_sha256(critic)
    # Bind all source metadata as well as actual loaded arrays; schedule and target
    # budget can extend on resume without changing the learner contract.
    manifest_hash=hashlib.sha256(json.dumps(demo_manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    identity=dict(schema='jit_recovery_bc_contract_v1',seed=int(seed),batch_size=batch_size,
        learning_rate=float(learning_rate),grad_clip_norm=1.,initializer_sha256=pytree_sha256(initializer),
        normalizer_sha256=fixed_hash,critic_sha256=critic_hash,manifest_sha256=manifest_hash,
        data_sha256=pytree_sha256(data),weights_sha256=pytree_sha256(weights),
        selected_roots=sorted(set(np.asarray(demo_manifest['sample_roots'])[weights>0].tolist())),
        action_target='normalized_action_executed',prediction='parametric_action_distribution.mode',
        sampling=('explicit_online_family_category_root_weights' if online else 'root_equal/segment_equal/frame_uniform;split_rng_each_update'),phase='S_BC' if online else 'BC')
    identity_hash=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(learning_rate))
    state=optimizer.init(actor);rng=jax.random.PRNGKey(seed);completed=0
    roots=np.asarray(demo_manifest['sample_roots']);origins=np.asarray(demo_manifest['sample_origins'])
    counts={f'{r}/{s}':0 for r,s in sorted(set(zip(roots.tolist(),origins.tolist())))}
    if resume_checkpoint is not None:
        source=Path(resume_checkpoint);receipt=json.loads(source.with_suffix('.json').read_text())
        if file_sha(source)!=receipt['sha256']:raise ValueError('resume checkpoint hash mismatch')
        with source.open('rb') as stream:restored=pickle.load(stream)
        if restored['identity']!=identity or restored['contract_sha256']!=identity_hash:
            raise ValueError('resume contract differs')
        normalizer,actor,critic=(restored[k] for k in ('normalizer','actor','critic'))
        state=restored['optimizer_state'];rng=jp.asarray(restored['rng'])
        completed=restored['completed_supervised_updates'];counts=restored['sample_counts']
        if type(completed) is not int or not 0<=completed<=updates:raise ValueError('resume update counter outside target')
        if not finite((normalizer,actor,critic,state,rng)):raise FloatingPointError('nonfinite restored learner')
        for name,tree in [('actor',actor),('optimizer',state),('normalizer',normalizer),('critic',critic),('rng',rng)]:
            if pytree_sha256(tree)!=receipt[name+'_sha256']:raise ValueError('resume learner identity mismatch')
        if pytree_sha256(normalizer)!=fixed_hash or pytree_sha256(critic)!=critic_hash:raise ValueError('frozen state changed')
        if int(state[1][0].count)!=completed:raise ValueError('resume Adam/update counter mismatch')
        if set(counts)!={f'{r}/{s}' for r,s in zip(roots.tolist(),origins.tolist())} or sum(counts.values())!=completed*batch_size:
            raise ValueError('resume sample counters mismatch')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    report=dict(schema='jit_recovery_bc_v1',identity=identity,contract_sha256=identity_hash,
        requested_updates=updates,completed_updates=completed,environment_interactions=0,status='running',
        available_roots=sorted(set(roots.tolist())),selected_roots=identity['selected_roots'],
        normalizer_sha256=fixed_hash,critic_sha256=critic_hash,sample_counts=counts,checkpoints=[])
    def save():
        report['completed_updates']=completed
        payload=dict(identity=identity,contract_sha256=identity_hash,normalizer=normalizer,actor=actor,critic=critic,
            optimizer_state=state,rng=rng,completed_supervised_updates=completed,sample_counts=counts)
        learner=output/f'learner_update_{completed:04d}.pkl';inference=output/f'update_{completed:04d}.pkl'
        with learner.open('xb') as f:pickle.dump(jax.device_get(payload),f)
        with inference.open('xb') as f:pickle.dump(jax.device_get((normalizer,actor,critic)),f)
        receipt=dict(path=str(learner.resolve()),sha256=file_sha(learner),completed_supervised_updates=completed,
            contract_sha256=identity_hash,inference_path=str(inference.resolve()),inference_sha256=file_sha(inference))
        for name,tree in [('actor',actor),('optimizer',state),('normalizer',normalizer),('critic',critic),('rng',rng)]:
            receipt[name+'_sha256']=pytree_sha256(tree)
        atomic_json(learner.with_suffix('.json'),receipt)
        atomic_json(output/'last_completed.json',receipt)
        report['checkpoints'].append(dict(update=completed,**receipt))
        atomic_json(output/'warmup_status.json',report)
        return bool(checkpoint_callback(completed,(normalizer,actor,critic))) if checkpoint_callback else False
    obs,targets=map(jp.asarray,data[:2]);probs=jp.asarray(weights)
    @jax.jit
    def step(params,opt_state,key):
        next_key,sample_key=jax.random.split(key)
        ids=jax.random.choice(sample_key,len(obs),(batch_size,),p=probs)
        def loss(p):
            prediction=network.parametric_action_distribution.mode(network.policy_network.apply(normalizer,p,{'state':obs[ids]}))
            return jp.mean(jp.square(prediction-targets[ids]))
        value,grad=jax.value_and_grad(loss)(params)
        delta,new_state=optimizer.update(grad,opt_state,params)
        candidate=optax.apply_updates(params,delta)
        ok=jp.all(jp.stack([jp.all(jp.isfinite(x)) for x in jax.tree.leaves((value,grad,delta,candidate,new_state))]))
        return candidate,new_state,next_key,ids,value,optax.global_norm(grad),optax.global_norm(delta),ok
    started=time.monotonic()
    try:
        if save():report['status']='stopped_by_callback';return normalizer,actor,critic
        with (output/'metrics.jsonl').open('x') as stream:
            while completed<updates:
                candidate,new_state,next_rng,ids,loss,norm,delta,ok=step(actor,state,rng)
                indices=np.asarray(ids)
                if not bool(ok):
                    np.savez(output/'nonfinite_batch.npz',indices=indices,observations=np.asarray(obs)[indices],targets=np.asarray(targets)[indices],rng=np.asarray(rng),absolute_update=completed+1)
                    raise FloatingPointError('nonfinite recovery BC proposal rejected atomically')
                actor,state,rng=candidate,new_state,next_rng;completed+=1
                for i in indices:counts[f'{roots[i]}/{origins[i]}']+=1
                row=dict(update=completed,loss=float(loss),demo=float(loss),actor_grad_norm=float(norm),actor_delta_norm=float(delta),sample_counts=dict(counts))
                stream.write(json.dumps(row)+'\n');stream.flush()
                report['completed_updates']=completed
                if metrics_callback:metrics_callback(row)
                if completed in schedule or completed==updates:
                    if save():report['status']='stopped_by_callback';break
            else:report['status']='completed'
        if pytree_sha256(normalizer)!=fixed_hash or pytree_sha256(critic)!=critic_hash:raise ValueError('frozen state changed')
    except BaseException as exc:
        report.update(status='error',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        report['completed_updates']=completed;report['wall_seconds']=time.monotonic()-started
        atomic_json(output/'warmup_status.json',report)
    return normalizer,actor,critic
