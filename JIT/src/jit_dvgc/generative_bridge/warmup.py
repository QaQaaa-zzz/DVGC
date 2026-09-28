"""Bounded Actor-only supervised warmup; fixed normalizer and explicit teacher."""
from pathlib import Path
import time
import numpy as np
from .protocol import atomic_json


def select_warmup_checkpoint(scores):
    """Predeclared development score, larger better, earliest on ties (including 0)."""
    if not scores or 0 not in scores or not all(np.isfinite(v) for v in scores.values()):
        raise ValueError('finite development scores including update zero required')
    return min(scores,key=lambda step:(-scores[step],step))


def warmup_actor(network, initializer, retention_reference, demo_manifest, retention,
                 output, *, seed=0, updates=2000, checkpoint_callback=None,
                 probe_callback=None, batch_size=256, metrics_callback=None):
    import jax
    import jax.numpy as jp
    import optax
    from .student import load_optional_demo
    from ..policy_retention import action_mse
    from ..handoff_bank import pytree_sha256
    if type(updates) is not int or not 0<=updates<=2000:raise ValueError('warmup budget is 0..2000')
    if type(batch_size) is not int or batch_size<=0:raise ValueError('positive batch required')
    if initializer is None or retention_reference is None:raise ValueError('explicit initializer and reference required')
    data=load_optional_demo(demo_manifest)
    normalizer,actor,critic=initializer
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    fixed_hash=pytree_sha256(normalizer); critic_hash=pytree_sha256(critic)
    report=dict(schema='jit_bridge_warmup_v1_2',requested_updates=updates,completed_updates=0,
        learning_rate=1e-5,optimizer='Adam',grad_clip_norm=1.,demo_coefficient=1.,keep_coefficient=1.,
        environment_interactions=0,checkpoint_semantics='inference Actor plus unchanged normalizer/critic; no optimizer resume',
        normalizer_sha256=fixed_hash,critic_sha256=critic_hash,checkpoints=[],status='running')
    def checkpoint(step):
        import pickle
        from .contracts import file_sha
        path=output/f'update_{step:04d}.pkl'
        with path.open('xb') as stream:pickle.dump(jax.device_get((normalizer,actor,critic)),stream)
        if checkpoint_callback is not None:checkpoint_callback(step,(normalizer,actor,critic))
        probe=probe_callback(step,(normalizer,actor,critic)) if probe_callback is not None else None
        report['checkpoints'].append(dict(update=step,actor_sha256=pytree_sha256(actor),probe=probe,path=str(path.resolve()),sha256=file_sha(path)))
        atomic_json(output/'warmup_status.json',report)
    checkpoint(0)
    if data is None:
        report['status']='skipped_empty_demo';atomic_json(output/'warmup_status.json',report)
        return initializer
    if retention is None:raise ValueError('warmup requires successful TRAIN anchors')
    keep_obs,keep_probs=map(np.asarray,retention)
    if keep_obs.ndim!=2 or keep_obs.shape[1]!=76 or not len(keep_obs) or not np.isfinite(keep_obs).all():
        raise ValueError('invalid retention observations')
    if keep_probs.shape!=(len(keep_obs),) or not np.isfinite(keep_probs).all() or np.any(keep_probs<0) or keep_probs.sum()<=0:
        raise ValueError('invalid retention weights')
    obs,targets,probs=map(jp.asarray,data);keep_obs=jp.asarray(keep_obs);keep_probs=jp.asarray(keep_probs/keep_probs.sum())
    reference_norm,reference_actor=jax.tree.map(jax.lax.stop_gradient,retention_reference)
    def prediction(norm,params,x):
        return network.parametric_action_distribution.mode(network.policy_network.apply(norm,params,{'state':x}))
    optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(1e-5));state=optimizer.init(actor)
    @jax.jit
    def step(params,state,key):
        dk,kk=jax.random.split(key)
        ids=jax.random.choice(dk,len(obs),(batch_size,),p=probs)
        kids=jax.random.choice(kk,len(keep_obs),(batch_size,),p=keep_probs)
        teacher=prediction(reference_norm,reference_actor,keep_obs[kids])
        def loss(p):
            demo=action_mse(prediction(normalizer,p,obs[ids]),targets[ids])
            keep=action_mse(prediction(normalizer,p,keep_obs[kids]),teacher)
            return demo+keep,(demo,keep)
        (value,parts),grad=jax.value_and_grad(loss,has_aux=True)(params)
        delta,new_state=optimizer.update(grad,state,params)
        candidate=optax.apply_updates(params,delta)
        finite=jp.all(jp.stack([jp.all(jp.isfinite(x)) for x in jax.tree.leaves((value,grad,candidate,new_state))]))
        return candidate,new_state,value,parts,optax.global_norm(grad),finite
    started=time.monotonic();key=jax.random.PRNGKey(seed)
    try:
        with (output/'metrics.jsonl').open('x') as stream:
            import json
            for index in range(1,updates+1):
                result=step(actor,state,jax.random.fold_in(key,index))
                candidate,new_state,value,parts,norm,finite=result
                if not bool(finite):raise FloatingPointError('nonfinite warmup update rejected')
                actor,state=candidate,new_state
                report['completed_updates']=index
                row=dict(update=index,loss=float(value),demo=float(parts[0]),keep=float(parts[1]),actor_grad_norm=float(norm))
                stream.write(json.dumps(row)+'\n');stream.flush()
                if metrics_callback is not None:metrics_callback(row)
                if index in (500,1000,2000) or index==updates:checkpoint(index)
        if pytree_sha256(normalizer)!=fixed_hash or pytree_sha256(critic)!=critic_hash:
            raise ValueError('warmup changed frozen normalizer or critic')
        report['status']='completed'
    except BaseException:
        report['status']='error';raise
    finally:
        report['wall_seconds']=time.monotonic()-started
        atomic_json(output/'warmup_status.json',report)
    return normalizer,actor,critic
