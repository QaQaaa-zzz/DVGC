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
                 probe_callback=None, batch_size=256, metrics_callback=None, full_learner=False, actual_gradient_audit=False,
                 demo_coefficient=1., keep_coefficient=1., update_guard=None, checkpoint_steps=None,
                 resume_checkpoint=None):
    if update_guard is not None or checkpoint_steps is not None or resume_checkpoint is not None:
        return _constrained_warmup(network,initializer,retention_reference,demo_manifest,retention,output,
            seed=seed,updates=updates,checkpoint_callback=checkpoint_callback,probe_callback=probe_callback,
            batch_size=batch_size,metrics_callback=metrics_callback,actual_gradient_audit=actual_gradient_audit,
            demo_coefficient=demo_coefficient,keep_coefficient=keep_coefficient,update_guard=update_guard,
            checkpoint_steps=checkpoint_steps,resume_checkpoint=resume_checkpoint)
    import jax
    import jax.numpy as jp
    import optax
    from .student import load_optional_demo
    from ..policy_retention import action_mse
    from ..handoff_bank import pytree_sha256
    if type(updates) is not int or not 0<=updates<=2000:raise ValueError('warmup budget is 0..2000')
    if type(batch_size) is not int or batch_size<=0:raise ValueError('positive batch required')
    if not all(np.isfinite(x) and x>0 for x in (demo_coefficient,keep_coefficient)):
        raise ValueError('positive finite supervised coefficients required')
    if initializer is None or retention_reference is None:raise ValueError('explicit initializer and reference required')
    data=load_optional_demo(demo_manifest)
    normalizer,actor,critic=initializer
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    fixed_hash=pytree_sha256(normalizer); critic_hash=pytree_sha256(critic)
    optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(1e-5));state=optimizer.init(actor)
    key=jax.random.PRNGKey(seed)
    report=dict(schema='jit_bridge_warmup_v1_2',requested_updates=updates,completed_updates=0,
        learning_rate=1e-5,optimizer='Adam',grad_clip_norm=1.,demo_coefficient=demo_coefficient,keep_coefficient=keep_coefficient,
        environment_interactions=0,checkpoint_semantics='full BC Actor/critic/normalizer/Adam/RNG/update clock' if full_learner else 'inference Actor plus unchanged normalizer/critic; no optimizer resume',
        normalizer_sha256=fixed_hash,critic_sha256=critic_hash,checkpoints=[],status='running')
    def checkpoint(step):
        import pickle
        from .contracts import file_sha
        path=output/f'update_{step:04d}.pkl'
        with path.open('xb') as stream:pickle.dump(jax.device_get((normalizer,actor,critic)),stream)
        if full_learner:
            full=output/f'learner_update_{step:04d}.pkl'
            with full.open('xb') as stream:pickle.dump(jax.device_get(dict(normalizer=normalizer,actor=actor,critic=critic,optimizer_state=state,rng=key,completed_supervised_updates=step,seed=seed,sampling='fold_in(rng,absolute_update)',optimizer='Adam1e-5/clip1',phase='BC',demo_coefficient=demo_coefficient,keep_coefficient=keep_coefficient)),stream)
            atomic_json(output/f'learner_update_{step:04d}.json',dict(path=str(full.resolve()),sha256=file_sha(full),completed_supervised_updates=step,actor_sha256=pytree_sha256(actor),optimizer_sha256=pytree_sha256(state),normalizer_sha256=fixed_hash,critic_sha256=critic_hash))
        stop_requested=bool(checkpoint_callback(step,(normalizer,actor,critic))) if checkpoint_callback is not None else False
        probe=probe_callback(step,(normalizer,actor,critic)) if probe_callback is not None else None
        report['checkpoints'].append(dict(update=step,actor_sha256=pytree_sha256(actor),probe=probe,path=str(path.resolve()),sha256=file_sha(path)))
        atomic_json(output/'warmup_status.json',report)
        return stop_requested
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
    @jax.jit
    def step(params,state,key):
        dk,kk=jax.random.split(key)
        ids=jax.random.choice(dk,len(obs),(batch_size,),p=probs)
        kids=jax.random.choice(kk,len(keep_obs),(batch_size,),p=keep_probs)
        teacher=prediction(reference_norm,reference_actor,keep_obs[kids])
        def loss(p):
            demo=action_mse(prediction(normalizer,p,obs[ids]),targets[ids])
            keep=action_mse(prediction(normalizer,p,keep_obs[kids]),teacher)
            return demo_coefficient*demo+keep_coefficient*keep,(demo,keep)
        (value,parts),grad=jax.value_and_grad(loss,has_aux=True)(params)
        delta,new_state=optimizer.update(grad,state,params)
        candidate=optax.apply_updates(params,delta)
        finite=jp.all(jp.stack([jp.all(jp.isfinite(x)) for x in jax.tree.leaves((value,grad,candidate,new_state))]))
        telemetry={}
        if actual_gradient_audit:
            from .actual_update import weighted_actor_metrics
            telemetry=weighted_actor_metrics(params,dict(demo=lambda p:demo_coefficient*loss(p)[1][0],keep=lambda p:keep_coefficient*loss(p)[1][1]))
            telemetry.update({'actual/actor_adam_delta_norm':optax.global_norm(delta),'actual/global_clip_scale':jp.minimum(1.,1./jp.maximum(optax.global_norm(grad),1e-20))})
        return candidate,new_state,value,parts,optax.global_norm(grad),finite,telemetry
    started=time.monotonic();key=jax.random.PRNGKey(seed)
    try:
        with (output/'metrics.jsonl').open('x') as stream:
            import json
            for index in range(1,updates+1):
                result=step(actor,state,jax.random.fold_in(key,index))
                candidate,new_state,value,parts,norm,finite,telemetry=result
                if not bool(finite):raise FloatingPointError('nonfinite warmup update rejected')
                actor,state=candidate,new_state
                report['completed_updates']=index
                row=dict(update=index,loss=float(value),demo=float(parts[0]),keep=float(parts[1]),actor_grad_norm=float(norm),**{k:float(v) for k,v in telemetry.items()})
                stream.write(json.dumps(row)+'\n');stream.flush()
                if metrics_callback is not None:metrics_callback(row)
                if index in ((100,500,1000,2000) if full_learner else (500,1000,2000)) or index==updates:
                    if checkpoint(index):
                        report['status']='stopped_retention_guard';break
        if pytree_sha256(normalizer)!=fixed_hash or pytree_sha256(critic)!=critic_hash:
            raise ValueError('warmup changed frozen normalizer or critic')
        if report['status']!='stopped_retention_guard':report['status']='completed'
    except BaseException:
        report['status']='error';raise
    finally:
        report['wall_seconds']=time.monotonic()-started
        atomic_json(output/'warmup_status.json',report)
    return normalizer,actor,critic


def _constrained_warmup(network, initializer, retention_reference, demo_manifest, retention, output,
                        *, seed, updates, checkpoint_callback, probe_callback, batch_size,
                        metrics_callback, actual_gradient_audit, demo_coefficient, keep_coefficient,
                        update_guard, checkpoint_steps, resume_checkpoint):
    """Attempt-clock protocol; defaults in warmup_actor preserve the legacy path."""
    import jax
    import jax.numpy as jp
    import optax
    import pickle
    import json
    from .student import load_optional_demo
    from ..policy_retention import action_mse
    from ..handoff_bank import pytree_sha256
    from .contracts import file_sha
    from .retention_guard import guarded_adam_step, guard_metrics, require_finite
    if jax.local_device_count()!=1:raise ValueError('constrained warmup supports one device only')
    if type(updates) is not int or not 0<=updates<=1000:raise ValueError('attempt budget is 0..1000')
    if type(batch_size) is not int or batch_size<=0:raise ValueError('positive batch required')
    if not all(np.isfinite(x) and x>0 for x in (demo_coefficient,keep_coefficient)):raise ValueError('positive finite supervised coefficients required')
    if initializer is None or retention_reference is None:raise ValueError('explicit initializer and reference required')
    normalizer,actor,critic=initializer
    require_finite((initializer,retention_reference),'initializer/reference')
    data=load_optional_demo(demo_manifest)
    if retention is None:raise ValueError('warmup requires successful TRAIN anchors')
    keep_obs,keep_probs=map(np.asarray,retention)
    if keep_obs.ndim!=2 or keep_obs.shape[1]!=76 or not len(keep_obs):raise ValueError('invalid retention observations')
    require_finite((keep_obs,keep_probs),'retention inputs')
    if keep_probs.shape!=(len(keep_obs),) or np.any(keep_probs<0) or keep_probs.sum()<=0:raise ValueError('invalid retention weights')
    fixed_hash=pytree_sha256(normalizer);critic_hash=pytree_sha256(critic)
    guard=update_guard or {};enabled=guard.get('enabled',True) if update_guard is not None else False
    bank,config=guard.get('bank'),guard.get('config')
    reference_norm,reference_actor=jax.tree.map(jax.lax.stop_gradient,retention_reference)
    @jax.jit
    def prediction(p,x):
        return network.parametric_action_distribution.mode(network.policy_network.apply(normalizer,p,{'state':jp.asarray(x)}))
    @jax.jit
    def parent_prediction(x):
        return network.parametric_action_distribution.mode(network.policy_network.apply(reference_norm,reference_actor,{'state':jp.asarray(x)}))
    if enabled:
        if not bank or not config or not guard.get('identity'):raise ValueError('bound guard identity/bank/config required')
        guard_metrics(prediction,actor,bank,config)
        expected=np.asarray(parent_prediction(bank['observations']))
        if not np.array_equal(expected,np.asarray(bank['reference_actions'])):raise ValueError('guard reference identity does not match frozen parent actions')
    rejection_limit=config.get('max_consecutive_rejections') if enabled else None
    if rejection_limit is not None and (type(rejection_limit) is not int or rejection_limit<=0):raise ValueError('positive consecutive rejection limit required')
    schedule=[0,100,250,500,1000] if checkpoint_steps is None else list(checkpoint_steps)
    if any(type(x) is not int or x<0 or x>1000 for x in schedule):raise ValueError('invalid attempt checkpoint schedule')
    identity=dict(schema='jit_retention_guard_warmup_v1',seed=seed,batch_size=batch_size,
        demo_coefficient=demo_coefficient,keep_coefficient=keep_coefficient,
        normalizer_sha256=fixed_hash,critic_sha256=critic_hash,parent_sha256=pytree_sha256(retention_reference),
        initializer_sha256=pytree_sha256(initializer),data_sha256=pytree_sha256(data),keep_sha256=pytree_sha256(retention),
        guard_sha256=pytree_sha256(bank) if bank is not None else None,guard_identity=guard.get('identity'),
        guard_config=config,guard_enabled=enabled,checkpoint_steps=schedule,optimizer='Adam1e-5/clip1',sampling='fold_in(base_rng,absolute_attempt)')
    # Canonical JSON removes tuple/numpy scalar distinctions from resume contracts.
    identity=json.loads(json.dumps(identity))
    optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(1e-5));state=optimizer.init(actor)
    key=jax.random.PRNGKey(seed);attempts=accepted=rejected=backtracking=consecutive=0
    if resume_checkpoint is not None:
        resume_checkpoint=Path(resume_checkpoint)
        receipt=json.loads(resume_checkpoint.with_suffix('.json').read_text())
        if file_sha(resume_checkpoint)!=receipt['sha256']:raise ValueError('resume checkpoint identity/hash changed')
        with resume_checkpoint.open('rb') as f:restored=pickle.load(f)
        if restored['identity']!=identity:raise ValueError('resume identity differs')
        normalizer,actor,critic=restored['normalizer'],restored['actor'],restored['critic'];state=restored['optimizer_state'];key=jp.asarray(restored['rng'])
        consecutive=restored['consecutive_rejections']
        attempts=restored['attempts'];accepted=restored['accepted_updates'];rejected=restored['rejected_updates'];backtracking=restored['backtracking_evaluations']
        require_finite((normalizer,actor,critic,state,key),'restored learner')
        if pytree_sha256(normalizer)!=fixed_hash or pytree_sha256(critic)!=critic_hash:raise ValueError('restored frozen identity differs from parent')
        if not np.array_equal(np.asarray(key),np.asarray(jax.random.PRNGKey(seed))):raise ValueError('restored base RNG identity differs')
        if attempts>updates or attempts!=accepted+rejected:raise ValueError('invalid resume attempt counters')
        for name,tree in [('actor',actor),('optimizer',state),('normalizer',normalizer),('critic',critic)]:
            if pytree_sha256(tree)!=receipt[name+'_sha256']:raise ValueError('resume learner identity mismatch')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    report=dict(identity=identity,requested_attempts=updates,attempts=attempts,accepted_updates=accepted,rejected_updates=rejected,
        completed_updates=accepted,backtracking_evaluations=backtracking,environment_interactions=0,status='running',
        policy_retention='deterministic policy retention',checkpoints=[],normalizer_sha256=fixed_hash,critic_sha256=critic_hash)
    def sync_counts():
        report.update(attempts=attempts,accepted_updates=accepted,rejected_updates=rejected,completed_updates=accepted,backtracking_evaluations=backtracking,consecutive_rejections=consecutive)
    def checkpoint(call_callback=True):
        sync_counts();path=output/f'learner_attempt_{attempts:04d}.pkl'
        if not path.exists():
            payload=dict(identity=identity,normalizer=normalizer,actor=actor,critic=critic,optimizer_state=state,rng=key,
                attempts=attempts,accepted_updates=accepted,rejected_updates=rejected,backtracking_evaluations=backtracking,consecutive_rejections=consecutive)
            with path.open('xb') as f:pickle.dump(jax.device_get(payload),f)
            receipt=dict(path=str(path.resolve()),sha256=file_sha(path),attempts=attempts,accepted_updates=accepted,
                actor_sha256=pytree_sha256(actor),optimizer_sha256=pytree_sha256(state),normalizer_sha256=pytree_sha256(normalizer),critic_sha256=pytree_sha256(critic))
            atomic_json(path.with_suffix('.json'),receipt)
            inference=output/f'attempt_{attempts:04d}.pkl'
            with inference.open('xb') as f:pickle.dump(jax.device_get((normalizer,actor,critic)),f)
            report['checkpoints'].append({**receipt,'inference_path':str(inference.resolve())})
            atomic_json(output/'last_completed.json',receipt)
        stop=bool(checkpoint_callback(attempts,(normalizer,actor,critic))) if call_callback and checkpoint_callback is not None else False
        if call_callback and probe_callback is not None:report['checkpoints'][-1]['probe']=probe_callback(attempts,(normalizer,actor,critic))
        atomic_json(output/'warmup_status.json',report)
        return stop
    started=time.monotonic()
    try:
        if checkpoint():report['status']='stopped_retention_guard';return normalizer,actor,critic
        if rejection_limit is not None and consecutive>=rejection_limit:
            report['status']='blocked_by_guard';return normalizer,actor,critic
        if data is None:report['status']='skipped_empty_demo';return normalizer,actor,critic
        obs,targets,probs=map(jp.asarray,data);keep_obs=jp.asarray(keep_obs);keep_probs=jp.asarray(keep_probs/keep_probs.sum())
        def parts(p,dx,dt,kx,kt):
            return action_mse(prediction(p,dx),dt),action_mse(prediction(p,kx),kt)
        def loss_aux(p,dx,dt,kx,kt):
            demo,keep=parts(p,dx,dt,kx,kt)
            return demo_coefficient*demo+keep_coefficient*keep,(demo,keep)
        @jax.jit
        def propose(p,s,dx,dt,kx,kt):
            (value,terms),grad=jax.value_and_grad(loss_aux,has_aux=True)(p,dx,dt,kx,kt)
            delta,new=optimizer.update(grad,s,p)
            return value,terms,grad,delta,new
        @jax.jit
        def telemetry(p,dx,dt,kx,kt):
            from .actual_update import weighted_actor_metrics
            return weighted_actor_metrics(p,dict(demo=lambda p:demo_coefficient*parts(p,dx,dt,kx,kt)[0],keep=lambda p:keep_coefficient*parts(p,dx,dt,kx,kt)[1]))
        with (output/'update_audit.jsonl').open('x') as stream:
            for index in range(attempts+1,updates+1):
                attempt_started=time.monotonic()
                dk,kk=jax.random.split(jax.random.fold_in(key,index))
                ids=jax.random.choice(dk,len(obs),(batch_size,),p=probs);kids=jax.random.choice(kk,len(keep_obs),(batch_size,),p=keep_probs)
                teacher=parent_prediction(keep_obs[kids]);dx,dt,kx=obs[ids],targets[ids],keep_obs[kids]
                require_finite((dx,dt,teacher,actor,state),'minibatch/input learner')
                sampled_at=time.monotonic()
                value,terms,grad,delta,proposed=propose(actor,state,dx,dt,kx,teacher)
                require_finite((value,terms,grad,delta,proposed),'loss/gradient/Adam proposal')
                proposed_at=time.monotonic()
                diagnostic=telemetry(actor,dx,dt,kx,teacher) if enabled or actual_gradient_audit else {}
                require_finite(diagnostic,'gradient telemetry')
                diagnosed_at=time.monotonic()
                if enabled:
                    candidate,new_state,row=guarded_adam_step(actor,state,None,prediction,bank,optimizer,config,proposal=(value,grad,delta,proposed))
                else:
                    candidate=optax.apply_updates(actor,delta);new_state=proposed
                    require_finite(candidate,'unguarded candidate')
                    row=dict(accepted=True,alpha=1.,loss=float(value),actor_grad_norm=float(optax.global_norm(grad)),original_delta_norm=float(optax.global_norm(delta)),accepted_delta_norm=float(optax.global_norm(delta)),backtracking_evaluations=0)
                checked_at=time.monotonic()
                row.update(sampling_seconds=sampled_at-attempt_started,
                    proposal_seconds=proposed_at-sampled_at,
                    diagnostic_seconds=diagnosed_at-proposed_at,
                    guard_seconds=checked_at-diagnosed_at if enabled else 0.)
                row.update({k:float(v) for k,v in diagnostic.items()})
                require_finite((row['actor_grad_norm'],row['original_delta_norm']),'update telemetry')
                actor,state=candidate,new_state;attempts=index;accepted+=int(row['accepted']);rejected+=int(not row['accepted']);backtracking+=row['backtracking_evaluations']
                consecutive=0 if row['accepted'] else consecutive+1
                from .actual_update import adam_update_count
                row.update(attempt=index,update=index,accepted_updates=accepted,rejected_updates=rejected,demo=float(terms[0]),keep=float(terms[1]),
                    demo_indices=np.asarray(ids).tolist(),keep_indices=np.asarray(kids).tolist(),data_sha256=identity['data_sha256'],parent_sha256=identity['parent_sha256'],optimizer_sha256=pytree_sha256(state),adam_count=int(adam_update_count(state)),wall_seconds=time.monotonic()-attempt_started)
                stream.write(json.dumps(row)+'\n');stream.flush();sync_counts()
                if metrics_callback is not None:metrics_callback(row)
                if rejection_limit is not None and consecutive>=rejection_limit:
                    report['status']='blocked_by_guard';break
                if index in schedule or index==updates:
                    if checkpoint():report['status']='stopped_retention_guard';break
            else:report['status']='completed'
        if pytree_sha256(normalizer)!=fixed_hash or pytree_sha256(critic)!=critic_hash:raise ValueError('warmup changed frozen normalizer or critic')
        return normalizer,actor,critic
    except BaseException as error:
        report.update(status='error',error_type=type(error).__name__,error=str(error),failed_attempt=attempts+1)
        checkpoint(call_callback=False)
        atomic_json(output/'failure_context.json',dict(attempt=attempts+1,identity=identity,error_type=type(error).__name__,error=str(error),last_consistent_attempt=attempts))
        raise
    finally:
        report['wall_seconds']=time.monotonic()-started
        checkpoint(call_callback=False)
        atomic_json(output/'warmup_status.json',report)
