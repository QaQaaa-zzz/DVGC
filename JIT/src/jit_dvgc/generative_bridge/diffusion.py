"""Explicit DDPM mean epsilon loss and trailing DDIM20; no physics execution."""
import numpy as np
from .performance import measure

TIMESTEPS=tuple(range(99,-1,-5))


def cosine_schedule():
    t=np.arange(101,dtype=np.float64)/100
    f=np.cos((t+.008)/1.008*np.pi/2)**2
    beta=np.minimum(1-f[1:]/f[:-1],.999)
    return beta,np.cumprod(1-beta)


def noise_mse(prediction,target):
    import jax.numpy as jp
    return jp.mean(jp.square(prediction-target))


def ddim_sample(predict,observations,initial_noise):
    import jax.numpy as jp
    _,schedule=cosine_schedule();ab=jp.asarray(schedule)
    x=initial_noise
    for k in TIMESTEPS:
        eps=predict(x,observations,jp.full((len(x),),k,dtype=jp.int32))
        clean=jp.clip((x-jp.sqrt(1-ab[k])*eps)/jp.sqrt(ab[k]),-1,1)
        corrected=(x-jp.sqrt(ab[k])*clean)/jp.sqrt(1-ab[k])
        previous=ab[k-5] if k>=5 else jp.asarray(1.)
        x=jp.sqrt(previous)*clean+jp.sqrt(1-previous)*corrected
    return x


def update_if_new_data(incumbent,corpus,*,updates,train):
    if not corpus['new_data']:
        return incumbent,{'status':'skipped_no_new_data','updates':0,'rng_advanced':False}
    if type(updates) is not int or not 0<updates<=2000:
        raise ValueError('new data requires declared incremental budget 1..2000')
    return train(incumbent,corpus,updates)


def _optimizer():
    import optax
    # LR applied after this stable state structure: new stage LR does not reset Adam.
    return optax.chain(optax.clip_by_global_norm(1.),optax.scale_by_adam(),
                       optax.add_decayed_weights(1e-4),optax.scale(-1.))


def pretrain_learning_rate(*,max_updates=20000,warmup_updates=1000,
                           learning_rate=1e-4,min_learning_rate=1e-5):
    """LR for the zero-based update about to run; final update reaches the floor."""
    import jax.numpy as jp
    if (type(max_updates) is not int or not 1<max_updates<=20000
        or type(warmup_updates) is not int or not 0<=warmup_updates<max_updates
        or not np.isfinite(learning_rate) or not np.isfinite(min_learning_rate)
        or not 0<min_learning_rate<=learning_rate):
        raise ValueError('invalid bounded pretraining LR schedule')
    def schedule(update):
        index=jp.asarray(update)
        start=jp.asarray(learning_rate)
        floor=jp.asarray(min_learning_rate)
        progress=jp.clip((index-warmup_updates)/max(1,max_updates-warmup_updates-1),0,1)
        cosine=floor+(start-floor)*(.5+.5*jp.cos(jp.pi*progress))
        if warmup_updates:
            return jp.where(index<warmup_updates,start*(index+1)/warmup_updates,cosine)
        return cosine
    return schedule


def _finite(tree):
    import jax
    return all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(tree))


def create_train_state(params,rng,normalizer):
    import jax.numpy as jp
    if (np.shape(normalizer['mean'])!=(76,) or np.shape(normalizer['std'])!=(76,)
        or np.any(np.asarray(normalizer['std'])<=0) or not _finite((params,normalizer))):
        raise ValueError('finite fixed source normalizer and params required')
    return {'params':params,'ema':params,'optimizer':_optimizer().init(params),
            'rng':rng,'updates':jp.asarray(0),'normalizer':normalizer}


def make_train_step(predict,*,learning_rate):
    import jax
    import jax.numpy as jp
    import optax
    if not callable(learning_rate) and (not np.isfinite(learning_rate) or learning_rate<=0):
        raise ValueError('finite positive LR required')
    optimizer=_optimizer();ab=jp.asarray(cosine_schedule()[1])
    def finite_device(tree):
        return jp.all(jp.stack([jp.all(jp.isfinite(v)) for v in jax.tree.leaves(tree)]))
    @jax.jit
    def input_guard(state,observations,actions):
        return jp.stack((finite_device((state,observations,actions)),
                         jp.all(jp.abs(actions)<=1)))
    @jax.jit
    def propose(state,observations,actions):
        rng,key_k,key_noise=jax.random.split(state['rng'],3)
        k=jax.random.randint(key_k,(len(actions),),0,100)
        eps=jax.random.normal(key_noise,actions.shape)
        alpha=ab[k,None,None]
        noisy=jp.sqrt(alpha)*actions+jp.sqrt(1-alpha)*eps
        obs=(observations-state['normalizer']['mean'])/state['normalizer']['std']
        value,grad=jax.value_and_grad(lambda p:noise_mse(predict(p,noisy,obs,k),eps))(state['params'])
        delta,optstate=optimizer.update(grad,state['optimizer'],state['params'])
        lr=learning_rate(state['updates']) if callable(learning_rate) else learning_rate
        params=optax.apply_updates(state['params'],jax.tree.map(lambda x:lr*x,delta))
        ema=jax.tree.map(lambda a,b:.999*a+.001*b,state['ema'],params)
        proposed={**state,'params':params,'ema':ema,'optimizer':optstate,'rng':rng,
                  'updates':state['updates']+1}
        # Gradients and complete state stay on device; no buffer donation on rejection.
        return proposed,(finite_device((proposed,value,grad)),value)
    def step(state,observations,actions):
        if np.shape(observations)!=(len(actions),76) or np.shape(actions)[1:]!=(16,4):
            raise ValueError('generator batch shape mismatch')
        with measure('numerical_guard',backend=jax.default_backend(),batch_size=len(actions),
                     phase='inputs',device_to_host_bytes=2):
            finite,bounded=jax.device_get(input_guard(state,observations,actions))
        if not finite:raise FloatingPointError('nonfinite generator inputs')
        if not bounded:raise ValueError('action permission drift')
        # The existing post-proposal safety transfer is the completion boundary.
        # This event includes the fused proposal-finiteness reduction.
        with measure('forward_backward_optimizer',backend=jax.default_backend(),batch_size=len(actions),
                     includes_proposal_guard=True,device_to_host_bytes=5):
            proposed,checks=propose(state,observations,actions)
            finite,value=jax.device_get(checks)
        if not finite:raise FloatingPointError('nonfinite generator proposal rejected')
        return proposed,float(value)
    return step


def save_state(path,state,identity):
    with measure('checkpoint_write',checkpoint=str(path)):
        return _save_state(path,state,identity)


def _save_state(path,state,identity):
    from pathlib import Path
    from flax import serialization
    import jax
    from .contracts import file_sha
    from .protocol import atomic_json
    if not _finite(state):raise FloatingPointError('cannot checkpoint nonfinite G state')
    path=Path(path);path.mkdir(parents=True,exist_ok=False)
    import os
    from .provenance import _sync_directory
    payload=path/'state.msgpack'
    with payload.open('xb') as stream:
        stream.write(serialization.to_bytes(jax.device_get(state)))
        stream.flush();os.fsync(stream.fileno())
    atomic_json(path/'manifest.json',{'schema':'jit_generator_training_state_v1_1',
        'identity':identity,'state_sha256':file_sha(payload),'updates':int(state['updates']),
        'inference_parameters':'ema','includes':['params','ema','optimizer','rng','normalizer','updates']})
    _sync_directory(path);_sync_directory(path.parent)


def restore_state(path,template,identity):
    from pathlib import Path
    import json
    from flax import serialization
    from .contracts import file_sha
    path=Path(path);manifest=json.loads((path/'manifest.json').read_text())
    if manifest['identity']!=identity or manifest['state_sha256']!=file_sha(path/'state.msgpack'):
        raise ValueError('generator checkpoint identity/hash mismatch')
    state=serialization.from_bytes(template,(path/'state.msgpack').read_bytes())
    if not _finite(state) or int(state['updates'])!=manifest['updates']:
        raise ValueError('invalid restored generator state')
    return state


def select_checkpoint(scored):
    if not scored or any(not np.isfinite(loss) for loss,_ in scored):
        raise ValueError('finite fixed-dev scores including incumbent required')
    # Python min is stable: list incumbent first to keep it on exact ties.
    return min(scored,key=lambda item:item[0])[1]


def train_pretrain(initial,corpus,*,predict,dev_fixture,output,identity,updates,
                   charged_updates_before,
                   batch_size=256,max_updates=20000,warmup_updates=1000,
                   learning_rate=1e-4,min_learning_rate=1e-5,
                   validation_every=1000,max_wall_seconds):
    """Train on admitted real H16 windows; return the selected complete state.

    ``updates`` is this invocation's charged update cap. The caller must pass
    cumulative prior charged optimizer attempts, including discarded work;
    checkpoint age alone cannot establish remaining budget. ``max_updates`` is
    both the total charged cap and the learning-rate schedule's horizon.
    An existing output is never replayed automatically: reconcile its receipts
    and start a new attempt from a selected full-state checkpoint explicitly.
    """
    import time
    import jax
    import jax.numpy as jp
    from pathlib import Path
    from .feedback_data import sample_corpus
    from .corpus_index import compile_corpus_index
    from .protocol import atomic_json
    start_update=int(initial['updates'])
    schedule=pretrain_learning_rate(max_updates=max_updates,warmup_updates=warmup_updates,
                                    learning_rate=learning_rate,min_learning_rate=min_learning_rate)
    if (type(updates) is not int or updates<=0 or start_update<0
        or type(charged_updates_before) is not int
        or charged_updates_before<start_update
        or charged_updates_before+updates>max_updates or start_update+updates>max_updates
        or type(batch_size) is not int or batch_size<=0
        or type(validation_every) is not int or validation_every<=0
        or not np.isfinite(max_wall_seconds) or max_wall_seconds<=0):
        raise ValueError('declared bounded pretraining update, batch and wall budget required')
    if not _finite(initial):raise FloatingPointError('nonfinite initial generator state')
    if (not corpus.get('groups') or not any(corpus['groups'].values())
        or any(len(trace['arrays']['normalized_action_executed'])<16
               for traces in corpus['groups'].values() for trace in traces)):
        raise ValueError('admitted real generator corpus required')
    observations,actions,k,eps=map(jp.asarray,dev_fixture)
    if (len(actions)==0 or np.shape(observations)!=(len(actions),76)
        or np.shape(actions)[1:]!=(16,4)
        or np.shape(k)!=(len(actions),) or np.shape(eps)!=np.shape(actions)
        or np.asarray(dev_fixture[2]).dtype.kind not in 'iu'
        or not _finite(dev_fixture) or np.any(np.asarray(k)<0) or np.any(np.asarray(k)>=100)
        or np.any(np.abs(np.asarray(actions))>1)):
        raise ValueError('invalid fixed generator dev fixture')
    ab=jp.asarray(cosine_schedule()[1])[k,None,None]
    def score(state):
        obs=(observations-state['normalizer']['mean'])/state['normalizer']['std']
        noisy=jp.sqrt(ab)*actions+jp.sqrt(1-ab)*eps
        value=float(noise_mse(predict(state['ema'],noisy,obs,k),eps))
        if not np.isfinite(value):raise FloatingPointError('nonfinite generator dev score')
        return value
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();state=initial;logs=[];scores=[]
    try:
        save_state(root/'initial',state,identity)
        scores.append((score(state),'initial'))
        with measure('corpus_index_build',backend='cpu',batch_size=batch_size):
            compiled_corpus=compile_corpus_index(corpus)
        step=make_train_step(predict,learning_rate=schedule)
        for i in range(updates):
            if time.monotonic()-started>=max_wall_seconds:
                raise TimeoutError('generator pretraining wall budget exhausted')
            next_rng,data_key=jax.random.split(state['rng'])
            seed=int(jax.random.bits(data_key,(),dtype=jp.uint32))
            with measure('corpus_sampling',backend='cpu',batch_size=batch_size,update=i+1):
                obs,act,sources=sample_corpus(compiled_corpus,np.random.default_rng(seed),batch_size)
            # Charge before proposing an update: a failed/nonfinite attempt is not free.
            atomic_json(root/'cost_progress.json',{'charged_updates':i+1,
                        'charged_updates_before':charged_updates_before,
                        'total_charged_updates':charged_updates_before+i+1,
                        'initial_updates':start_update})
            state,value=step({**state,'rng':next_rng},jp.asarray(obs),jp.asarray(act))
            count=start_update+i+1
            logs.append({'update':count,'noise_mse':value,
                         'learning_rate':float(schedule(count-1)),
                         'source_counts':{g:sources.count(g) for g in corpus['groups']}})
            import json
            with (root/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps({'update':count,'metrics':{'generator/noise_mse':float(value),
                    'generator/learning_rate':float(schedule(count-1))}})+'\n')
            if count%validation_every==0 or i+1==updates:
                name=f'update_{count:05d}' if max_updates>9999 else f'update_{count:04d}'
                save_state(root/name,state,identity)
                scores.append((score(state),name))
        selected=select_checkpoint(scores)
        restored=restore_state(root/selected,initial,identity)
        report={'status':'completed','updates':updates,'initial_updates':start_update,
                'final_updates':start_update+updates,'selected':selected,'scores':scores,
                'requested_mix':corpus.get('requested_mix'),
                'realized_mix':corpus.get('realized_mix'),
                'wall_seconds':time.monotonic()-started,'environment_interactions':0,
                'charged_updates':updates,'charged_updates_before':charged_updates_before,
                'total_charged_updates':charged_updates_before+updates,'metrics':logs}
        with measure('logs_and_reports',backend='cpu',batch_size=batch_size):
            atomic_json(root/'generator_selection.json',report)
        return restored,report
    except BaseException as error:
        atomic_json(root/'failure.json',{'status':'failed','error':repr(error),
            'charged_updates':json_cost(root),'charged_updates_before':charged_updates_before,
            'total_charged_updates':charged_updates_before+json_cost(root),
            'completed_updates':len(logs),
            'initial_updates':start_update,'wall_seconds':time.monotonic()-started,
            'environment_interactions':0,'metrics':logs})
        raise


def json_cost(root):
    """Read the durable charged count after a failed attempt."""
    import json
    path=root/'cost_progress.json'
    return json.loads(path.read_text())['charged_updates'] if path.exists() else 0


def train_incremental(incumbent,corpus,*,predict,dev_fixture,output,identity,
                      updates,batch_size=256,max_wall_seconds,
                      generator_update_policy='fixed_dev_best',charged_updates_before=None):
    """Bounded update with explicit continuation and separate old-dev monitoring.

    last_valid returns the final durable full state, including its inference EMA.
    Legacy fixed_dev_best retains the historical dev selection rule. Neither mode
    recovers silently after a failed proposal or checkpoint write.
    """
    import time
    import jax
    import jax.numpy as jp
    from pathlib import Path
    from .feedback_data import sample_corpus
    from .corpus_index import compile_corpus_index
    from .protocol import atomic_json
    if generator_update_policy not in ('fixed_dev_best','last_valid'):
        raise ValueError('unknown generator update policy')
    initial_updates=int(incumbent['updates'])
    if charged_updates_before is None:charged_updates_before=initial_updates
    if type(charged_updates_before) is not int or charged_updates_before<initial_updates:
        raise ValueError('cumulative charged updates must cover incumbent state updates')
    common={'generator_update_policy':generator_update_policy,'inference_parameters':'ema',
            'initial_state_updates':initial_updates,'charged_updates_before':charged_updates_before}
    if not _finite(incumbent):raise FloatingPointError('nonfinite incumbent generator state')
    if not corpus['new_data']:
        return incumbent,{**common,'status':'skipped_no_new_data','updates':0,'charged_updates':0,
            'rng_advanced':False,'state_updates':initial_updates,
            'total_charged_updates':charged_updates_before}
    if (type(updates) is not int or not 0<updates<=2000
        or type(batch_size) is not int or batch_size<=0
        or not 0<max_wall_seconds<float('inf')):
        raise ValueError('declared bounded incremental compute budget required')
    root=Path(output);root.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    from .provenance import CompactProvenance
    provenance_log=CompactProvenance(updates=updates,batch_size=batch_size)
    provenance_pointer=None
    logs=[];state=incumbent;scores=[];state_counts={'incumbent':initial_updates}
    try:
        with measure('corpus_index_build',backend='cpu',batch_size=batch_size):
            compiled_corpus=compile_corpus_index(corpus,include_trajectory_identity=True)
        step=make_train_step(predict,learning_rate=1e-5)
        observations,actions,k,eps=map(jp.asarray,dev_fixture)
        if not _finite(dev_fixture):raise FloatingPointError('invalid fixed dev noise fixture')
        ab=jp.asarray(cosine_schedule()[1])[k,None,None]
        def score(state):
            obs=(observations-state['normalizer']['mean'])/state['normalizer']['std']
            loss=float(noise_mse(predict(state['ema'],jp.sqrt(ab)*actions+jp.sqrt(1-ab)*eps,obs,k),eps))
            if not np.isfinite(loss):raise FloatingPointError('nonfinite generator dev score')
            return loss
        save_state(root/'incumbent',incumbent,identity)
        scores.append((score(incumbent),'incumbent'))
        for i in range(updates):
            if time.monotonic()-start>max_wall_seconds:raise TimeoutError('generator compute budget exhausted')
            next_rng,data_key=jax.random.split(state['rng'])
            seed=int(jax.random.bits(data_key,(),dtype=jp.uint32))
            with measure('corpus_sampling',backend='cpu',batch_size=batch_size,update=i+1):
                obs,act,sources,provenance=sample_corpus(compiled_corpus,np.random.default_rng(seed),batch_size,
                    return_metadata=True,include_trajectory_identity=True)
            provenance_log.append(provenance,update=i+1,state_updates=initial_updates+i+1,sampling_seed=seed)
            atomic_json(root/'cost_progress.json',{**common,'charged_updates':i+1,
                'total_charged_updates':charged_updates_before+i+1})
            state,value=step({**state,'rng':next_rng},jp.asarray(obs),jp.asarray(act))
            logs.append({'update':i+1,'state_updates':int(state['updates']),'noise_mse':value,
                'source_counts':{g:sources.count(g) for g in corpus['groups']},
                'history_fraction':sources.count('history')/batch_size,
                'new_fraction':1-sources.count('history')/batch_size,'sample_provenance_index':i})
            if (i+1)%500==0 or i+1==updates:
                name=f'update_{i+1:04d}'
                save_state(root/name,state,identity)
                state_counts[name]=int(state['updates']);scores.append((score(state),name))
        monitoring_best=select_checkpoint(scores)
        selected=name if generator_update_policy=='last_valid' else monitoring_best
        restored=restore_state(root/selected,incumbent,identity)
        best={'checkpoint':monitoring_best,'noise_mse':dict((n,v) for v,n in scores)[monitoring_best],
            'state_updates':state_counts[monitoring_best],
            'age_updates':int(state['updates'])-state_counts[monitoring_best]}
        with measure('logs_and_reports',backend='cpu',batch_size=batch_size):
            provenance_pointer=provenance_log.save(root)
        report={**common,'sample_provenance':provenance_pointer,'status':'completed','updates':updates,'charged_updates':updates,
            'total_charged_updates':charged_updates_before+updates,'state_updates':int(restored['updates']),
            'last_state_updates':int(state['updates']),'selected':selected,'scores':scores,
            'monitoring_best':best,'old_dev_metric':'monitor_only' if generator_update_policy=='last_valid' else 'selection',
            'requested_mix':corpus['requested_mix'],'realized_mix':corpus['realized_mix'],
            'wall_seconds':time.monotonic()-start,'environment_interactions':0,'metrics':logs}
        with measure('logs_and_reports',backend='cpu',batch_size=batch_size):
            atomic_json(root/'generator_selection.json',report)
        return restored,report
    except BaseException as e:
        provenance_error=None
        if provenance_pointer is None:
            try:provenance_pointer=provenance_log.save(root)
            except BaseException as error:provenance_error=repr(error)
        atomic_json(root/'failure.json',{'sample_provenance':provenance_pointer,
            'provenance_write_error':provenance_error,**common,'status':'failed','error':repr(e),
            'completed_updates':len(logs),'charged_updates':json_cost(root),
            'total_charged_updates':charged_updates_before+json_cost(root),
            'wall_seconds':time.monotonic()-start,'metrics':logs,'environment_interactions':0})
        raise
