"""Explicit DDPM mean epsilon loss and trailing DDIM20; no physics execution."""
import numpy as np

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
    if not np.isfinite(learning_rate) or learning_rate<=0:raise ValueError('finite positive LR required')
    optimizer=_optimizer();ab=jp.asarray(cosine_schedule()[1])
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
        params=optax.apply_updates(state['params'],jax.tree.map(lambda x:learning_rate*x,delta))
        ema=jax.tree.map(lambda a,b:.999*a+.001*b,state['ema'],params)
        return {**state,'params':params,'ema':ema,'optimizer':optstate,'rng':rng,
                'updates':state['updates']+1},value,grad
    def step(state,observations,actions):
        if np.shape(observations)!=(len(actions),76) or np.shape(actions)[1:]!=(16,4):
            raise ValueError('generator batch shape mismatch')
        if not _finite((state,observations,actions)):raise FloatingPointError('nonfinite generator inputs')
        if np.any(np.abs(np.asarray(actions))>1):raise ValueError('action permission drift')
        proposed,value,grad=propose(state,observations,actions)
        if not _finite((proposed,value,grad)):raise FloatingPointError('nonfinite generator proposal rejected')
        return proposed,float(value)
    return step


def save_state(path,state,identity):
    from pathlib import Path
    from flax import serialization
    import jax
    from .contracts import file_sha
    from .protocol import atomic_json
    if not _finite(state):raise FloatingPointError('cannot checkpoint nonfinite G state')
    path=Path(path);path.mkdir(parents=True,exist_ok=False)
    payload=path/'state.msgpack';payload.write_bytes(serialization.to_bytes(jax.device_get(state)))
    atomic_json(path/'manifest.json',{'schema':'jit_generator_training_state_v1_1',
        'identity':identity,'state_sha256':file_sha(payload),'updates':int(state['updates']),
        'inference_parameters':'ema','includes':['params','ema','optimizer','rng','normalizer','updates']})


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


def train_incremental(incumbent,corpus,*,predict,dev_fixture,output,identity,
                      updates,batch_size=256,max_wall_seconds):
    """Bounded supervised phase; callers must reserve global compute budget first.

    All checkpoint candidates contain corresponding optimizer/RNG/EMA states.
    No new data returns the very same incumbent without touching any RNG.
    """
    if not corpus['new_data']:
        return incumbent,{'status':'skipped_no_new_data','updates':0,'rng_advanced':False}
    import time
    import jax
    import jax.numpy as jp
    from pathlib import Path
    from .feedback_data import sample_corpus
    from .protocol import atomic_json
    if type(updates) is not int or not 0<updates<=2000 or not 0<max_wall_seconds<float('inf'):
        raise ValueError('declared bounded incremental compute budget required')
    root=Path(output);root.mkdir(parents=True,exist_ok=False);start=time.monotonic()
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
    scores=[(score(incumbent),'incumbent')];state=incumbent;logs=[]
    try:
        for i in range(updates):
            if time.monotonic()-start>max_wall_seconds:raise TimeoutError('generator compute budget exhausted')
            next_rng,data_key=jax.random.split(state['rng'])
            seed=int(jax.random.bits(data_key,(),dtype=jp.uint32))
            obs,act,sources=sample_corpus(corpus,np.random.default_rng(seed),batch_size)
            atomic_json(root/'cost_progress.json',{'charged_updates':i+1})
            state,value=step({**state,'rng':next_rng},jp.asarray(obs),jp.asarray(act))
            logs.append({'update':i+1,'noise_mse':value,'source_counts':{g:sources.count(g) for g in corpus['groups']}})
            if (i+1)%500==0 or i+1==updates:
                name=f'update_{i+1:04d}';save_state(root/name,state,identity);scores.append((score(state),name))
        selected=select_checkpoint(scores)
        restored=restore_state(root/selected,incumbent,identity)
        report={'status':'completed','updates':updates,'selected':selected,'scores':scores,
            'requested_mix':corpus['requested_mix'],'realized_mix':corpus['realized_mix'],
            'wall_seconds':time.monotonic()-start,'environment_interactions':0,'metrics':logs}
        atomic_json(root/'generator_selection.json',report)
        return restored,report
    except BaseException as e:
        atomic_json(root/'failure.json',{'error':repr(e),'completed_updates':len(logs),
            'wall_seconds':time.monotonic()-start,'metrics':logs,'environment_interactions':0})
        raise
