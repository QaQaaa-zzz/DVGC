"""Official RSL PPO updates with an equivalent JAX sampler for MJX pulses.

The policy acts in latent Gaussian coordinates; tanh is an environment action
transform. Entropy is latent Gaussian entropy (not bounded-action entropy).
With gamma=lambda=1 and only a delayed terminal reward, every valid pulse action
has the same Monte Carlo return. Suffix/padding/unknown rows never enter storage.
"""
from pathlib import Path
import io
import math
import numpy as np


def torch_policy(spec, state=None):
    import torch
    from importlib.metadata import version
    if version('rsl-rl-lib')!='3.2.0':raise ValueError('Pinned RSL-RL 3.2.0 required')
    from tensordict import TensorDict
    from rsl_rl.modules import ActorCritic
    torch.set_num_threads(4)
    torch.manual_seed(spec['seed'])
    p=ActorCritic(TensorDict({'obs':torch.zeros(1,106)},[1]),
        {'policy':['obs'],'critic':['obs']},4,
        actor_hidden_dims=[128]*3,critic_hidden_dims=[128]*3,activation='elu',
        state_dependent_std=True,noise_std_type='log',init_noise_std=.6)
    if state is not None:
        saved=torch.load(io.BytesIO(state['torch_state']),map_location='cpu',weights_only=False)
        p.load_state_dict(saved['policy'])
    else:
        head=[m for m in p.actor.modules() if isinstance(m,torch.nn.Linear)][-1]
        with torch.no_grad():head.weight[:4].zero_();head.bias[:4].zero_()
    return p


def export(p):
    import torch
    return {name:{str(i):{'kernel':m.weight.detach().numpy().T.copy(),
                          'bias':m.bias.detach().numpy().copy()}
                  for i,m in enumerate(x for x in getattr(p,name).modules() if isinstance(x,torch.nn.Linear))}
            for name in ['actor','critic']}


def serialize_torch(p, optimizer=None, lr=.01):
    import torch
    b=io.BytesIO();torch.save(dict(policy=p.state_dict(),optimizer=optimizer,
                                 learning_rate=lr,rng=torch.get_rng_state()),b)
    return b.getvalue()


def initialize(spec,mean,std):
    import jax
    p=torch_policy(spec)
    return dict(params=export(p),normalizer_mean=np.asarray(mean,dtype=np.float32),
        normalizer_std=np.asarray(std,dtype=np.float32),rng=np.asarray(jax.random.PRNGKey(spec['seed'])),
        torch_state=serialize_torch(p,lr=spec['learning_rate']),total_updates=0,
        backend='rsl_rl_3.2.0',architecture='106_128_128_128_elu_state_dependent_log_std')


def normalized(state,obs):
    return (obs-state['normalizer_mean'])/state['normalizer_std']


def infer(state,obs):
    import jax
    import jax.numpy as jnp
    x=normalized(state,obs)
    def forward(layers):
        y=x
        for i in range(len(layers)):
            layer=layers[str(i)];y=y@jnp.asarray(layer['kernel'])+jnp.asarray(layer['bias'])
            if i<len(layers)-1:y=jax.nn.elu(y)
        return y
    out=forward(state['params']['actor'])
    return out[...,:4],jnp.exp(out[...,4:]),forward(state['params']['critic'])[...,0]


def sample(state,obs,key):
    import jax
    import jax.numpy as jnp
    mu,sd,v=infer(state,obs)
    raw=mu+sd*jax.random.normal(key,mu.shape)
    lp=(-.5*((raw-mu)/sd)**2-jnp.log(sd)-.5*math.log(2*math.pi)).sum(-1)
    return raw,jnp.tanh(raw),lp,v


def restore(path):
    from flax.serialization import msgpack_restore
    state=msgpack_restore(Path(path).read_bytes())
    if state.get('backend')!='rsl_rl_3.2.0':raise ValueError('RSL checkpoint backend mismatch')
    return state


def update_batch(spec,state,tape,feedback):
    import torch
    from tensordict import TensorDict
    from rsl_rl.algorithms import PPO
    from rsl_rl.storage import RolloutStorage
    mask=np.asarray(tape['mask'],bool)&np.asarray(feedback['eligible'],bool)[None,:]
    n=int(mask.sum());pulse_reward=np.zeros(mask.shape,np.float32)
    for e in range(mask.shape[1]):
        ticks=np.flatnonzero(mask[:,e])
        if len(ticks):pulse_reward[ticks[-1],e]=feedback['rewards'][e]
    returns=np.broadcast_to(np.asarray(feedback['rewards'],np.float32),mask.shape).copy()*mask
    learning=dict(mask=mask,reward=pulse_reward,returns=returns,
                  advantages=(returns-np.asarray(tape['value']))*mask)
    metrics=dict(backend='rsl_rl_3.2.0',effective_training_samples=n,
        eligible_episodes=int(np.asarray(feedback['eligible']).sum()),reward=float(np.sum(feedback['rewards'])),
        reward_components=feedback['component_sums'],optimizer_updates=0,post_update_kl=0.,
        post_update_clip_fraction=0.,value_explained_variance=None,total_loss=0.,actor_loss=0.,
        critic_loss=0.,entropy=0.,gradient_norm=0.,entropy_space='latent_gaussian')
    if not n:return state,dict(metrics,update_skipped=True),learning,[]
    p=torch_policy(spec,state)
    saved=torch.load(io.BytesIO(state['torch_state']),map_location='cpu',weights_only=False)
    torch.set_rng_state(saved['rng'])
    obs=normalized(state,np.asarray(tape['observation'])[mask]).astype(np.float32)
    td=TensorDict({'obs':torch.from_numpy(obs)},[n])
    raw=torch.tensor(np.asarray(tape['raw_action'])[mask],dtype=torch.float32)
    old_lp=torch.tensor(np.asarray(tape['log_prob'])[mask],dtype=torch.float32)
    old_v=torch.tensor(np.asarray(tape['value'])[mask],dtype=torch.float32)
    target=torch.tensor(returns[mask],dtype=torch.float32)
    with torch.no_grad():
        p.act(td);old_mu=p.action_mean.clone();old_sd=p.action_std.clone()
        error=float((p.get_actions_log_prob(raw)-old_lp).abs().max())
        value_error=float((p.evaluate(td).flatten()-old_v).abs().max())
    if error>2e-3 or value_error>2e-3:
        raise ValueError(f'JAX/RSL behavior mismatch: logprob={error}, value={value_error}')
    class ValidStorage(RolloutStorage):
        def mini_batch_generator(self,num_mini_batches,num_epochs=1):
            # All valid samples, including a smaller final minibatch, once/epoch.
            adv=target-old_v;adv=(adv-adv.mean())/(adv.std()+1e-8) if n>1 else adv*0
            for _ in range(num_epochs):
                for ix in torch.randperm(n).split(spec['minibatch_size']):
                    yield (td[ix],raw[ix],old_v[ix,None],adv[ix,None],target[ix,None],
                           old_lp[ix,None],old_mu[ix],old_sd[ix],(None,None),None)
    storage=ValidStorage('rl',n,1,td,[4])
    batches=math.ceil(n/spec['minibatch_size'])
    alg=PPO(p,storage,num_learning_epochs=1,num_mini_batches=batches,
        learning_rate=saved['learning_rate'],gamma=1.,lam=1.,clip_param=spec['clip'],
        desired_kl=spec['target_kl'],schedule='adaptive',use_clipped_value_loss=True,
        value_loss_coef=spec['value_coefficient'],entropy_coef=spec['entropy_coefficient'],
        max_grad_norm=spec['max_grad_norm'])
    if saved['optimizer'] is not None:alg.optimizer.load_state_dict(saved['optimizer'])
    logs=[];steps=[]
    def hook(opt,args,kwargs):
        norm=float(torch.sqrt(sum((x.grad.detach()**2).sum() for x in p.parameters() if x.grad is not None)))
        steps.append(dict(learning_rate=alg.learning_rate,clipped_gradient_norm=norm))
    def cap_learning_rate(opt,args,kwargs):
        # RSL's internal adaptive ceiling is .01; this experiment declares .001.
        alg.learning_rate=min(alg.learning_rate,spec.get('max_learning_rate',spec['learning_rate']))
        for group in opt.param_groups:group['lr']=alg.learning_rate
    cap_handle=alg.optimizer.register_step_pre_hook(cap_learning_rate)
    handle=alg.optimizer.register_step_post_hook(hook)
    for epoch in range(spec['epochs']):
        loss=alg.update()
        with torch.no_grad():
            p.act(td);lp=p.get_actions_log_prob(raw);lr=lp-old_lp
            kl=float((torch.log(p.action_std/old_sd)+(old_sd**2+(old_mu-p.action_mean)**2)/(2*p.action_std**2)-.5).sum(-1).mean())
            clip=float(((lr.exp()-1).abs()>spec['clip']).float().mean())
        total=loss['surrogate']+spec['value_coefficient']*loss['value']-spec['entropy_coefficient']*loss['entropy']
        if not all(np.isfinite(x) for x in [total,kl,*loss.values()]):raise ValueError('nonfinite RSL update')
        logs.append(dict(epoch=epoch,optimizer_updates=batches,total_loss=total,
            actor_loss=loss['surrogate'],critic_loss=loss['value'],entropy=loss['entropy'],
            post_update_kl=kl,post_update_clip_fraction=clip,learning_rate=alg.learning_rate,
            optimizer_steps=steps[-batches:]))
        if kl>spec.get('max_epoch_kl',.03):break
    handle.remove();cap_handle.remove()
    result={**state,'params':export(p),'torch_state':serialize_torch(p,alg.optimizer.state_dict(),alg.learning_rate),
            'total_updates':int(state['total_updates'])+len(steps)}
    with torch.no_grad():v=p.evaluate(td).flatten().numpy()
    variance=float(np.var(target.numpy()))
    metrics.update({k:float(np.mean([l[k] for l in logs])) for k in ['total_loss','actor_loss','critic_loss','entropy']})
    metrics.update(optimizer_updates=len(steps),post_update_kl=kl,post_update_clip_fraction=clip,
        value_explained_variance=float(1-np.var(target.numpy()-v)/variance) if variance>1e-12 else None,
        gradient_norm=max(s['clipped_gradient_norm'] for s in steps),learning_rate=alg.learning_rate,
        behavior_log_prob_max_error=error,behavior_value_max_error=value_error,
        lifetime_optimizer_updates=result['total_updates'],completed_epochs=len(logs))
    return result,metrics,learning,logs


def update_runtime(spec,output):
    from flax.serialization import msgpack_serialize
    from .exploration_loop import read,write
    from torch.utils.tensorboard import SummaryWriter
    import time
    start=time.monotonic();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    source=Path(spec['collection']);state=restore(source/'update_state.msgpack')
    with np.load(source/'prefixes.npz') as tape:
        state,metrics,learning,logs=update_batch(spec,state,tape,read(spec['feedback']))
    (output/'state.msgpack').write_bytes(msgpack_serialize(state))
    np.savez_compressed(output/'learning.npz',**learning)
    write(output/'optimizer_updates.json',logs);write(output/'hyperparameters.json',spec)
    metrics['wall_seconds']=time.monotonic()-start
    write(output/'metrics.json',metrics)
    write(output/'network_inventory.json',dict(backend=state['backend'],architecture=state['architecture'],
        parameters={name:sum(np.size(a) for layer in layers.values() for a in layer.values())
                    for name,layers in state['params'].items()},
        normalizer='frozen at explorer initialization; independent of changing base pi',
        gaussian_space='latent; environment executes tanh(raw)',
        total_updates=state['total_updates']))
    with SummaryWriter(str(output.parent.parent/'tensorboard')) as writer:
        for k,v in metrics.items():
            if isinstance(v,(int,float)) and not isinstance(v,bool):writer.add_scalar(k,v,spec['round_index']+1)
        for k,v in metrics['reward_components'].items():writer.add_scalar('reward/'+k,v,spec['round_index']+1)
        writer.add_text('hyperparameters',str({k:v for k,v in spec.items() if k not in ['input_files','source_locks']}),spec['round_index']+1)
    write(output/'status.json',dict(phase='completed',charged_interactions=0,wall_seconds=metrics['wall_seconds']))
