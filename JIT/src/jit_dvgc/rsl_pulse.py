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


NEIGHBORHOOD_FEATURE_LAYOUT = 'relative12_evidence4_valid1_near_far_stats8_v1'


def neighborhood_config(config):
    """Canonical network and observation semantics; dynamic map contents excluded."""
    if config is None:
        return None
    from .neighborhood import _config
    config = dict(config)
    # Collection uses Flax to_bytes, which encodes lists as index-keyed dicts;
    # update checkpoints use msgpack_serialize, which preserves ordinary lists.
    widths = config.get('medium_halfwidths')
    if isinstance(widths, dict):
        if set(widths) != {str(i) for i in range(12)}:
            raise ValueError('neighborhood checkpoint has invalid width indices')
        config['medium_halfwidths'] = [widths[str(i)] for i in range(12)]
    config = _config(config)
    defaults = dict(neighbors=16, feature_dim=17, summary_dim=64, base_dim=106, stats_dim=8)
    result = {}
    for key, default in defaults.items():
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'neighborhood {key} must be a positive integer')
        result[key] = int(value)
    if result['feature_dim'] < 2:
        raise ValueError('neighborhood feature_dim must include features and a valid mask')
    return dict(result, medium_halfwidths=list(config['medium_halfwidths']),
                far_scale=float(config['far_scale']))


def validate_neighborhood_identity(spec, state):
    """Reject changed feature semantics even when the raw input shape is equal.

    Omitted spec configuration is permitted for tape-only updates. Collection
    callers must additionally require the requested presence/absence to match.
    Legacy 106D checkpoints require no neighborhood metadata.
    """
    saved = state.get('neighborhood')
    if saved is not None:
        if (state.get('neighborhood_feature_layout') != NEIGHBORHOOD_FEATURE_LAYOUT or
                not {'medium_halfwidths', 'far_scale'}.issubset(saved)):
            raise ValueError('neighborhood checkpoint lacks a compatible complete observation identity')
    elif state.get('neighborhood_feature_layout') is not None or 'actor_encoder' in state.get('params', {}):
        raise ValueError('neighborhood checkpoint is missing observation configuration')
    config = neighborhood_config(saved)
    if 'neighborhood' in spec and neighborhood_config(spec['neighborhood']) != config:
        raise ValueError('neighborhood observation/architecture does not match checkpoint; initialize a fresh explorer')
    return config


def torch_policy(spec, state=None):
    import torch
    from importlib.metadata import version
    if version('rsl-rl-lib')!='3.2.0':raise ValueError('Pinned RSL-RL 3.2.0 required')
    from tensordict import TensorDict
    from rsl_rl.modules import ActorCritic
    config = neighborhood_config(spec.get('neighborhood'))
    if state is not None:
        config = validate_neighborhood_identity(spec, state)
    torch.set_num_threads(4)
    torch.manual_seed(spec['seed'])
    head_dim = 106 if config is None else config['base_dim'] + config['summary_dim'] + config['stats_dim']
    p=ActorCritic(TensorDict({'obs':torch.zeros(1,head_dim)},[1]),
        {'policy':['obs'],'critic':['obs']},4,
        actor_hidden_dims=[128]*3,critic_hidden_dims=[128]*3,activation='elu',
        state_dependent_std=True,noise_std_type='log',init_noise_std=.6)
    if config is not None:
        from .neighborhood_network import NeighborhoodNetwork
        p.actor = NeighborhoodNetwork(config, p.actor)
        p.critic = NeighborhoodNetwork(config, p.critic)
    if state is not None:
        saved=torch.load(io.BytesIO(state['torch_state']),map_location='cpu',weights_only=False)
        p.load_state_dict(saved['policy'])
    else:
        head=[m for m in p.actor.modules() if isinstance(m,torch.nn.Linear)][-1]
        with torch.no_grad():head.weight[:4].zero_();head.bias[:4].zero_()
    return p


def export(p):
    import torch
    modules = {}
    for name in ('actor', 'critic'):
        network = getattr(p, name)
        modules[name] = getattr(network, 'head', network)
        if hasattr(network, 'encoder'):
            modules[name + '_encoder'] = network.encoder
    return {name:{str(i):{'kernel':m.weight.detach().numpy().T.copy(),
                          'bias':m.bias.detach().numpy().copy()}
                  for i,m in enumerate(x for x in module.modules() if isinstance(x,torch.nn.Linear))}
            for name, module in modules.items()}


def serialize_torch(p, optimizer=None, lr=.01):
    import torch
    b=io.BytesIO();torch.save(dict(policy=p.state_dict(),optimizer=optimizer,
                                 learning_rate=lr,rng=torch.get_rng_state()),b)
    return b.getvalue()


def initialize(spec,mean,std):
    import jax
    p=torch_policy(spec)
    config = neighborhood_config(spec.get('neighborhood'))
    mean, std = np.asarray(mean,dtype=np.float32), np.asarray(std,dtype=np.float32)
    if config is not None:
        if mean.shape != (config['base_dim'],) or std.shape != mean.shape:
            raise ValueError('neighborhood initialization requires base-only normalizer arrays')
        context_dim = config['neighbors'] * config['feature_dim'] + config['stats_dim']
        mean = np.concatenate((mean, np.zeros(context_dim, np.float32)))
        std = np.concatenate((std, np.ones(context_dim, np.float32)))
    result = dict(params=export(p),normalizer_mean=mean,
        normalizer_std=std,rng=np.asarray(jax.random.PRNGKey(spec['seed'])),
        torch_state=serialize_torch(p,lr=spec['learning_rate']),total_updates=0,
        backend='rsl_rl_3.2.0',architecture='106_128_128_128_elu_state_dependent_log_std')
    if config is not None:
        result.update(neighborhood=config,
                      neighborhood_feature_layout=NEIGHBORHOOD_FEATURE_LAYOUT,
                      architecture='neighborhood_masked_mean_v1_128_128_128_elu_state_dependent_log_std')
    return result


def normalized(state,obs):
    return (obs-state['normalizer_mean'])/state['normalizer_std']


def infer(state,obs,precision="highest"):
    import jax
    import jax.numpy as jnp
    x=normalized(state,obs)
    def forward(layers, y, activate_last=False):
        for i in range(len(layers)):
            layer=layers[str(i)];y=jnp.matmul(y,jnp.asarray(layer['kernel']),precision=precision)+jnp.asarray(layer['bias'])
            if i<len(layers)-1 or activate_last:y=jax.nn.elu(y)
        return y
    def network(name):
        config = state.get('neighborhood')
        y = x
        if config is not None:
            end = config['base_dim'] + config['neighbors'] * config['feature_dim']
            rows = x[..., config['base_dim']:end].reshape(
                *x.shape[:-1], config['neighbors'], config['feature_dim'])
            valid = rows[..., -1:] > 0
            features = jnp.where(valid, rows[..., :-1], 0.)
            encoded = forward(state['params'][name + '_encoder'], features, activate_last=True)
            summary = jnp.where(valid, encoded, 0.).sum(axis=-2)
            summary = summary / jnp.maximum(valid.sum(axis=-2), 1)
            y = jnp.concatenate((x[..., :config['base_dim']], summary, x[..., end:]), axis=-1)
        return forward(state['params'][name], y)
    out=network('actor')
    return out[...,:4],jnp.exp(out[...,4:]),network('critic')[...,0]


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
    config = validate_neighborhood_identity({}, state)
    if config is not None:
        state['neighborhood'] = config
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
    # Validate the actual collection arithmetic, then bound distribution drift
    # against Torch. Legacy GPU default dots used reduced precision; do not
    # silently replace the saved behavior probabilities with recomputed ones.
    import jax
    import jax.numpy as jnp
    precision=spec.get('behavior_matmul_precision','highest')
    def replay(x,a):
        mu,sd,v=infer(state,x,precision=precision)
        lp=(-.5*((a-mu)/sd)**2-jnp.log(sd)-.5*math.log(2*math.pi)).sum(-1)
        return mu,sd,v,lp
    mu,sd,rv,rlp=map(np.asarray,jax.jit(replay)(
        jnp.asarray(np.asarray(tape['observation'])[mask]),jnp.asarray(raw.numpy())))
    replay_error=float(np.max(np.abs(rlp-old_lp.numpy())))
    replay_value_error=float(np.max(np.abs(rv-old_v.numpy())))
    with torch.no_grad():
        actual_mu=torch.from_numpy(mu.copy());actual_sd=torch.from_numpy(sd.copy())
        drift=(torch.log(old_sd/actual_sd)+(actual_sd**2+(actual_mu-old_mu)**2)/(2*old_sd**2)-.5).sum(-1)
        max_drift=float(drift.max())
    if (not np.isfinite([replay_error,replay_value_error,max_drift,value_error]).all()
            or replay_error>2e-3 or replay_value_error>2e-3 or max_drift>1e-5 or value_error>2e-3):
        raise ValueError(f'JAX/RSL behavior mismatch: replay={replay_error}, value={value_error}, distribution_KL={max_drift}')
    old_mu,old_sd=actual_mu,actual_sd
    metrics.update(behavior_replay_log_prob_error=replay_error,
                   behavior_distribution_max_kl=max_drift,behavior_matmul_precision=precision)
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
    collection_spec=read(source/'hyperparameters.json')
    spec={**spec,'behavior_matmul_precision':collection_spec.get('inference_precision','default')}
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
