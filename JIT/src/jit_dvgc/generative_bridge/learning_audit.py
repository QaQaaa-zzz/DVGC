"""Fixed-observation probes and gradients from the installed Brax PPO loss."""
import inspect
import hashlib
import numpy as np


def action_probe(network, normalizer, actor, demo_manifest, *, reference=None):
    from .student import load_optional_demo
    from ..handoff_bank import pytree_sha256
    data=load_optional_demo(demo_manifest)
    result={'normalizer_sha256':pytree_sha256(normalizer),'actor_sha256':pytree_sha256(actor),
            'count':0 if data is None else len(data[0]),'groups':{}}
    if reference is not None:
        ref_norm,ref_actor=reference
        result['reference_normalizer_sha256']=pytree_sha256(ref_norm)
        result['reference_actor_sha256']=pytree_sha256(ref_actor)
        for field in ('mean','std'):
            if hasattr(normalizer,field) and hasattr(ref_norm,field):
                now=getattr(normalizer,field);old=getattr(ref_norm,field)
                if isinstance(now,dict):now=now['state'];old=old['state']
                result['normalizer_'+field+'_change']=(np.asarray(now)-np.asarray(old)).tolist()
    if data is None:return result
    obs,targets,_=data
    logits=network.policy_network.apply(normalizer,actor,{'state':obs})
    dist=network.parametric_action_distribution
    predictions=np.asarray(dist.mode(logits));error=(predictions-targets)**2
    roots=np.asarray(demo_manifest['sample_roots']);origins=np.asarray(demo_manifest['sample_origins'])
    for root in sorted(set(roots)):
        for origin in ('bridge_prefix','source_tail'):
            mask=(roots==root)&(origins==origin)
            if not mask.any():continue
            mse=error[mask].mean(axis=0)
            result['groups'][root+'/'+origin]={'count':int(mask.sum()),'mse_channels':mse.tolist(),'rmse_channels':np.sqrt(mse).tolist()}
    distribution=dist.create_dist(logits)
    result['action_saturation_fraction']=float(np.mean(np.abs(predictions)>=.99))
    if hasattr(distribution,'scale'):
        result['log_std_mean_channels']=np.log(np.asarray(distribution.scale)).mean(axis=0).tolist()
    if reference is not None:
        ref_norm,ref_actor=reference
        original=np.asarray(dist.mode(network.policy_network.apply(ref_norm,ref_actor,{'state':obs})))
        result['reference_mse_channels']=((original-targets)**2).mean(axis=0).tolist()
        result['action_change_mse_channels']=((predictions-original)**2).mean(axis=0).tolist()
    return result


def gradient_audit(network, params, normalizer, data, rng, *, demo_loss, keep_loss, options=None):
    """No optimizer update. PPO Actor gradient excludes value and entropy terms.

    Call with a saved fixed on-policy batch and the exact behavior normalizer.
    demo_loss/keep_loss are scalar actor-only functions using fixed probe data.
    """
    import jax
    import jax.numpy as jp
    from jax.flatten_util import ravel_pytree
    from brax.training.agents.ppo import losses
    options=options or {}
    def metric(p,name):return losses.compute_ppo_loss(p,normalizer,data,rng,network,**options)[1][name]
    objectives={'ppo_policy':lambda actor:metric(params.replace(policy=actor),'policy_loss'),
                'demo':demo_loss,'keep':keep_loss}
    gradients={name:np.asarray(ravel_pytree(jax.grad(fn)(params.policy))[0],dtype=np.float64) for name,fn in objectives.items()}
    norms={name:float(np.linalg.norm(g)) for name,g in gradients.items()}
    cosines={}
    for a,b in (('ppo_policy','demo'),('ppo_policy','keep'),('demo','keep')):
        denominator=norms[a]*norms[b]
        cosines[a+'/'+b]=float(np.vdot(gradients[a],gradients[b])/denominator) if denominator else None
    critic=ravel_pytree(jax.grad(lambda value:metric(params.replace(value=value),'v_loss'))(params.value))[0]
    _,metrics=losses.compute_ppo_loss(params,normalizer,data,rng,network,**options)
    logits=network.policy_network.apply(normalizer,params.policy,data.observation)
    logp=network.parametric_action_distribution.log_prob(logits,data.extras['policy_extras']['raw_action'])
    delta=logp-data.extras['policy_extras']['log_prob'];ratio=jp.exp(delta)
    # Match installed Brax time-major GAE for scalar critic explained variance.
    time_data=jax.tree.map(lambda x:jp.swapaxes(x,0,1),data)
    baseline=network.value_network.apply(normalizer,params.value,time_data.observation)
    explained_variance=None
    if not options.get('use_distributional_critic',False):
        bootstrap=network.value_network.apply(normalizer,params.value,jax.tree.map(lambda x:x[-1],time_data.next_observation))
        trunc=time_data.extras['state_extras']['truncation']
        vs,_=losses.compute_gae(truncation=trunc,termination=(1-time_data.discount)*(1-trunc),
            rewards=time_data.reward*options.get('reward_scaling',1.),values=baseline,bootstrap_value=bootstrap,
            lambda_=options.get('gae_lambda',.95),discount=options.get('discounting',.9))
        variance=float(jp.var(vs))
        if variance>0:explained_variance=float(1-jp.var(vs-baseline)/variance)
    source=inspect.getsource(losses.compute_ppo_loss)
    current_dist=network.parametric_action_distribution.create_dist(logits)
    kl_source=inspect.getsource(type(current_dist).kl_divergence) if hasattr(current_dist,'kl_divergence') else ''
    self_kl=float(jp.mean(current_dist.kl_divergence(current_dist))) if kl_source else None
    result=dict(actor_gradient_norms=norms,actor_gradient_cosines=cosines,
        critic_value_gradient_norm=float(jp.linalg.norm(critic)),
        installed_brax_loss_sha256=hashlib.sha256(source.encode()).hexdigest(),
        critic_explained_variance=explained_variance,
        installed_kl_mean=float(metrics['kl_mean']),kl_definition='installed Brax mean KL(saved behavior || current), pre-tanh; includes log ratio +1e-5 numerical offset',
        installed_self_kl=self_kl,installed_kl_implementation=kl_source,
        sampled_log_ratio_mean=float(jp.mean(delta)),clip_fraction=float(jp.mean(jp.abs(ratio-1)>options.get('clipping_epsilon',.3))),
        behavior_logprob_max_abs_difference=float(jp.max(jp.abs(delta))),optimizer_updates=0)
    if not all(np.isfinite(v) for v in [*norms.values(),result['critic_value_gradient_norm'],result['installed_kl_mean']]):
        raise FloatingPointError('nonfinite learning audit')
    return result


def joint_loss_metrics(network,actor,normalizer,data,ppo_objective,demo_objective,keep_objective,*,clipping_epsilon=.3):
    """JAX-transformable metrics in the actual guarded PPO loss execution."""
    import jax
    import jax.numpy as jp
    from jax.flatten_util import ravel_pytree
    objectives={'ppo_policy':ppo_objective,'demo':demo_objective,'keep':keep_objective}
    grads={name:ravel_pytree(jax.grad(fn)(actor))[0] for name,fn in objectives.items()}
    norms={name:jp.linalg.norm(value) for name,value in grads.items()}
    result={'audit/actor_gradient_norm/'+name:value for name,value in norms.items()}
    for a,b in (('ppo_policy','demo'),('ppo_policy','keep'),('demo','keep')):
        denominator=norms[a]*norms[b]
        result['audit/actor_gradient_cosine/'+a+'/'+b]=jp.where(denominator>0,jp.vdot(grads[a],grads[b])/jp.maximum(denominator,1e-30),0.)
        result['audit/actor_gradient_cosine_defined/'+a+'/'+b]=(denominator>0).astype(jp.float32)
    logits=network.policy_network.apply(normalizer,actor,data.observation)
    dist=network.parametric_action_distribution
    delta=dist.log_prob(logits,data.extras['policy_extras']['raw_action'])-data.extras['policy_extras']['log_prob']
    current=dist.create_dist(logits);old=dist.create_dist(data.extras['policy_extras']['distribution_params'])
    result.update({'audit/behavior_logprob_max_abs_difference':jp.max(jp.abs(delta)),
        'audit/clip_fraction':jp.mean(jp.abs(jp.exp(delta)-1)>clipping_epsilon),
        'audit/installed_self_kl':jp.mean(current.kl_divergence(current)),
        'audit/installed_behavior_kl':jp.mean(current.kl_divergence(old))})
    return result


def warmup_inference_override(source_payload, frozen_policy, reference):
    """In-memory inference candidate; original checkpoint identity stays provenance.

    This does not freeze, adopt, serialize as a formal checkpoint, or claim PPO
    optimizer restoration. The caller must label candidate evaluation explicitly.
    """
    import pickle
    import jax
    from dataclasses import replace
    from pathlib import Path
    from .contracts import file_sha
    from ..handoff_bank import pytree_sha256
    policy=frozen_policy.get('policy',frozen_policy)
    if file_sha(reference['path'])!=reference['sha256']:raise ValueError('warmup checkpoint hash drift')
    for key,value in (('actor_sha256',source_payload.actor_params),('normalizer_sha256',source_payload.observation_normalizer)):
        if pytree_sha256(value)!=policy[key]:raise ValueError('source payload identity mismatch')
    if reference['source_actor_sha256']!=policy['actor_sha256']:raise ValueError('warmup source identity mismatch')
    with Path(reference['path']).open('rb') as stream:values=pickle.load(stream)
    if not isinstance(values,tuple) or len(values)!=3:raise ValueError('warmup inference tuple required')
    norm,actor,_=values
    if pytree_sha256(norm)!=reference['normalizer_sha256'] or reference['normalizer_sha256']!=policy['normalizer_sha256']:
        raise ValueError('warmup normalizer identity mismatch')
    old_leaves,old_structure=jax.tree.flatten(source_payload.actor_params)
    leaves,structure=jax.tree.flatten(actor)
    if structure!=old_structure or any(x.shape!=y.shape for x,y in zip(leaves,old_leaves)):
        raise ValueError('warmup actor architecture mismatch')
    if not all(np.isfinite(x).all() for x in leaves):raise ValueError('nonfinite warmup Actor')
    candidate=replace(source_payload,actor_params=actor,observation_normalizer=norm)
    metadata={**policy,'actor_sha256':pytree_sha256(actor),'normalizer_sha256':pytree_sha256(norm),
        'inference_override':dict(reference),'checkpoint_role':'source_provenance_only_with_explicit_warmup_override',
        'optimizer_restored':False,'adopted':False}
    return candidate,metadata


def fixed_weighted_probe_indices(weights,limit=256):
    """Deterministic equal-mass strata of the declared sampling distribution.

    Midpoint inverse-CDF selection may repeat high-mass rows. These bounded
    telemetry probes never replace the full training sampling distribution.
    """
    weights=np.asarray(weights,np.float64)
    if type(limit) is not int or not 1<=limit<=256:raise ValueError('probe limit must be 1..256')
    if weights.ndim!=1 or not len(weights) or not np.isfinite(weights).all() or np.any(weights<0) or weights.sum()<=0:
        raise ValueError('positive finite probe mass required')
    count=min(limit,len(weights));cdf=np.cumsum(weights/weights.sum());cdf[-1]=1.
    return np.searchsorted(cdf,(np.arange(count)+.5)/count,side='right')


def critic_loss_metrics(network,critic,normalizer,data,value_objective,*,options=None):
    """Exact installed value-loss gradient, with its time-major GAE targets.

    A zero target variance is explicitly undefined, represented by a separate
    flag and a finite zero placeholder so telemetry cannot poison PPO guards.
    """
    import jax
    import jax.numpy as jp
    from jax.flatten_util import ravel_pytree
    from brax.training.agents.ppo import losses
    options=options or {}
    grad=ravel_pytree(jax.grad(value_objective)(critic))[0]
    time_data=jax.tree.map(lambda x:jp.swapaxes(x,0,1),data)
    apply=network.value_network.apply
    baseline=apply(normalizer,critic,time_data.observation)
    bootstrap=apply(normalizer,critic,jax.tree.map(lambda x:x[-1],time_data.next_observation))
    if options.get('use_distributional_critic',False):
        baseline=baseline[0];bootstrap=bootstrap[0]
    truncation=time_data.extras['state_extras']['truncation']
    targets,_=losses.compute_gae(truncation=truncation,
        termination=(1-time_data.discount)*(1-truncation),
        rewards=time_data.reward*options.get('reward_scaling',1.),values=baseline,
        bootstrap_value=bootstrap,lambda_=options.get('gae_lambda',.95),
        discount=options.get('discounting',.9))
    variance=jp.var(targets)
    defined=variance>0
    explained=jp.where(defined,1-jp.var(targets-baseline)/jp.maximum(variance,1e-30),0.)
    return {'audit/critic_value_gradient_norm':jp.linalg.norm(grad),
        'audit/critic_explained_variance':explained,
        'audit/critic_explained_variance_defined':defined.astype(jp.float32),
        'audit/critic_gae_target_variance':variance}
