"""Synchronous deterministic action budgets anchored to a fixed parent bank.

Weighted quantiles use the left inverse empirical CDF (no interpolation).
Weights are normalized independently inside each group. This checks all bank
rows; it is finite-sample behavior protection, not a global retention theorem.
"""
import time
import numpy as np


def require_finite(tree, context):
    import jax
    for leaf in jax.tree.leaves(tree):
        if not np.isfinite(np.asarray(leaf)).all():
            raise FloatingPointError(f'nonfinite {context}; update not committed')


def guard_metrics(predict, actor, bank, config):
    require_finite(actor, 'Actor')
    obs=np.asarray(bank['observations']);ref=np.asarray(bank['reference_actions'])
    weights=np.asarray(bank['weights'],dtype=float);groups=np.asarray(bank['group_ids'])
    require_finite((obs,ref,weights),'guard inputs')
    if obs.ndim!=2 or obs.shape[1]!=76 or not len(obs) or ref.shape!=(len(obs),4) or weights.shape!=(len(obs),) or groups.shape!=(len(obs),):
        raise ValueError('guard bank requires complete 76D observations and four actions')
    if np.any(weights<0) or np.any(np.abs(ref)>1):raise ValueError('invalid guard weights or normalized actions')
    names=sorted(set(map(str,groups)))
    for field in ('mse_thresholds','p99_thresholds'):
        thresholds=config.get(field)
        if not isinstance(thresholds,dict) or set(thresholds)!=set(names):raise ValueError('threshold groups must match entire nonempty guard bank')
        for name in names:
            values=thresholds[name]
            if values is None or any(x is None for x in values):raise ValueError('unresolved guard thresholds')
            v=np.asarray(values,dtype=float)
            if v.shape!=(4,) or not np.isfinite(v).all() or np.any(v<0):raise ValueError('four finite nonnegative channel thresholds required')
    actions=np.asarray(predict(actor,obs));require_finite(actions,'candidate actions')
    if actions.shape!=ref.shape or np.any(np.abs(actions)>1):raise ValueError('invalid deterministic normalized candidate actions')
    errors=np.abs(actions-ref);result={'passed':True,'groups':{},'rejection_reasons':[]}
    for name in names:
        mask=groups.astype(str)==name;w=weights[mask]
        if w.sum()<=0:raise ValueError('guard group requires positive weight mass')
        w=w/w.sum();e=errors[mask]
        mse=np.sum(w[:,None]*e**2,axis=0);p99=[]
        for channel in range(4):
            order=np.argsort(e[:,channel],kind='stable')
            index=min(int(np.searchsorted(np.cumsum(w[order]),.99,side='left')),len(order)-1)
            p99.append(float(e[order[index],channel]))
        require_finite((mse,p99),'guard metrics')
        result['groups'][name]={'mse':mse.tolist(),'p99':p99,'rows':int(mask.sum())}
        for metric,values,field in [('mse',mse,'mse_thresholds'),('p99',p99,'p99_thresholds')]:
            for c,(value,threshold) in enumerate(zip(values,config[field][name])):
                if value>threshold:result['rejection_reasons'].append(f'{name}/{metric}/channel{c}')
    result['passed']=not result['rejection_reasons']
    return result


def guarded_adam_step(actor, opt_state, loss_fn, predict, bank, optimizer, config, *, proposal=None):
    """Evaluate one gradient and one optimizer proposal, then scale its delta.

    loss_fn must return a scalar. RNG/minibatch selection happens at the caller
    exactly once. Fatal nonfinite checks raise before any state is returned.
    """
    import jax
    import optax
    started=time.monotonic();require_finite((actor,opt_state),'input learner')
    if proposal is None:
        value,gradient=jax.value_and_grad(loss_fn)(actor)
        require_finite((value,gradient),'loss/gradient')
        delta,proposed=optimizer.update(gradient,opt_state,actor)
    else:
        value,gradient,delta,proposed=proposal
        require_finite((value,gradient),'loss/gradient')
    require_finite((delta,proposed),'Adam proposal')
    alphas=config.get('alphas',[1.,.5,.25,.125,.0625,.03125])
    if list(alphas)!=[1.,.5,.25,.125,.0625,.03125]:raise ValueError('fixed six backtracking alphas required')
    audit={'accepted':False,'alpha':None,'loss':float(value),'actor_grad_norm':float(optax.global_norm(gradient)),
        'original_delta_norm':float(optax.global_norm(delta)),'accepted_delta_norm':0.,'backtracking_evaluations':0,'candidates':[]}
    require_finite((audit['actor_grad_norm'],audit['original_delta_norm']),'proposal norms')
    for alpha in alphas:
        candidate=optax.apply_updates(actor,jax.tree.map(lambda d:alpha*d,delta))
        require_finite(candidate,'candidate parameters')
        metrics=guard_metrics(predict,candidate,bank,config)
        audit['backtracking_evaluations']+=1;audit['candidates'].append({'alpha':alpha,**metrics})
        if metrics['passed']:
            actual_norm=float(optax.global_norm(jax.tree.map(lambda c,a:c-a,candidate,actor)))
            require_finite(actual_norm,'accepted delta norm')
            audit.update(accepted=True,alpha=alpha,accepted_delta_norm=actual_norm,scaled_proposal_delta_norm=alpha*audit['original_delta_norm'],guard=metrics,wall_seconds=time.monotonic()-started)
            return candidate,proposed,audit
    audit.update(guard=metrics,wall_seconds=time.monotonic()-started,rejection_reason='blocked_by_guard')
    return actor,opt_state,audit
