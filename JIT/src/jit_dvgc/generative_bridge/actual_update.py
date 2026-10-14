"""Opt-in telemetry from the exact loss call and accepted Adam parameter update."""
import ast
import inspect
import jax
import jax.numpy as jp


def norm(tree):
    return jp.sqrt(sum(jp.sum(jp.square(x)) for x in jax.tree.leaves(tree)))


def weighted_actor_metrics(actor,objectives,*,return_sum=False):
    gradients={name:jax.grad(fn)(actor) for name,fn in objectives.items()}
    result={f'actual/gradient_norm_{name}':norm(g) for name,g in gradients.items()}
    total=jax.tree.map(lambda *x:sum(x),*gradients.values())
    result['actual/actor_gradient_sum_norm']=norm(total)
    names=list(gradients)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            dot=sum(jp.sum(x*y) for x,y in zip(jax.tree.leaves(gradients[a]),jax.tree.leaves(gradients[b])))
            result[f'actual/cosine_{a}_{b}']=jp.clip(dot/jp.maximum(norm(gradients[a])*norm(gradients[b]),1e-20),-1.,1.)
    return (result,total) if return_sum else result


def adam_update_count(optimizer_state):
    """Find the one Adam counter through optional Optax transform nesting."""
    is_adam=lambda value: all(hasattr(value, key) for key in ('count','mu','nu'))
    states=[v for v in jax.tree.leaves(optimizer_state,is_leaf=is_adam) if is_adam(v)]
    if len(states)!=1:raise ValueError('exactly one Adam state required for update clock')
    return states[0].count


def adam_update_counts(optimizer_state):
    """Explicit branch clocks, retaining the historical single-Adam clock."""
    from .split_optimizer import SplitOptimizerState
    if isinstance(optimizer_state,SplitOptimizerState):
        return dict(actor_adam_count=adam_update_count(optimizer_state.actor),
                    critic_adam_count=adam_update_count(optimizer_state.critic))
    return dict(optimizer_updates=adam_update_count(optimizer_state))


def instrument_updates(trainer,sink,*,max_grad_norm,probe,record_optimizer_count=False,
                       split_clip_norms=None):
    """Observe pre-clipping gradients and actual Adam deltas without changing math."""
    source=getattr(trainer,'_jit_source_text',None) or inspect.getsource(trainer)
    tree=ast.parse(source);count=[0]
    class Hooks(ast.NodeTransformer):
        def visit_Assign(self,node):
            if len(node.targets)==1 and ast.unparse(node.targets[0])=='params' and ast.unparse(node.value)=='optax.apply_updates(params, params_update)':
                count[0]+=1
                arguments=', optimizer_state' if record_optimizer_count else ''
                return [*ast.parse('_actual_before_params = params').body,node,*ast.parse('metrics = _actual_update_metrics(_actual_before_params, params, grads, metrics, normalizer_params, ppo_network'+arguments+')').body]
            return self.generic_visit(node)
    tree=Hooks().visit(tree)
    if count!=[1]:raise ValueError('unsupported upstream optimizer update layout')
    def record(values):sink({k:float(v) for k,v in values.items()})
    probe=jp.asarray(probe)
    def update(before,after,grads,metrics,normalizer,network,optimizer_state=None):
        metrics=dict(metrics)
        expected_actor_gradient=metrics.pop('_actual_weighted_actor_gradient_sum',None)
        gn=norm(grads);scale=jp.minimum(1.,max_grad_norm/jp.maximum(gn,1e-20))
        delta=jax.tree.map(lambda a,b:a-b,after.policy,before.policy)
        prediction=lambda p:network.parametric_action_distribution.mode(network.policy_network.apply(normalizer,p,{'state':probe}))
        prefixes=('actual/','effective_lambda','retention_coefficient','ppo_loss','total_loss','demo_action_mse','retention_action_mse')
        if record_optimizer_count:prefixes+=('audit/','fixed_probe/')
        values={**{k:v for k,v in metrics.items() if k.startswith(prefixes)},
            'actual/global_actor_critic_gradient_norm':gn,'actual/global_clip_scale':scale,
            'actual/actor_gradient_norm':norm(grads.policy),'actual/critic_gradient_norm':norm(grads.value),
            'actual/actor_adam_delta_norm':norm(delta),
            'actual/critic_adam_delta_norm':norm(jax.tree.map(lambda a,b:a-b,after.value,before.value)),
            'actual/NEW_TRAIN_probe_action_change_mse':jp.mean(jp.square(prediction(after.policy)-prediction(before.policy)))}
        if split_clip_norms is not None:
            # A shared norm here is descriptive only; clipping is per branch.
            values.pop('actual/global_clip_scale')
            for branch,gradient,limit in zip(('actor','critic'),(grads.policy,grads.value),split_clip_norms):
                branch_norm=norm(gradient)
                branch_scale=jp.minimum(1.,limit/jp.maximum(branch_norm,1e-20))
                values['actual/'+branch+'_clip_scale']=branch_scale
                values['actual/'+branch+'_clipped_gradient_norm']=branch_norm*branch_scale
            if 'actual/gradient_norm_demo' in values and 'actual/gradient_norm_ppo' in values:
                values['actual/demo_to_ppo_gradient_norm_ratio']=values['actual/gradient_norm_demo']/jp.maximum(values['actual/gradient_norm_ppo'],1e-20)
        if 'actual/actor_gradient_sum_norm' in values:
            values['actual/actor_gradient_norm_sum_difference']=(
                norm(jax.tree.map(lambda a,b:a-b,grads.policy,expected_actor_gradient)) if expected_actor_gradient is not None
                else jp.abs(values['actual/actor_gradient_sum_norm']-values['actual/actor_gradient_norm']))
        if record_optimizer_count:
            values.update({'actual/'+k:v for k,v in adam_update_counts(optimizer_state).items()})
        jax.debug.callback(record,values,ordered=False)
        return {**metrics,**values}
    namespace={**trainer.__globals__,'_actual_update_metrics':update}
    tree=ast.fix_missing_locations(tree);exec(compile(tree,'<actual same update telemetry>','exec'),namespace)
    wrapped=namespace[trainer.__name__];wrapped._jit_source_text=ast.unparse(tree)
    return wrapped
