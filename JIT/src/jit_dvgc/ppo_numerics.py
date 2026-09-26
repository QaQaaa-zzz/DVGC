"""Stop and preserve the first invalid PPO batch before optimizer mutation.

This is a diagnostic boundary, not NaN replacement or a stability guarantee.
The hook is scoped to a single synchronous trainer in its dedicated process.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import pickle
import jax
import jax.numpy as jp
import numpy as np


def _finite(tree):
    leaves=jax.tree.leaves(tree)
    return jp.all(jp.stack([jp.all(jp.isfinite(x)) for x in leaves])) if leaves else jp.asarray(True)


def checked_loss_and_grad(calculate, run_dir):
    root=Path(run_dir)
    def checked(*args, **kwargs):
        result=calculate(*args, **kwargs)
        def capture(values):
            inputs, outputs=values
            root.mkdir(parents=True,exist_ok=True)
            path=root/'nonfinite_update.pkl'
            if not path.exists():
                with path.open('xb') as stream:
                    pickle.dump(jax.device_get(values),stream)
                def valid(tree):
                    return all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(tree))
                report=dict(inputs_finite=valid(inputs),loss_finite=valid(outputs[0]),
                    gradients_finite=valid(outputs[1]),optimizer_update_applied=False)
                (root/'nonfinite_update.json').write_text(json.dumps(report,indent=2)+'\n')
            raise FloatingPointError('nonfinite PPO inputs/loss/gradient: captured before optimizer update')
        def invalid(_):
            jax.debug.callback(capture,((args,kwargs),result))
            return jp.asarray(0)
        valid=_finite((args,kwargs,result))
        jax.lax.cond(valid,lambda _:jp.asarray(0),invalid,None)
        # Connect validity to the optimizer guard, independent of callback order.
        grads=jax.tree.map(lambda g:jp.where(valid,g,jp.full_like(g,jp.nan)),result[1])
        return result[0],grads
    return checked


def finite_optimizer_update(optimizer, grads, state, params):
    """Reject an invalid proposal without advancing parameters or Adam state."""
    zeros=jax.tree.map(jp.zeros_like,grads)
    valid=_finite((grads,state,params))
    updates,candidate=jax.lax.cond(valid,
        lambda _:optimizer.update(grads,state,params),lambda _:(zeros,state),None)
    import optax
    proposed_params=optax.apply_updates(params,updates) if params is not None else ()
    accepted=valid & _finite((updates,candidate,proposed_params))
    updates,next_state=jax.lax.cond(accepted,lambda _:(updates,candidate),
        lambda _:(zeros,state),None)
    return updates,next_state,accepted


@contextmanager
def guard_ppo_updates(run_dir):
    import optax
    from brax.training import gradients
    original=gradients.loss_and_pgrad
    original_chain=optax.chain
    def guarded(loss_fn,pmap_axis_name,has_aux=False):
        return checked_loss_and_grad(original(loss_fn,pmap_axis_name,has_aux),run_dir)
    def guarded_chain(*transforms):
        optimizer=original_chain(*transforms)
        def update(grads,state,params=None):
            updates,next_state,accepted=finite_optimizer_update(optimizer,grads,state,params)
            def fail(_):
                def capture(values):
                    path=Path(run_dir)/'nonfinite_optimizer.pkl'
                    if not path.exists():
                        with path.open('xb') as stream:pickle.dump(jax.device_get(values),stream)
                    raise FloatingPointError('nonfinite PPO optimizer proposal rejected; parameters and optimizer state retained')
                jax.debug.callback(capture,(grads,state,params))
                return jp.asarray(0)
            # Invalid gradients already have one loss-guard capture callback.
            # A competing exception could suppress the original batch evidence.
            needs_capture=_finite(grads) & ~accepted
            jax.lax.cond(needs_capture,fail,lambda _:jp.asarray(0),None)
            return updates,next_state
        return optax.GradientTransformation(optimizer.init,update)
    gradients.loss_and_pgrad=guarded
    optax.chain=guarded_chain
    try:
        yield
    finally:
        gradients.loss_and_pgrad=original
        optax.chain=original_chain
