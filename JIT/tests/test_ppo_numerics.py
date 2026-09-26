import json
import jax
import jax.numpy as jp
import pytest
from jit_dvgc.ppo_numerics import checked_loss_and_grad

def test_finite_gradient_is_unchanged(tmp_path):
 fn=checked_loss_and_grad(lambda x: ((x*x,{'loss':x*x}),2*x),tmp_path)
 value,grad=jax.jit(fn)(jp.array(3.))
 assert float(value[0])==9 and float(grad)==6
 assert not list(tmp_path.iterdir())

def test_bad_gradient_is_captured_before_returning_to_optimizer(tmp_path):
 fn=checked_loss_and_grad(lambda x: ((x*x,{'loss':x*x}),jp.array(float('nan'))),tmp_path)
 with pytest.raises(Exception,match='nonfinite PPO'):
  jax.block_until_ready(jax.jit(fn)(jp.array(3.)))
 report=json.loads((tmp_path/'nonfinite_update.json').read_text())
 assert report['inputs_finite'] and report['loss_finite'] and not report['gradients_finite']
 assert (tmp_path/'nonfinite_update.pkl').exists()

def test_bad_input_is_not_silently_sanitized(tmp_path):
 fn=checked_loss_and_grad(lambda x: ((jp.array(0.),{}),jp.array(0.)),tmp_path)
 with pytest.raises(Exception,match='nonfinite PPO'):
  jax.block_until_ready(jax.jit(fn)(jp.array(float('inf'))))
 assert not json.loads((tmp_path/'nonfinite_update.json').read_text())['inputs_finite']

def test_guard_restores_shared_library_hook_on_exception(tmp_path):
 from brax.training import gradients
 from jit_dvgc.ppo_numerics import guard_ppo_updates
 original=gradients.loss_and_pgrad
 with pytest.raises(RuntimeError):
  with guard_ppo_updates(tmp_path):
   assert gradients.loss_and_pgrad is not original
   raise RuntimeError('trainer failed')
 assert gradients.loss_and_pgrad is original

def test_compiled_optimizer_rejection_preserves_parameters_and_momentum():
 import optax
 from jit_dvgc.ppo_numerics import finite_optimizer_update
 optimizer=optax.adam(.01)
 params=jp.array([1.,2.]);state=optimizer.init(params)
 updates,state=optimizer.update(jp.array([.2,.3]),state,params)
 params=optax.apply_updates(params,updates)
 def step(p,s,g):
  updates,next_state,accepted=finite_optimizer_update(optimizer,g,s,p)
  return optax.apply_updates(p,updates),next_state,accepted
 result,new_state,accepted=jax.jit(step)(params,state,jp.array([float('nan'),1.]))
 assert not bool(accepted)
 for old,new in zip(jax.tree.leaves((params,state)),jax.tree.leaves((result,new_state))):
  assert bool(jp.array_equal(old,new))
 result,_,accepted=jax.jit(step)(params,state,jp.ones(2))
 assert bool(accepted) and not bool(jp.array_equal(result,params))

def test_compiled_loss_and_optimizer_emit_original_batch_diagnostic(tmp_path):
 import optax
 from brax.training import gradients
 from jit_dvgc.ppo_numerics import guard_ppo_updates
 with guard_ppo_updates(tmp_path):
  calc=gradients.loss_and_pgrad(lambda x:(jp.sqrt(x),{}),pmap_axis_name=None,has_aux=True)
  optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(.01))
  state=optimizer.init(jp.array(-1.))
  def step(x,s):
   _,g=calc(x)
   u,s=optimizer.update(g,s,x)
   return optax.apply_updates(x,u),s
  with pytest.raises(Exception,match='nonfinite PPO inputs/loss/gradient'):
   jax.block_until_ready(jax.jit(step)(jp.array(-1.),state))
 assert (tmp_path/'nonfinite_update.pkl').exists()
 assert not (tmp_path/'nonfinite_optimizer.pkl').exists()
