import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import jax,jax.numpy as jnp
from brax.training import types
from brax.training.agents.ppo.losses import compute_gae
from brax_physx_learner import Learner

class Stability(unittest.TestCase):
 def setUp(self):
  p={'seed':1,'batch_size':2,'num_minibatches':1,'num_updates_per_batch':2,'unroll_length':3,'num_parallel_envs':2,'max_grad_norm':.5,'learning_rate':.001,'entropy_cost':.001,'discounting':.99,'reward_scaling':.1,'gae_lambda':.95,'clipping_epsilon':.1,'stability':{'enabled':True,'freeze_normalizer':True,'soft_kl':.02,'hard_kl':.05}}
  self.l=Learner(None,p)
  obs={'state':np.random.default_rng(1).normal(size=(2,3,76)).astype('float32'),'privileged_state':np.zeros((2,3,106),'float32')}
  a,e=self.l.sample(obs,key=jax.random.PRNGKey(7))
  self.d=types.Transition(observation=obs,next_observation=obs,action=a,reward=np.ones((2,3),'float32'),discount=np.ones((2,3),'float32'),extras={'policy_extras':e,'state_extras':{'truncation':np.zeros((2,3)),'time_out':np.zeros((2,3))}})
 def test_no_update_and_distribution(self):
  m=self.l.rollout_policy_metrics(self.d)
  self.assertLess(m['ratio_max_abs_error_from_one'],1e-6);self.assertAlmostEqual(m['full_rollout_kl_new_old'],0,places=6)
  e=self.d.extras['policy_extras'];np.testing.assert_allclose(np.tanh(e['raw_action']),self.d.action,atol=1e-6)
  dist=self.l.network.parametric_action_distribution
  raw=jnp.asarray(e['raw_action']);gauss=dist.create_dist(e['distribution_params'])
  jac=2*(jnp.log(2.)-raw-jax.nn.softplus(-2*raw))
  expected=jnp.sum(gauss.log_prob(raw)-jac,axis=-1)
  np.testing.assert_allclose(e['log_prob'],expected,atol=1e-6)

 def test_full_rollback_including_normalizer_and_adam(self):
  self.l.ppo['stability'].update(soft_kl=0.,hard_kl=0.,freeze_normalizer=True)
  old=[np.array(x) for x in jax.tree.leaves((self.l.params,self.l.optimizer_state,self.l.normalizer))]
  m=self.l.update(self.d,key=jax.random.PRNGKey(2));self.assertEqual(m['hard_rejected'],1)
  for x,y in zip(old,jax.tree.leaves((self.l.params,self.l.optimizer_state,self.l.normalizer))):np.testing.assert_array_equal(x,y)
  self.assertEqual(self.l.training_transitions,6);self.assertEqual(self.l.optimizer_steps,0)
  self.assertLessEqual(m['grad_norm_post_clip'],.5)
 def test_normalizer_candidate_rollback(self):
  self.l.ppo['stability'].update(soft_kl=0.,hard_kl=0.,freeze_normalizer=False)
  old=[np.array(x) for x in jax.tree.leaves((self.l.params,self.l.optimizer_state,self.l.normalizer))]
  m=self.l.update(self.d,key=jax.random.PRNGKey(2));self.assertEqual(m['hard_rejected'],1)
  for x,y in zip(old,jax.tree.leaves((self.l.params,self.l.optimizer_state,self.l.normalizer))):np.testing.assert_array_equal(x,y)
 def test_soft_stop_and_freeze(self):
  self.l.ppo['stability'].update(soft_kl=0.,hard_kl=100.)
  count=np.asarray(self.l.normalizer.count).copy()
  m=self.l.update(self.d,key=jax.random.PRNGKey(2))
  self.assertEqual(m['soft_stopped'],1);self.assertEqual(m['optimizer_steps_this_update'],1)
  np.testing.assert_array_equal(self.l.normalizer.count,count)
 def test_invalid_samples(self):
  with self.assertRaises(ValueError):self.l.update(self.d._replace(reward=np.full((2,3),np.nan)))
  with self.assertRaises(ValueError):self.l.update(self.d._replace(reward=np.zeros((0,3))))
 def test_gae_terminal_no_bootstrap(self):
  v,a=compute_gae(truncation=jnp.zeros((1,1)),termination=jnp.ones((1,1)),rewards=jnp.array([[2.]]),values=jnp.zeros((1,1)),bootstrap_value=jnp.array([100.]),lambda_=.95,discount=.99)
  np.testing.assert_allclose(v,[[2.]]);np.testing.assert_allclose(a,[[2.]])
if __name__=='__main__':unittest.main()
