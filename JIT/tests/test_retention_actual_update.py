import jax
import jax.numpy as jp
import numpy as np

def test_weighted_same_batch_gradients_preserve_cancellation():
    from jit_dvgc.generative_bridge.actual_update import weighted_actor_metrics
    actor={'w':jp.array([2.,3.])}
    terms={'ppo':lambda a:jp.sum(a['w']), 'keep':lambda a:-.5*jp.sum(a['w'])}
    m=weighted_actor_metrics(actor,terms)
    np.testing.assert_allclose(m['actual/actor_gradient_sum_norm'],np.sqrt(.5))
    np.testing.assert_allclose(m['actual/cosine_ppo_keep'],-1)
    np.testing.assert_allclose(m['actual/gradient_norm_keep'],np.sqrt(.5))

def test_stage_resume_keeps_transition_clock(tmp_path):
    from jit_dvgc.generative_bridge.learner_continuation import LearnerHooks
    from brax.training.agents.ppo.train import TrainingState
    from brax.training.agents.ppo.losses import PPONetworkParams
    from brax.training import types
    import optax
    p=PPONetworkParams(policy={'w':jp.ones(2)},value={'w':jp.ones(2)})
    s=TrainingState(params=p,optimizer_state=optax.adam(.01).init(p),normalizer_params={'x':jp.ones(2)},env_steps=types.UInt64(hi=0,lo=50))
    first=LearnerHooks(tmp_path/'a',contract='c',trainer_sha='t',parent=None,preserve_stage_clock=True)
    first.initialize(s,jax.random.PRNGKey(2));first.save(50,s,jax.random.PRNGKey(2))
    second=LearnerHooks(tmp_path/'b',contract='c',trainer_sha='t',parent=first.latest_receipt(),preserve_stage_clock=True)
    restored,_=second.initialize(s.replace(env_steps=types.UInt64(hi=0,lo=0)),jax.random.PRNGKey(4))
    assert int(restored.env_steps)==50 and second.offset==0
    second.save(100,restored,jax.random.PRNGKey(2))
    assert second.latest_receipt()['lifetime_transitions']==100
