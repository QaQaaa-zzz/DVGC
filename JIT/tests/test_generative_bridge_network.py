import importlib
import numpy as np
import pytest


def test_network_count_shapes_and_grad():
    import jax
    import jax.numpy as jp
    m=importlib.import_module('jit_dvgc.generative_bridge.network')
    net=m.ConditionalUNet()
    x=jp.zeros((2,16,4));obs=jp.zeros((2,76));steps=jp.array([0,99])
    params=net.init(jax.random.PRNGKey(2),x,obs,steps)
    assert sum(v.size for v in jax.tree.leaves(params))==3131716
    y=net.apply(params,x,obs,steps)
    assert y.shape==x.shape and np.isfinite(y).all()
    value,grad=jax.value_and_grad(lambda p:jp.mean(net.apply(p,x,obs,steps)**2))(params)
    assert np.isfinite(value) and all(np.isfinite(v).all() for v in jax.tree.leaves(grad))
    assert net.apply(params,x[:1],obs[:1],steps[:1]).shape==(1,16,4)


def test_ddim_numpy_reference_and_no_latent_clipping():
    import jax.numpy as jp
    m=importlib.import_module('jit_dvgc.generative_bridge.diffusion')
    beta,ab=m.cosine_schedule()
    assert len(beta)==100 and np.all((beta>0)&(beta<=.999))
    latent=np.linspace(-3,3,64).reshape(1,16,4).astype(np.float32)
    reference=latent.copy()
    for k in range(99,-1,-5):
        clean=np.clip(reference/np.sqrt(ab[k]),-1,1)
        eps=(reference-np.sqrt(ab[k])*clean)/np.sqrt(1-ab[k])
        prev=ab[k-5] if k>=5 else 1.
        reference=np.sqrt(prev)*clean+np.sqrt(1-prev)*eps
    result=m.ddim_sample(lambda x,obs,k:jp.zeros_like(x),jp.zeros((1,76)),jp.asarray(latent))
    np.testing.assert_allclose(result,reference,atol=2e-5)
    assert m.noise_mse(jp.tile(jp.array([.2,-.1,0,.3]),(256,16,1)),jp.zeros((256,16,4)))==pytest.approx(.035)


def test_generator_no_data_no_state_or_rng_advance(tmp_path):
    m=importlib.import_module('jit_dvgc.generative_bridge.diffusion')
    incumbent={'weights':'unchanged','optimizer':'unchanged','rng':'unchanged'}
    selected,report=m.update_if_new_data(incumbent,{'new_data':False},updates=0,train=None)
    assert selected is incumbent and report['status']=='skipped_no_new_data'


def test_teacher_full_success_ranking_and_handoff_delta():
    m=importlib.import_module('jit_dvgc.generative_bridge.teacher')
    candidates=[dict(candidate_id='A',full_success=True,action_delta_cost=2.4,task_return=100),
        dict(candidate_id='B',full_success=True,action_delta_cost=.6,task_return=80),
        dict(candidate_id='C',full_success=False,action_delta_cost=.02,task_return=200)]
    assert m.select_teacher(candidates)['candidate_id']=='B'
    assert m.select_teacher(candidates[-1:]) is None
    assert m.action_delta_cost(np.array([[1,0,0,0],[0,0,0,0]]),np.zeros(4))==2


def test_full_generator_checkpoint_restores_optimizer_rng_and_incumbent_selection(tmp_path):
    import jax
    import jax.numpy as jp
    m=importlib.import_module('jit_dvgc.generative_bridge.diffusion')
    # Small real parameter tree exercises the same generic state/checkpoint path.
    state=m.create_train_state({'w':jp.ones((2,))},jax.random.PRNGKey(4),
                               {'mean':jp.zeros(76),'std':jp.ones(76)})
    identity={'model':'fixture_small_tree_not_production_G','dataset':'fixture','normalizer':'fixed'}
    m.save_state(tmp_path/'checkpoint',state,identity)
    restored=m.restore_state(tmp_path/'checkpoint',state,identity)
    for a,b in zip(jax.tree.leaves(state),jax.tree.leaves(restored)):np.testing.assert_array_equal(a,b)
    assert m.select_checkpoint([(4.,'incumbent'),(4.,'new')])=='incumbent'
    assert m.select_checkpoint([(4.,'incumbent'),(3.,'new')])=='new'
    with pytest.raises(ValueError):m.restore_state(tmp_path/'checkpoint',state,{**identity,'dataset':'other'})
    payload=tmp_path/'checkpoint/state.msgpack';payload.write_bytes(payload.read_bytes()+b'bad')
    with pytest.raises(ValueError):m.restore_state(tmp_path/'checkpoint',state,identity)


def test_generator_finite_step_and_restore_next_step_equivalence(tmp_path):
    import jax
    import jax.numpy as jp
    m=importlib.import_module('jit_dvgc.generative_bridge.diffusion')
    # Real optimizer update, scalar predictor: independent of expensive production U-Net.
    predict=lambda params,x,obs,k:jp.ones_like(x)*params['w']
    state=m.create_train_state({'w':jp.array(.1)},jax.random.PRNGKey(4),
                               {'mean':jp.zeros(76),'std':jp.ones(76)})
    step=m.make_train_step(predict,learning_rate=1e-5)
    obs=jp.ones((2,76));actions=jp.ones((2,16,4))*.2
    next_state,loss=step(state,obs,actions)
    assert np.isfinite(loss) and next_state['updates']==1
    m.save_state(tmp_path/'one',next_state,{'test':'fixture'})
    restored=m.restore_state(tmp_path/'one',state,{'test':'fixture'})
    a,_=step(next_state,obs,actions);b,_=step(restored,obs,actions)
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)
    with pytest.raises(FloatingPointError):step(state,obs.at[0,0].set(jp.nan),actions)
