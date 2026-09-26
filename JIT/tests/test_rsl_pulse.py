import numpy as np
import pytest

def test_rsl_inference_and_persistent_update():
    from jit_dvgc.rsl_pulse import initialize, infer, update_batch, torch_policy
    import torch
    import jax.numpy as jp
    spec=dict(seed=23,learning_rate=.001,epochs=2,minibatch_size=8,clip=.2,
              target_kl=.01,entropy_coefficient=.001,value_coefficient=.5,max_grad_norm=1.)
    state=initialize(spec,np.full(106,2.),np.full(106,3.))
    obs=np.random.default_rng(7).normal(size=(3,8,106)).astype('float32')
    mu,sd,val=infer(state,jp.asarray(obs))
    policy=torch_policy(spec,state)
    from tensordict import TensorDict
    td=TensorDict({'obs':torch.tensor((obs.reshape(-1,106)-2)/3)},[24])
    with torch.no_grad():
        policy.act(td)
        np.testing.assert_allclose(np.asarray(mu).reshape(-1,4),policy.action_mean.numpy(),atol=1e-6)
        np.testing.assert_allclose(np.asarray(sd).reshape(-1,4),policy.action_std.numpy(),rtol=1e-5)
        np.testing.assert_allclose(np.asarray(val).reshape(-1),policy.evaluate(td).numpy().ravel(),atol=1e-6)
    raw=np.asarray(mu)+np.asarray(sd)*np.random.default_rng(4).normal(size=mu.shape)
    lp=(-.5*((raw-mu)/sd)**2-np.log(sd)-.5*np.log(2*np.pi)).sum(-1)
    mask=np.ones((3,8),bool);mask[1:,0]=False
    feedback={'eligible':[True]*7+[False],'rewards':[1,-1,1,-1,1,-1,1,0],
              'component_sums':{'novelty':0,'quality':1}}
    tape=dict(observation=obs,raw_action=raw,log_prob=lp,value=np.asarray(val),mask=mask)
    updated,metrics,learning,logs=update_batch(spec,state,tape,feedback)
    assert metrics['effective_training_samples']==19
    assert metrics['optimizer_updates']>0
    assert all(step['learning_rate']<=.001 for log in logs for step in log['optimizer_steps'])
    assert metrics['backend']=='rsl_rl_3.2.0'
    assert np.isfinite(metrics['post_update_kl'])
    np.testing.assert_array_equal(state['normalizer_mean'],updated['normalizer_mean'])
    assert updated['total_updates']>0
    from flax.serialization import msgpack_serialize,msgpack_restore
    updated=msgpack_restore(msgpack_serialize(updated))
    with pytest.raises(ValueError,match='behavior mismatch'):
        update_batch(spec,state,dict(tape,log_prob=np.asarray(lp)+1),feedback)
    again,m,_,_=update_batch(spec,updated,dict(tape,**fresh_tape(updated,obs,mask)),feedback)
    assert again['total_updates']>updated['total_updates']

def fresh_tape(state,obs,mask):
    from jit_dvgc.rsl_pulse import infer
    import jax.numpy as jp
    mu,sd,val=map(np.asarray,infer(state,jp.asarray(obs)))
    raw=mu+sd*.2
    lp=(-.5*((raw-mu)/sd)**2-np.log(sd)-.5*np.log(2*np.pi)).sum(-1)
    return dict(raw_action=raw,log_prob=lp,value=val)

def test_promotion_keeps_explorer_checkpoint():
    from jit_dvgc.pulse_exploration import promoted_explorer_checkpoint
    assert promoted_explorer_checkpoint({'explorer_backend':'rsl_rl'},'trained.msgpack')=='trained.msgpack'

def test_collection_inventory_accepts_rsl_parameters():
    from jit_dvgc.pulse_exploration_runtime import explorer_inventory
    from jit_dvgc.rsl_pulse import initialize
    state=initialize({'seed':1,'learning_rate':.001},np.zeros(106),np.ones(106))
    inv=explorer_inventory({'explorer_backend':'rsl_rl'},state['params'])
    assert inv['hidden']==[128,128,128]
    assert inv['actor_parameters']==47752
    assert inv['critic_parameters']==46849
