from copy import deepcopy
import numpy as np
import pytest
import jax
import jax.numpy as jp
from jit_dvgc.generative_bridge import explorer_admission as a


def spec(ids=range(32)):
    return dict(controller_mode='learned_residual',explorer_backend='rsl_rl',num_envs=len(ids),
        pulse_steps=3,pulse_start_schedule=[0],delta_limit=[.25]*4,round_index=2,
        explorer_initialization=dict(mode='symmetric',latent_std=.6),
        explorer_admission_v1_2=dict(run_id='pilot',round=2,collection_id='round2fresh',
            master_seed=3,episode_ids=list(ids),uniform_episode_fraction=.2))


def test_mixture_is_episode_constant_partition_invariant_and_nontrivial():
    learned,draws=a.collection_mixture(spec(range(256)))
    x,y=a.collection_mixture(spec(range(31)));z,w=a.collection_mixture(spec(range(31,256)))
    np.testing.assert_array_equal(learned,np.concatenate((x,z)))
    np.testing.assert_array_equal(draws,np.concatenate((y,w)))
    assert 30 < (~learned).sum() < 80
    assert draws.shape==(256,3,4)
    with pytest.raises(ValueError,match='duplicate'):
        a.collection_mixture(spec([1,1]))
    bad=spec();bad['reuse_prefix_collection']='old'
    with pytest.raises(ValueError,match='fresh'):
        a.collection_mixture(bad)


def test_uniform_samples_never_have_fabricated_policy_logprob():
    raw,delta,lp,value,valid=a.mix_sample(jp.ones((2,4)),jp.ones((2,4))*.5,
        jp.array([-.5,-.6]),jp.ones(2),jp.array([True,False]),jp.ones((2,4))*.1,jp.ones(2,bool))
    np.testing.assert_array_equal(raw[0],1.)
    assert np.isnan(np.asarray(raw[1])).all() and np.isnan(lp[1])
    np.testing.assert_allclose(delta[1],.1)
    np.testing.assert_array_equal(valid,[True,False])
    assert value[1]==0


def receipt_and_tape():
    s=spec([1,2]);identity={'behavior_actor_sha256':'actor','behavior_normalizer_sha256':'norm'}
    s['explorer_admission_v1_2'].update(identity)
    receipt=dict(s['explorer_admission_v1_2'],fresh_collection=True,episode_modes=['learned','uniform'])
    tape=dict(mask=np.ones((3,2),bool),prefix_mask=np.ones((3,2),bool),
              explorer_learned=np.tile([True,False],(3,1)),on_policy_mask=np.tile([True,False],(3,1)),
              log_prob_valid=np.tile([True,False],(3,1)),log_prob=np.tile([-.2,np.nan],(3,1)),
              raw_action=np.tile(np.array([[.1]*4,[np.nan]*4]),(3,1,1)))
    return s,receipt,tape,identity


def test_admission_rejects_history_hash_drift_and_uniform_onpolicy_forgery():
    s,r,t,identity=receipt_and_tape()
    mask=a.validate_admission(s,r,t,identity)
    np.testing.assert_array_equal(mask,t['on_policy_mask'])
    for field,value in [('round',1),('run_id','old'),('behavior_actor_sha256','other'),
                         ('behavior_normalizer_sha256','other'),('fresh_collection',False)]:
        with pytest.raises(ValueError):a.validate_admission(s,{**r,field:value},t,identity)
    bad=deepcopy(t);bad['on_policy_mask'][:]=True
    with pytest.raises(ValueError):a.validate_admission(s,r,bad,identity)
    bad=deepcopy(t);bad['log_prob'][:,1]=-.2
    with pytest.raises(ValueError):a.validate_admission(s,r,bad,identity)
    bad=deepcopy(t);bad['raw_action'][0,0]=np.nan
    with pytest.raises(ValueError):a.validate_admission(s,r,bad,identity)


def test_update_claim_cannot_be_replayed(tmp_path):
    a.claim_update(tmp_path,{'run_id':'r','round':2,'collection_id':'c'})
    with pytest.raises(ValueError,match='already'):
        a.claim_update(tmp_path,{'run_id':'r','round':2,'collection_id':'c'})


def test_real_rsl_update_excludes_uniform_nan_samples_and_checks_actual_behavior():
    from jit_dvgc.rsl_pulse import initialize,infer,update_batch
    s=spec(range(8));s.update(seed=23,learning_rate=.001,epochs=1,minibatch_size=4,clip=.2,
        target_kl=.01,entropy_coefficient=.001,value_coefficient=.5,max_grad_norm=1.)
    state=initialize(s,np.zeros(106),np.ones(106))
    obs=np.random.default_rng(7).normal(size=(3,8,106)).astype(np.float32)
    mu,sd,value=map(np.asarray,infer(state,jp.asarray(obs)))
    np.testing.assert_array_equal(mu,0.)
    np.testing.assert_allclose(sd,.6,atol=1e-7)
    raw=mu+sd*.2;lp=(-.5*((raw-mu)/sd)**2-np.log(sd)-.5*np.log(2*np.pi)).sum(-1)
    learned=np.tile([True,False]*4,(3,1));raw[~learned]=np.nan;lp[~learned]=np.nan
    tape=dict(observation=obs,raw_action=raw,log_prob=lp,value=np.where(learned,value,0.),
        mask=np.ones((3,8),bool),prefix_mask=np.ones((3,8),bool),
        explorer_learned=learned,on_policy_mask=learned,log_prob_valid=learned)
    receipt=a.collection_receipt(s,state,learned[0]);s['explorer_admission_v1_2'].update(a.behavior_identity(state))
    s.update(_validated_explorer_admission=receipt,_validated_behavior_identity=a.behavior_identity(state))
    feedback=dict(eligible=[True]*8,rewards=[1,-9,-1,-9,1,-9,-1,-9],component_sums={})
    updated,metrics,learning,logs=update_batch(s,state,tape,feedback)
    assert metrics['effective_training_samples']==12 and metrics['eligible_episodes']==4
    np.testing.assert_array_equal(learning['mask'],learned)
    assert metrics['optimizer_updates']>0 and logs
    with pytest.raises(ValueError,match='behavior'):
        update_batch(s,updated,tape,feedback)
    # Low-level calls also cannot silently treat a mixture as ordinary PPO.
    legacy={k:v for k,v in s.items() if not k.startswith('_validated') and k!='explorer_admission_v1_2'}
    with pytest.raises(ValueError,match='admission'):
        update_batch(legacy,state,tape,feedback)


def test_disk_admission_binds_artifacts_and_rejects_replay_or_optout(tmp_path):
    from jit_dvgc.generative_bridge.contracts import file_sha
    import json
    s,r,t,_=receipt_and_tape()
    state=dict(params={'actor':{'weight':np.array([1.],np.float32)}},
               normalizer_mean=np.zeros(1,np.float32),normalizer_std=np.ones(1,np.float32))
    identity=a.behavior_identity(state);s['explorer_admission_v1_2'].update(identity);r.update(identity)
    for name in ('prefixes.npz','behavior.msgpack','update_state.msgpack'):
        (tmp_path/name).write_bytes(name.encode())
    r['files']={name:file_sha(tmp_path/name) for name in ('prefixes.npz','behavior.msgpack','update_state.msgpack')}
    (tmp_path/'explorer_admission.json').write_text(json.dumps(r))
    prepared=a.admit_update(s,tmp_path,state,t)
    assert prepared['_validated_behavior_identity']==identity
    with pytest.raises(ValueError,match='already'):a.admit_update(s,tmp_path,state,t)
    with pytest.raises(ValueError,match='bypass'):a.admit_update({},tmp_path,state,t)
    (tmp_path/'prefixes.npz').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='artifact drift'):a.admit_update(s,tmp_path,state,t)
