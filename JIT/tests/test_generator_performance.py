"""CPU invariants for numerical guards and lossless compact audit records."""
import numpy as np
import pytest


def test_guard_never_uses_host_tree_finite(monkeypatch):
    import jax
    import jax.numpy as jp
    from jit_dvgc.generative_bridge import diffusion as d
    state=d.create_train_state({'w':jp.array(.1)},jax.random.PRNGKey(9),
        {'mean':jp.zeros(76),'std':jp.ones(76)})
    def forbidden(tree):raise AssertionError('whole-tree host finite scan')
    monkeypatch.setattr(d,'_finite',forbidden)
    step=d.make_train_step(lambda p,x,o,k:jp.ones_like(x)*p['w'],learning_rate=1e-5)
    new,loss=step(state,jp.zeros((2,76)),jp.zeros((2,16,4)))
    assert int(new['updates'])==1 and np.isfinite(loss)


def test_compact_lossless_and_integrity(tmp_path):
    from jit_dvgc.generative_bridge.provenance import CompactProvenance, load_sample_provenance
    rows=[{'source_group':'history','root_episode_id':'ancestor','trajectory_sha256':'sha',
           'window_start':i,'window_start_step':None,'action_origins':['actor_only'],'onset':None}
          for i in (0,8,16)]
    writer=CompactProvenance(updates=2,batch_size=3)
    writer.append(rows,update=1,state_updates=11,sampling_seed=4294967295)
    writer.append(rows[::-1],update=2,state_updates=12,sampling_seed=7)
    pointer=writer.save(tmp_path)
    assert load_sample_provenance(tmp_path,pointer,0)==rows
    assert load_sample_provenance(tmp_path,pointer,1)==rows[::-1]
    arrays=tmp_path/pointer['arrays']
    arrays.write_bytes(arrays.read_bytes()+b'tampered')
    with pytest.raises(ValueError,match='hash'):load_sample_provenance(tmp_path,pointer,0)


@pytest.mark.parametrize('where,value',[('observation',np.nan),('action',np.inf),('params',np.nan),
                                        ('optimizer',np.inf),('ema',np.inf),('proposal',np.nan),('proposal',np.inf)])
def test_invalid_rejected_old_state_retained(where,value):
    import jax
    import jax.numpy as jp
    from jit_dvgc.generative_bridge import diffusion as d
    state=d.create_train_state({'w':jp.array(.1)},jax.random.PRNGKey(9),
        {'mean':jp.zeros(76),'std':jp.ones(76)})
    obs=jp.zeros((2,76));act=jp.zeros((2,16,4))
    if where=='observation':obs=obs.at[0,0].set(value)
    if where=='action':act=act.at[0,0,0].set(value)
    if where in ('params','ema'):state={**state,where:{'w':jp.array(value)}}
    if where=='optimizer':state={**state,'optimizer':jax.tree.map(lambda a:jp.full_like(a,value) if a.dtype.kind=='f' else a,state['optimizer'])}
    before=[np.asarray(a).copy() for a in jax.tree.leaves(state)]
    predict=lambda p,x,o,k:jp.ones_like(x)*(value if where=='proposal' else p['w'])
    with pytest.raises(FloatingPointError):d.make_train_step(predict,learning_rate=1e-5)(state,obs,act)
    for old,new in zip(before,jax.tree.leaves(state)):np.testing.assert_array_equal(old,new)


def test_fp32_matches_frozen_old_equations_and_scalar_transfers(monkeypatch):
    """Tolerance declared: FP32 atol=1e-7, rtol=2e-6 for full optimizer/EMA tree."""
    import jax
    import jax.numpy as jp
    import optax
    from jit_dvgc.generative_bridge import diffusion as d
    predict=lambda p,x,o,k:jp.tanh(x*p['w']+o[:,0,None,None]*p['bias'])
    state=d.create_train_state({'w':jp.array(.1),'bias':jp.array(.03)},jax.random.PRNGKey(91),
        {'mean':jp.zeros(76),'std':jp.ones(76)})
    observations=jp.linspace(-1,1,4*76).reshape(4,76)
    actions=jp.linspace(-.9,.9,4*64).reshape(4,16,4)
    ab=jp.asarray(d.cosine_schedule()[1]);optimizer=d._optimizer()
    @jax.jit
    def old_propose(state,observations,actions):
        rng,key_k,key_noise=jax.random.split(state['rng'],3)
        k=jax.random.randint(key_k,(len(actions),),0,100)
        eps=jax.random.normal(key_noise,actions.shape)
        alpha=ab[k,None,None]
        noisy=jp.sqrt(alpha)*actions+jp.sqrt(1-alpha)*eps
        obs=(observations-state['normalizer']['mean'])/state['normalizer']['std']
        value,grad=jax.value_and_grad(lambda p:d.noise_mse(predict(p,noisy,obs,k),eps))(state['params'])
        delta,optstate=optimizer.update(grad,state['optimizer'],state['params'])
        params=optax.apply_updates(state['params'],jax.tree.map(lambda x:1e-5*x,delta))
        ema=jax.tree.map(lambda a,b:.999*a+.001*b,state['ema'],params)
        return {**state,'params':params,'ema':ema,'optimizer':optstate,'rng':rng,
                'updates':state['updates']+1},value,grad
    expected,loss,_=old_propose(state,observations,actions)
    transfers=[];original=jax.device_get
    def record(tree):
        transfers.append([(np.shape(v),np.dtype(v.dtype).itemsize) for v in jax.tree.leaves(tree)])
        return original(tree)
    monkeypatch.setattr(jax,'device_get',record)
    got,value=d.make_train_step(predict,learning_rate=1e-5)(state,observations,actions)
    assert transfers==[[((2,),1)],[((),1),((),4)]]
    for a,b in zip(jax.tree.leaves(expected),jax.tree.leaves(got)):
        np.testing.assert_allclose(a,b,atol=1e-7,rtol=2e-6)
    np.testing.assert_allclose(value,loss,atol=1e-7,rtol=2e-6)


def test_provenance_failure_blocks_completed_publication(tmp_path,monkeypatch):
    import jax.numpy as jp
    from .test_generator_last_valid import fixture
    from jit_dvgc.generative_bridge import diffusion as d
    from jit_dvgc.generative_bridge.provenance import CompactProvenance
    state,corpus,dev=fixture()
    def fail(self,root):raise OSError('provenance fsync failed')
    monkeypatch.setattr(CompactProvenance,'save',fail)
    with pytest.raises(OSError,match='fsync failed'):
        d.train_incremental(state,corpus,predict=lambda p,x,o,k:jp.ones_like(x)*p['w'],
            dev_fixture=dev,output=tmp_path/'train',identity={},updates=1,batch_size=2,
            max_wall_seconds=60)
    import json
    failure=json.loads((tmp_path/'train/failure.json').read_text())
    assert failure['charged_updates']==1 and failure['completed_updates']==1
    assert not (tmp_path/'train/generator_selection.json').exists()


def test_action_bound_guard_preserves_error_and_old_state():
    import jax
    import jax.numpy as jp
    from jit_dvgc.generative_bridge import diffusion as d
    state=d.create_train_state({'w':jp.array(.1)},jax.random.PRNGKey(9),
        {'mean':jp.zeros(76),'std':jp.ones(76)})
    step=d.make_train_step(lambda p,x,o,k:jp.ones_like(x)*p['w'],learning_rate=1e-5)
    with pytest.raises(ValueError,match='action permission drift'):
        step(state,jp.zeros((1,76)),jp.ones((1,16,4))*1.01)
    assert int(state['updates'])==0
    new,_=step(state,jp.zeros((1,76)),jp.ones((1,16,4)))
    assert int(new['updates'])==1


def test_opt_in_exact_trajectory_identity_preserves_legacy_draws():
    from .test_generator_last_valid import fixture
    from jit_dvgc.generative_bridge.corpus_index import compile_corpus_index
    from jit_dvgc.generative_bridge.feedback_data import sample_corpus
    from jit_dvgc.generative_bridge.data import trajectory_identity
    _,corpus,_=fixture()
    expected=trajectory_identity(corpus['groups']['actor_new'][0])
    index=compile_corpus_index(corpus,include_trajectory_identity=True)
    a=sample_corpus(index,np.random.default_rng(41),7,return_metadata=True)
    b=sample_corpus(index,np.random.default_rng(41),7,return_metadata=True,include_trajectory_identity=True)
    np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
    assert a[2]==b[2]
    assert all(row['trajectory_sha256']==expected for row in b[3])
    assert a[3]==[{k:v for k,v in row.items() if k!='trajectory_sha256'} for row in b[3]]
