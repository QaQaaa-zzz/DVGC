import importlib
import json
import pickle
from types import SimpleNamespace as NS
import jax
import jax.numpy as jp
import numpy as np
import pytest


def api():
    assert importlib.util.find_spec('jit_dvgc.recovery_acquisition_bc') is not None, 'dedicated demo-only BC module missing'
    return importlib.import_module('jit_dvgc.recovery_acquisition_bc')


def fixture(tmp_path):
    from jit_dvgc.generative_bridge.contracts import file_sha
    path=tmp_path/'demo.npz'
    np.savez(path,observations=np.ones((8,76),np.float32),actions=np.full((8,4),.7,np.float32),weights=np.ones(8))
    manifest=dict(schema='jit_bridge_demo_v1_2',count=8,path=str(path),sha256=file_sha(path),
        sample_roots=['a']*6+['b']*2,sample_origins=['bridge_prefix']+['source_tail']*5+['bridge_prefix','source_tail'])
    network=NS(policy_network=NS(apply=lambda n,p,o:jp.broadcast_to(p,(len(o['state']),4))),
        parametric_action_distribution=NS(mode=lambda x:jp.tanh(x)))
    return network,(jp.array(2.),jp.zeros(4),jp.array(8.)),manifest


def equal(a,b):
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)


def test_root_segment_mass_and_filter(tmp_path):
    _,_,manifest=fixture(tmp_path)
    weights=api().recovery_sample_weights(manifest)
    np.testing.assert_allclose(weights,[.25]+[.05]*5+[.25,.25])
    filtered=api().recovery_sample_weights(manifest,['a'])
    np.testing.assert_allclose(filtered,[.5]+[.1]*5+[0.,0.])
    with pytest.raises(ValueError):api().recovery_sample_weights(manifest,['absent'])


def test_demo_only_learning_and_frozen_inference(tmp_path):
    net,init,demo=fixture(tmp_path);rows=[]
    result=api().warmup_recovery_actor(net,init,demo,tmp_path/'run',updates=4,batch_size=4,
        checkpoint_steps=[0,4],learning_rate=.1,metrics_callback=rows.append)
    assert rows[-1]['loss']<rows[0]['loss']
    equal((result[0],result[2]),(init[0],init[2]))
    with (tmp_path/'run/update_0004.pkl').open('rb') as f:equal(pickle.load(f),result)
    status=json.loads((tmp_path/'run/warmup_status.json').read_text())
    assert status['completed_updates']==4 and status['environment_interactions']==0
    assert sum(status['sample_counts'].values())==16


def test_full_resume_next_update_equivalence_and_contract(tmp_path):
    net,init,demo=fixture(tmp_path)
    fn=api().warmup_recovery_actor
    full=fn(net,init,demo,tmp_path/'full',updates=5,batch_size=3,checkpoint_steps=[0,2,5])
    fn(net,init,demo,tmp_path/'first',updates=2,batch_size=3,checkpoint_steps=[0,2])
    resume=tmp_path/'first/learner_update_0002.pkl'
    continued=fn(net,init,demo,tmp_path/'next',updates=5,batch_size=3,checkpoint_steps=[2,5],resume_checkpoint=resume)
    equal(full,continued)
    for key in ('actor','optimizer_state','rng','sample_counts'):
        with (tmp_path/'full/learner_update_0005.pkl').open('rb') as f:a=pickle.load(f)[key]
        with (tmp_path/'next/learner_update_0005.pkl').open('rb') as f:b=pickle.load(f)[key]
        if key=='sample_counts':assert a==b
        else:equal(a,b)
    with pytest.raises(ValueError,match='contract'):
        fn(net,init,demo,tmp_path/'wrong',updates=5,batch_size=4,resume_checkpoint=resume)


def test_checkpoint_zero_callback_stops_and_schedule_budget(tmp_path):
    net,init,demo=fixture(tmp_path);called=[]
    result=api().warmup_recovery_actor(net,init,demo,tmp_path/'stop',updates=3001,
        checkpoint_steps=[0,3001],checkpoint_callback=lambda i,p:called.append(i) or True)
    assert called==[0];equal(result,init)
    with pytest.raises(ValueError,match='schedule'):
        api().warmup_recovery_actor(net,init,demo,tmp_path/'bad',updates=3,checkpoint_steps=[0,4])


def test_nonfinite_update_is_rejected_and_batch_preserved(tmp_path):
    net,init,demo=fixture(tmp_path)
    net.policy_network.apply=lambda n,p,o:jp.broadcast_to(p*jp.nan,(len(o['state']),4))
    with pytest.raises(FloatingPointError):
        api().warmup_recovery_actor(net,init,demo,tmp_path/'bad',updates=1,checkpoint_steps=[0,1])
    status=json.loads((tmp_path/'bad/warmup_status.json').read_text())
    assert status['completed_updates']==0 and status['status']=='error'
    assert (tmp_path/'bad/nonfinite_batch.npz').exists()
