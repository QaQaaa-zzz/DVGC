import numpy as np
import pytest
from types import SimpleNamespace as NS
import jax.numpy as jp

def fixture(tmp_path):
    from jit_dvgc.generative_bridge.contracts import file_sha
    path=tmp_path/'demo.npz'
    np.savez(path,observations=np.ones((8,76),np.float32),actions=np.full((8,4),.7,np.float32),weights=np.ones(8))
    manifest=dict(count=8,path=str(path),sha256=file_sha(path),sample_roots=['a']*6+['b']*2)
    network=NS(policy_network=NS(apply=lambda n,p,o:jp.broadcast_to(p,(len(o['state']),4))),parametric_action_distribution=NS(mode=lambda x:jp.tanh(x)))
    return network,(jp.array(2.),jp.zeros(4),jp.array(8.)),manifest
from jit_dvgc.recovery_acquisition_bc import warmup_recovery_actor

def test_online_explicit_weights_do_not_fake_teacher_segments(tmp_path):
    net,init,m=fixture(tmp_path)
    m.update(schema='frontierbridge_online_labels_v1',sample_origins=['student_visited_query']*8,sampling_weights=[.125]*8)
    warmup_recovery_actor(net,init,m,tmp_path/'online',updates=2,batch_size=4)
    import json
    report=json.loads((tmp_path/'online/warmup_status.json').read_text())
    assert report['identity']['phase']=='S_BC'
    assert set(k.split('/')[-1] for k in report['sample_counts'])=={'student_visited_query'}
    m['sampling_weights']=[1.]*8
    with pytest.raises(ValueError,match='sampling weights'):warmup_recovery_actor(net,init,m,tmp_path/'bad',updates=2)
