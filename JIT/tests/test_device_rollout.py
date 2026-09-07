from typing import NamedTuple
import jax
import jax.numpy as jp
import numpy as np
import pytest
from jit_dvgc.continuation.device_rollout import make_device_rollout
from jit_dvgc.acquisition.trajectory_sampling import observed_slice

class Data(NamedTuple):
    qpos: object
    qvel: object
class Up(NamedTuple):
    apex_seen: object
class Down(NamedTuple):
    recovery_success: object
    valid_contact_seen: object
class State(NamedTuple):
    data: object
    obs: object
    info: object
    done: object

def initial(target):
    return State(Data(jp.array([0.]),jp.array([0.])),{'state':jp.array([target])},
        {'up_events':Up(jp.array(False)), 'down_events':Down(jp.array(False),jp.array(False)),
         'phase_transitioned':jp.array(False),'expert_switching_used':jp.array(False)},jp.array(False))
def policy(obs,key):
    return jp.ones(4), {}
def step(s,a):
    x=s.data.qpos+1
    info={**s.info,'down_events':Down(jp.array(False), x[0]>=s.obs['state'][0])}
    return s._replace(data=Data(x,s.data.qvel),info=info)

def test_device_stops_each_lane_and_accounts_inactive_steps():
    states=jax.tree_util.tree_map(lambda *a:jp.stack(a),initial(2.),initial(4.))
    out,counts,bad,flags,cost=make_device_rollout(policy,step,10)(states,jax.random.split(jax.random.PRNGKey(1),2))
    np.testing.assert_array_equal(counts,[2,4])
    np.testing.assert_array_equal(out.data.qpos[:,0],[2,4])
    assert int(cost)==8 and int(cost)-int(sum(counts))==2
    assert not np.any(bad) and np.all(flags[:,3])

def test_horizon_and_nonfinite_are_not_success():
    states=jax.tree_util.tree_map(lambda *a:jp.stack(a),initial(20.),initial(30.))
    _,counts,bad,flags,cost=make_device_rollout(policy,step,3)(states,jax.random.split(jax.random.PRNGKey(1),2))
    assert list(counts)==[3,3] and int(cost)==6 and not np.any(flags[:,3])
    def broken(s,a): return s._replace(data=Data(s.data.qpos*jp.nan,s.data.qvel))
    _,counts,bad,_,_=make_device_rollout(policy,broken,3)(states,jax.random.split(jax.random.PRNGKey(1),2))
    assert np.all(bad) and list(counts)==[1,1]

def test_real_slice_selection_never_fills_skipped_bins():
    seen=set()
    a=observed_slice(2.51,'upstream',seen);seen.add(('upstream',a[0]))
    assert observed_slice(2.52,'upstream',seen) is None
    b=observed_slice(2.64,'upstream',seen)
    assert b==(53,2.6500000000000004)
    assert len(seen)==1
    assert observed_slice(2.51,'downstream',seen) is not None
    assert observed_slice(3.9,'downstream',seen) is None


def test_device_labeler_matches_serial_rows_and_accounts_padding(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from jit_dvgc import unified_continuation_shards as labels
    from jit_dvgc.jump_evidence_validation import write,read
    def start(target):
        s=initial(target)
        info={**s.info,'active_phase':jp.array(0),'success':jp.array(False),'physical_failure':jp.array(False),
              'timeout':jp.array(False),'end_code':jp.array(0)}
        return s._replace(info=info)
    states=[start(2.),start(4.),start(3.)]
    rows=[{'candidate_id':f'c{i}','candidate_kind':'reachable_unified_frontier_probe','state_sha256':str(i)*64,
           'phase':'upstream','phase_index':0,'snapshot':str(i),'source_bank':'bank','parent_group_id':'g',
           'parent_state_sha256':'p'*64} for i in range(3)]
    record={'iteration':0,'name':'pi_0','actor_sha256':'a'*64,'payload_sha256':'b'*64,'formal_config_sha256':'c'*64,'xml_sha256':'xml'}
    path=tmp_path/'catalog.json';write(path,{'candidate_count':3,'protocol_sha256':'d'*64,'entries':rows})
    write(tmp_path/'protocol.json',{'protocol_sha256':'d'*64})
    monkeypatch.setattr(labels,'validate_unified_boundary_catalog',lambda *a,**k:rows)
    monkeypatch.setattr(labels,'validate_candidate_snapshot',lambda *a,**k:None)
    monkeypatch.setattr(labels,'load_unified_envelope_snapshot',lambda p:SimpleNamespace(state=states[int(p.name)],observation=np.array([int(p.name)])))
    monkeypatch.setattr(labels,'fresh_unified_continuation_start',lambda snapshot,env:snapshot.state)
    env=SimpleNamespace(step=step,_bundle=SimpleNamespace(xml_sha256='xml'),resolved_config=SimpleNamespace(ppo=SimpleNamespace(episode_horizon=6)))
    reports=[]
    for backend,size in [('serial',1),('device',2)]:
        reports.append(labels.label_unified_continuation_shard(path,tmp_path/backend,env=env,policy=policy,policy_record=record,
            frozen_manifest_sha256='f'*64,shard_index=0,shard_count=1,max_ticks=6,protocol_seed=9,
            success_criterion='first_valid_landing',execution_backend=backend,batch_size=size))
    assert read(tmp_path/'serial/labels.json')==read(tmp_path/'device/labels.json')
    assert reports[0]['environment_interactions']==9
    assert reports[1]['environment_interactions']==11
    assert reports[1]['inactive_lane_interactions']==2
