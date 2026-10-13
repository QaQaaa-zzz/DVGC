"""Real-source conversion contracts; fixtures never step a physical environment."""
import numpy as np
import pytest
from jit_dvgc.retention_consolidation_data import _real_lane,collection_condition_identity
from .test_generative_bridge_student_v12 import teacher


def raw_lane():
    trace=teacher(n=18);raw={k:np.asarray(v)[:,None] for k,v in trace['arrays'].items()}
    raw['mask']=raw.pop('valid_mask');raw['physical_failure']=raw.pop('failure')
    raw['action_origin_code']=np.where(np.arange(18)<16,1,2)[:,None]
    return raw


def test_canonical_lane_keeps_real_observation_and_terminal_alignment():
    raw=raw_lane();arrays=_real_lane(raw,0,steps=18)
    np.testing.assert_array_equal(arrays['actor_observation_before'],raw['actor_observation_before'][:,0])
    assert arrays['success'][-1] and arrays['valid_mask'].all()


@pytest.mark.parametrize('change',['padding_gap','missing_after','wrong_length','missing_timeout'])
def test_source_lane_never_synthesizes_missing_history_or_timing(change):
    raw=raw_lane();steps=18
    if change=='padding_gap':raw['mask'][0,0]=False
    elif change=='missing_after':raw.pop('actor_observation_after')
    elif change=='missing_timeout':raw.pop('timeout')
    else:steps=19
    with pytest.raises((ValueError,KeyError)):_real_lane(raw,0,steps=steps)


def test_condition_identity_uses_physical_requests_not_episode_names():
    q=np.zeros(12);v=np.zeros(11);requests=np.zeros((400,4))
    one=collection_condition_identity(q,v,requests,protocol_sha256='a'*64)
    assert one==collection_condition_identity(q.copy(),v.copy(),requests.copy(),protocol_sha256='a'*64)
    changed=requests.copy();changed[2,1]=.1
    assert one!=collection_condition_identity(q,v,changed,protocol_sha256='a'*64)
    q[0]=np.nan
    with pytest.raises(ValueError):collection_condition_identity(q,v,requests,protocol_sha256='a'*64)
