from types import SimpleNamespace as NS
import importlib
import numpy as np
import pytest


def test_recording_real_preobs_and_default_disabled():
    m=importlib.import_module('jit_dvgc.generative_bridge.rollout')
    old=NS(obs={'state':np.ones((2,76))},data=NS(ctrl=np.zeros((2,4))))
    new=NS(obs={'state':np.ones((2,76))*2},data=NS(ctrl=np.ones((2,4))*3))
    action=np.ones((2,4))*.4
    fields=m.observation_fields(old,new,action,enabled=False)
    assert fields=={}
    fields=m.observation_fields(old,new,action,enabled=True)
    assert np.array_equal(fields['actor_observation_before'],old.obs['state'])
    assert np.array_equal(fields['actor_observation_after'],new.obs['state'])
    assert np.array_equal(fields['actuator_targets_after_mapping'],new.data.ctrl)
    assert np.array_equal(fields['normalized_action_executed'],action)


def test_prefix_switch_exact_boundary_and_source_only():
    import jax.numpy as jp
    m=importlib.import_module('jit_dvgc.generative_bridge.rollout')
    prefixes=jp.ones((2,16,4))*.8;source=jp.zeros((2,4))
    for tick in [0,15]:
        a=m.prefix_action(tick,source,prefixes,jp.array([False,True]))
        np.testing.assert_array_equal(a[0],prefixes[0,0]);np.testing.assert_array_equal(a[1],source[1])
    np.testing.assert_array_equal(m.prefix_action(16,source,prefixes,jp.array([False,False])),source)


def test_proposal_pool_seed_identity_and_early_baseline_fill():
    m=importlib.import_module('jit_dvgc.generative_bridge.proposals')
    generated=np.zeros((16,16,4));prefix=np.ones((5,4))*.2
    a=m.make_candidate_pool('root',prefix,generated,seed=21)
    b=m.make_candidate_pool('root',prefix,generated,seed=21)
    np.testing.assert_array_equal(a['actions'],b['actions'])
    assert a['actions'].shape==(32,16,4)
    assert a['proposal_reference_padded'] is True
    assert a['kinds'].count('source_only')==1
    assert not np.array_equal(a['actions'][17],a['actions'][18])


def test_existing_evaluation_export_needs_raw_preobs_and_matching_context(tmp_path):
    from .test_generative_bridge_contracts import trace
    from jit_dvgc.generative_bridge.contracts import file_sha
    m=importlib.import_module('jit_dvgc.generative_bridge.feedback_data')
    t=trace();a=t['arrays'];path=tmp_path/'tape.npz'
    np.savez(path,**{k:np.asarray(v)[:,None] for k,v in a.items()})
    attempt={'trace':str(path),'trace_sha256':file_sha(path),'trace_lane':0,
             'actor_sha256':'actor','normalizer_sha256':'norm','snapshot_context_sha256':'context','label':1,
             'model_sha256':'model','recording_schema':'jit_actor_success_trace_v1_1',
             'action_origin':'actor_only','outcome':'stable_forward_recovery'}
    result=m.trace_from_evaluation(attempt,t['metadata'])
    assert result['arrays']['actor_observation_before'].shape==(18,76)
    attempt['snapshot_context_sha256']='other'
    with pytest.raises(ValueError):m.trace_from_evaluation(attempt,t['metadata'])


def test_evaluation_provenance_cannot_be_replaced_by_metadata(tmp_path):
    from .test_generative_bridge_contracts import trace
    from jit_dvgc.generative_bridge.contracts import file_sha
    m=importlib.import_module('jit_dvgc.generative_bridge.feedback_data')
    t=trace();path=tmp_path/'tape.npz'
    np.savez(path,**{k:np.asarray(v)[:,None] for k,v in t['arrays'].items()})
    attempt={'trace':str(path),'trace_sha256':file_sha(path),'trace_lane':0,'actor_sha256':'actor',
             'normalizer_sha256':'norm','snapshot_context_sha256':'context','label':1,
             'model_sha256':'model','recording_schema':'jit_actor_success_trace_v1_1',
             'action_origin':'actor_only','outcome':'stable_forward_recovery'}
    for change in [{'model_sha256':'wrong'},{'action_origin':'bridge_prefix'},
                   {'outcome':'first_valid_landing'},{'recording_schema':None}]:
        with pytest.raises(ValueError):m.trace_from_evaluation({**attempt,**change},t['metadata'])
