import numpy as np
import pytest

def test_mixed_onsets_balance_rotate_and_reserve_longest_prefix():
    from jit_dvgc.pulse_schedule import lane_onsets
    from jit_dvgc.pulse_exploration import prefix_budget
    spec=dict(num_envs=1024,pulse_steps=3,horizon=400,pulse_start_schedule=[5,10,15,20,25,0],pulse_batch_mode='mixed')
    a=lane_onsets(spec,0);b=lane_onsets(spec,1)
    assert a.shape==(1024,) and a.dtype==np.int32
    assert set(a)==set(spec['pulse_start_schedule'])
    counts=np.array([(a==t).sum() for t in spec['pulse_start_schedule']])
    assert counts.max()-counts.min()==1
    assert not np.array_equal(a,b)
    assert prefix_budget(spec,0)==1024*28

def test_legacy_and_nominal_onsets_keep_contract():
    from jit_dvgc.pulse_schedule import lane_onsets
    np.testing.assert_array_equal(lane_onsets(dict(num_envs=4,pulse_start_schedule=[5,10],pulse_steps=3,horizon=400),1),[10]*4)
    spec=dict(num_envs=1,pulse_start_schedule=[0],pulse_steps=400,horizon=400,nominal_source_rollout=True,pulse_batch_mode='mixed',delta_limit=[0.,0.,0.,0.])
    np.testing.assert_array_equal(lane_onsets(spec,0),[0])
    with pytest.raises(ValueError):lane_onsets(dict(num_envs=4,pulse_steps=3,horizon=400,pulse_start_schedule=[399],pulse_batch_mode='mixed'),0)

def test_mixed_lanes_stop_at_own_third_action_and_keep_endpoint():
    from jit_dvgc.pulse_schedule import pulse_activity
    import jax.numpy as jp
    delays=np.array([0,5,10]);alive=jp.ones(3,bool);trigger=jp.full(3,-1);count=jp.zeros(3,int)
    last=np.full(3,-1)
    for tick in range(13):
        trigger,mask=pulse_activity(trigger,count,tick>=delays,alive,tick,3)
        last[np.asarray(alive)]=tick
        count=count+mask
        alive=alive&(count<3)
    np.testing.assert_array_equal(count,[3,3,3]);np.testing.assert_array_equal(last,[2,7,12])

def test_mixed_and_event_modes_are_rejected_at_budget_boundary():
    from jit_dvgc.pulse_exploration import prefix_budget
    spec=dict(num_envs=6,pulse_steps=3,horizon=400,pulse_batch_mode='mixed',pulse_event_schedule=['apex'])
    with pytest.raises(ValueError,match='mixed'):prefix_budget(spec,0)

def test_resume_reconstructs_only_completed_nominal_seed_evidence(tmp_path):
    import json
    from jit_dvgc.pulse_exploration import neighborhood_seed_history
    for rel,value in [('seed_support/evaluation/results.json',[{'index':0}]),('round_0000/next_source_seed/evaluation/results.json',[{'index':1}])]:
        p=tmp_path/rel;p.parent.mkdir(parents=True);p.write_text(json.dumps(value))
    assert neighborhood_seed_history(tmp_path)==[{'index':0},{'index':1}]
