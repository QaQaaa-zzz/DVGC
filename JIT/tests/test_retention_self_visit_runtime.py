"""Finite self-visit capture and remaining bridge-window contracts (CPU)."""
from types import SimpleNamespace as NS
import json
import numpy as np
import pytest

from jit_dvgc.generative_bridge import rollout
from jit_dvgc.generative_bridge.teacher_runtime import kernel_identity


@pytest.mark.parametrize('window', [4, 8, 12, 16])
def test_bridge_hands_back_at_remaining_window_and_keeps_source_lane(window):
    import jax
    import jax.numpy as jp
    source=jp.full((17,4),-.5)
    prefix=jp.full((17,16,4),.25)
    source_only=jp.arange(17)==0
    actions=jax.jit(lambda: jax.vmap(lambda t:rollout.prefix_action(
        t,source,prefix,source_only,execution_horizon=window))(jp.arange(18)))()
    np.testing.assert_array_equal(actions[:,0],-.5)
    np.testing.assert_array_equal(actions[:window,1:],.25)
    np.testing.assert_array_equal(actions[window:,1:],-.5)
    if window==16:
        old=jax.vmap(lambda t:rollout.prefix_action(t,source,prefix,source_only))(jp.arange(18))
        np.testing.assert_array_equal(actions,old)


def test_capture_requires_explicit_complete_layout_and_valid_steps():
    spec=dict(horizon=400,order=['pi0'],full_matrix=True,
              preserve_snapshot_episode_context=True,capture_snapshot_steps=[4,8,12],capture_snapshot_lanes=[7])
    assert rollout.validate_snapshot_capture({},17)==([],[])
    assert rollout.validate_snapshot_capture(spec,17)==([4,8,12],[7])
    for change in [dict(capture_snapshot_steps=[0]),dict(capture_snapshot_steps=[8,4]),
                   dict(capture_snapshot_steps=[4,4]),dict(capture_snapshot_steps=[401]),
                   dict(capture_snapshot_lanes=[17]),dict(capture_snapshot_lanes=[]),
                   dict(capture_snapshot_lanes=[True]),dict(full_matrix=False),
                   dict(preserve_snapshot_episode_context=False),dict(reuse_results='old'),
                   dict(evaluation_batch_size=17)]:
        with pytest.raises(ValueError):rollout.validate_snapshot_capture({**spec,**change},17)
    with pytest.raises(ValueError):rollout.validate_snapshot_capture({'capture_snapshot_lanes':[0]},17)


def test_window_and_capture_contracts_separate_compiled_kernels():
    spec=dict(horizon=400)
    policy=dict(actor_sha256='actor',normalizer_sha256='norm')
    key=kernel_identity(spec,policy,17,True,None,True)
    assert key==kernel_identity({**spec,'bridge_execution_horizon':16},policy,17,True,None,True)
    for change in [dict(bridge_execution_horizon=4),dict(capture_snapshot_steps=[4],capture_snapshot_lanes=[7])]:
        assert key!=kernel_identity({**spec,**change},policy,17,True,None,True)


@pytest.mark.parametrize('window', [0, 3, 17, True, '4'])
def test_invalid_remaining_window_rejected_before_runtime(window):
    with pytest.raises(ValueError):
        rollout.validate_evaluation_options(dict(bridge_execution_horizon=window,bridge_action_plan={'path':'x'}),{})


def test_remaining_window_requires_bridge_and_records_handoff():
    with pytest.raises(ValueError):rollout.validate_evaluation_options(dict(bridge_execution_horizon=4),{})
    provenance=rollout.evaluation_controller_provenance(
        dict(bridge_action_plan={'path':'x'},bridge_execution_horizon=4),dict(actor_sha256='a'),{})
    assert provenance['handoff_tick']==4 and provenance['reset_at_handoff'] is False


def capture_fixture():
    from jit_dvgc.exploration_continuation import INFO_FIELDS
    from jit_dvgc.unified_envelope_snapshot import UP_EVENT_FIELDS,DOWN_EVENT_FIELDS
    count=17
    info={k:np.zeros(count,np.int32) for k in INFO_FIELDS}
    info.update(last_action=np.arange(count*4,dtype=np.float32).reshape(count,4),
                rng=np.arange(count*2,dtype=np.uint32).reshape(count,2),
                history=NS(frames=np.arange(count*75,dtype=np.float32).reshape(count,3,25),valid_count=np.full(count,3,np.int32)),
                up_events=NS(**{k:np.full(count,i,np.int32) for i,k in enumerate(UP_EVENT_FIELDS)}),
                down_events=NS(**{k:np.full(count,i,np.int32) for i,k in enumerate(DOWN_EVENT_FIELDS)}),
                episode_step=np.arange(count)+20,phase_episode_step=np.arange(count)+10)
    state=NS(data=NS(qpos=np.arange(count*12,dtype=np.float32).reshape(count,12),
                    qvel=np.arange(count*11,dtype=np.float32).reshape(count,11),ctrl=info['last_action']),
             obs={'state':np.arange(count*76,dtype=np.float32).reshape(count,76)},
             done=np.zeros(count),info=info)
    mapping=NS(ctrl_min=np.zeros(4),ctrl_max=np.ones(4),hip_initial=0.,knee_initial=0.,
               base_rear_speed=0.,rear_speed_delta=1.,joint_target_semantics='absolute',knee_target_delta=1.)
    env=NS(_bundle=NS(xml_sha256='a'*64,action_mapping=mapping),sim_dt=.005,dt=.02)
    record=dict(formal_config_sha256='b'*64,actor_sha256='c'*64,payload_sha256='d'*64,iteration=0,
                inference_override=dict(path='candidate.pkl',sha256='e'*64))
    return state,env,record


def test_capture_exports_exact_poststep_context_and_excludes_terminal_or_unreached(tmp_path):
    from jit_dvgc.pulse_exploration_runtime import snapshot_capture_fields,export_evaluation_snapshots
    from jit_dvgc.unified_envelope_snapshot import load_unified_envelope_snapshot
    state,env,record=capture_fixture()
    fields=snapshot_capture_fields(state,[7])
    assert fields['snap/obs/state'].shape==(1,76)
    tape={k:np.repeat(v[None],12,axis=0) for k,v in fields.items()}
    tape.update(mask=np.ones((12,17),bool),done=np.zeros((12,17),bool),
                physical_failure=np.zeros((12,17),bool),success=np.zeros((12,17),bool),
                timeout=np.zeros((12,17),bool),time_after=np.repeat((.02*np.arange(1,13))[:,None],17,axis=1))
    tape['done'][7,7]=True;tape['mask'][8:,7]=False
    tape['snap/info/episode_step'][3,0]=31
    tape['snap/down/post_contact_ticks'][3,0]=9
    trace=tmp_path/'trace.npz';np.savez_compressed(trace,**tape)
    rows=[dict(root_id='TRAIN-root',root_episode_id='ancestor',candidate_id=i,index=i,
               snapshot_control_step=3,snapshot_context_sha256='f'*64,role='TRAIN',coordinates={'x':123.}) for i in range(17)]
    spec=dict(horizon=400,order=['pi0'],full_matrix=True,preserve_snapshot_episode_context=True,
              capture_snapshot_steps=[4,8,12],capture_snapshot_lanes=[7])
    result=export_evaluation_snapshots(spec,rows,tape,env,record,trace,tmp_path)
    assert [r['capture_status'] for r in result]==['captured','excluded','excluded']
    assert result[1]['reason']=='terminal_state' and result[2]['reason']=='step_not_reached'
    assert result[0]['snapshot_time']==pytest.approx(.08)
    assert result[0]['snapshot_control_step']==7 and result[0]['capture_lane']==7
    assert result[0]['parent_root_id']=='TRAIN-root' and result[0]['root_episode_id']=='ancestor'
    assert 'coordinates' not in result[0] and result[0]['parent_coordinates']=={'x':123.}
    assert result[1]['snapshot'] is None and result[2]['snapshot'] is None
    saved=load_unified_envelope_snapshot(result[0]['snapshot'])
    np.testing.assert_array_equal(saved.observation,state.obs['state'][7])
    np.testing.assert_array_equal(saved.observation_fifo,state.info['history'].frames[7])
    np.testing.assert_array_equal(saved.rng,state.info['rng'][7])
    np.testing.assert_array_equal(saved.last_action,state.info['last_action'][7])
    assert saved.episode_step==31 and int(saved.down_events['post_contact_ticks'])==9
    assert saved.policy_payload_sha256=='e'*64 and saved.policy_actor_sha256=='c'*64
    assert json.loads((tmp_path/'snapshots.json').read_text())==result


def test_capture_rejects_nonfinite_live_state_instead_of_labeling_failure(tmp_path):
    from jit_dvgc.pulse_exploration_runtime import snapshot_capture_fields,export_evaluation_snapshots
    state,env,record=capture_fixture();state.data.qpos[7,0]=np.nan
    tape={k:v[None] for k,v in snapshot_capture_fields(state,[7]).items()}
    tape.update(mask=np.ones((1,17),bool),done=np.zeros((1,17),bool),
                physical_failure=np.zeros((1,17),bool),success=np.zeros((1,17),bool),
                timeout=np.zeros((1,17),bool),time_after=np.ones((1,17)))
    spec=dict(horizon=400,order=['pi0'],full_matrix=True,preserve_snapshot_episode_context=True,
              capture_snapshot_steps=[1],capture_snapshot_lanes=[7])
    with pytest.raises(FloatingPointError):
        export_evaluation_snapshots(spec,[dict(root_id='r',candidate_id=i) for i in range(17)],tape,env,record,tmp_path/'trace.npz',tmp_path)
