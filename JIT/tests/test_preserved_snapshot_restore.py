"""Exact duplicate teacher roots may share restoration, never lane identity."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace as NS

import flax.struct
import jax
import jax.numpy as jp
import numpy as np
import pytest
from mujoco_playground._src import mjx_env

from jit_dvgc import pulse_exploration_runtime as restore
from jit_dvgc import unified_envelope_snapshot as snapshots
from jit_dvgc.unified_envelope_snapshot import (UnifiedEnvelopeSnapshot,UP_EVENT_FIELDS,
    DOWN_EVENT_FIELDS,save_unified_envelope_snapshot,snapshot_context_sha256)
from jit_dvgc.handoff_snapshot import compatibility_identity


@flax.struct.dataclass
class Data:
    qpos: object
    qvel: object
    ctrl: object
    time: object


def fixture(tmp_path,monkeypatch):
    mapping=NS(ctrl_min=np.zeros(4),ctrl_max=np.ones(4),hip_initial=0.,knee_initial=0.,
        base_rear_speed=0.,rear_speed_delta=1.,joint_target_semantics='absolute',knee_target_delta=1.)
    env=NS(_bundle=NS(xml_sha256='a'*64,action_mapping=mapping),sim_dt=.005,dt=.02,marker=1)
    snapshot=UnifiedEnvelopeSnapshot(qpos=np.array([1.,2.]),qvel=np.array([3.,4.]),ctrl=np.zeros(4),
        observation_fifo=np.arange(75).reshape(3,25),history_valid_count=3,
        observation=np.arange(76),last_action=np.arange(4),rng=np.array([0,9],np.uint32),
        up_events={k:np.array(3) for k in UP_EVENT_FIELDS},down_events={k:np.array(2) for k in DOWN_EVENT_FIELDS},
        active_phase=1,start_phase=0,phase_transitioned=True,episode_step=18,phase_episode_step=7,
        episode_return=1.,reset_from_soft_tube=False,source_tick=3,parent_group_index=4,
        tube_entry_index=5,tube_global_index=6,parent_trajectory='prefix',parent_state_sha256='b'*64,
        config_sha256='c'*64,xml_sha256='a'*64,policy_actor_sha256='d'*64,policy_payload_sha256='e'*64,
        policy_iteration=1,compatibility_identity=compatibility_identity(env))
    for name in ('first','second'):save_unified_envelope_snapshot(tmp_path/name,snapshot)
    calls=[]
    def numerical(snapshot,actual_env):
        calls.append(snapshot)
        return mjx_env.State(data=Data(jp.asarray(snapshot.qpos),jp.asarray(snapshot.qvel),
            jp.asarray(snapshot.ctrl),jp.asarray(0.)),obs={'state':jp.asarray(snapshot.observation)},
            reward=jp.asarray(0.),done=jp.asarray(0.),metrics={},info=dict(
                rng=jax.random.wrap_key_data(snapshot.rng),history=jp.asarray(snapshot.observation_fifo),
                episode_step=jp.asarray(snapshot.episode_step),phase_episode_step=jp.asarray(snapshot.phase_episode_step),
                last_action=jp.asarray(snapshot.last_action),env_marker=jp.asarray(actual_env.marker)))
    monkeypatch.setattr(snapshots,'restore_unified_envelope_snapshot',numerical)
    rows=[dict(snapshot=str(tmp_path/'first'),snapshot_context_sha256=snapshot_context_sha256(snapshot),
        snapshot_time=.36,candidate_id=i,label=i%2) for i in range(17)]
    return env,rows,calls


def test_seventeen_identical_rows_restore_once_and_keep_order_labels_rng_and_clocks(tmp_path,monkeypatch):
    from jit_dvgc.continuation.device_rollout import stack_worlds
    env,rows,calls=fixture(tmp_path,monkeypatch);before=deepcopy(rows)
    states,receipt=restore.restore_preserved_snapshot_rows(rows,env)
    assert len(states)==17 and len(calls)==1
    assert receipt['row_to_restore']==[0]*17 and receipt['restores']==1 and receipt['reused_rows']==16
    assert rows==before
    batch=stack_worlds(states)
    np.testing.assert_array_equal(batch.info['episode_step'],[18]*17)
    np.testing.assert_array_equal(batch.info['phase_episode_step'],[7]*17)
    np.testing.assert_allclose(batch.data.time,[.36]*17)
    np.testing.assert_array_equal(jax.random.key_data(batch.info['rng']),[[0,9]]*17)


def test_different_time_or_path_has_distinct_restore_and_preserves_original_order(tmp_path,monkeypatch):
    env,rows,calls=fixture(tmp_path,monkeypatch)
    rows[2]['snapshot_time']=.38;rows[4]['snapshot']=str(tmp_path/'second')
    states,receipt=restore.restore_preserved_snapshot_rows(rows,env)
    assert len(calls)==3 and receipt['restores']==3
    assert receipt['row_to_restore']==[0,0,1,0,2]+[0]*12
    assert float(states[2].data.time)==pytest.approx(.38)
    assert float(states[4].data.time)==pytest.approx(.36)


def test_matched_reference_mode_restores_all_rows_with_equal_state_trees(tmp_path,monkeypatch):
    env,rows,calls=fixture(tmp_path,monkeypatch)
    reference,old=restore.restore_preserved_snapshot_rows(rows,env,deduplicate=False)
    accelerated,new=restore.restore_preserved_snapshot_rows(rows,env)
    assert len(calls)==18 and old['restores']==17 and new['restores']==1
    for a,b in zip(jax.tree.leaves(reference),jax.tree.leaves(accelerated)):
        if str(getattr(a,'dtype','')).startswith('key<'):
            a,b=jax.random.key_data(a),jax.random.key_data(b)
        np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('change',['context','time_missing','time_nonfinite','time_negative','runtime'])
def test_every_row_is_checked_before_any_restore(tmp_path,monkeypatch,change):
    env,rows,calls=fixture(tmp_path,monkeypatch)
    if change=='context':rows[-1]['snapshot_context_sha256']='wrong'
    elif change=='time_missing':del rows[-1]['snapshot_time']
    elif change=='time_nonfinite':rows[-1]['snapshot_time']=float('nan')
    elif change=='time_negative':rows[-1]['snapshot_time']=-.1
    else:env._bundle.xml_sha256='f'*64
    with pytest.raises(ValueError):restore.restore_preserved_snapshot_rows(rows,env)
    assert calls==[]


def test_changed_snapshot_payload_is_rejected_before_dispatch(tmp_path,monkeypatch):
    env,rows,calls=fixture(tmp_path,monkeypatch)
    payload=tmp_path/'first/snapshot.pkl';payload.write_bytes(payload.read_bytes()+b'corrupt')
    with pytest.raises(ValueError,match='payload SHA'):restore.restore_preserved_snapshot_rows(rows,env)
    assert calls==[]


def test_no_cross_call_state_cache_even_when_model_contract_is_identical(tmp_path,monkeypatch):
    env,rows,calls=fixture(tmp_path,monkeypatch)
    first,_=restore.restore_preserved_snapshot_rows(rows,env)
    env.marker=2
    second,_=restore.restore_preserved_snapshot_rows(rows,env)
    assert len(calls)==2
    assert int(first[0].info['env_marker'])==1 and int(second[0].info['env_marker'])==2


def test_distinct_payload_and_context_keep_distinct_physical_states(tmp_path,monkeypatch):
    env,rows,calls=fixture(tmp_path,monkeypatch)
    original=snapshots.load_unified_envelope_snapshot(tmp_path/'second')
    changed=replace(original,qpos=np.array([7.,8.]),episode_step=21)
    save_unified_envelope_snapshot(tmp_path/'changed',changed)
    rows[8].update(snapshot=str(tmp_path/'changed'),snapshot_context_sha256=snapshot_context_sha256(changed))
    states,receipt=restore.restore_preserved_snapshot_rows(rows,env)
    assert len(calls)==2 and receipt['row_to_restore'][8]==1
    np.testing.assert_array_equal(states[0].data.qpos,[1.,2.])
    np.testing.assert_array_equal(states[8].data.qpos,[7.,8.])
    assert int(states[8].info['episode_step'])==21


def test_every_duplicate_row_checks_runtime_and_bad_restored_observation_fails(tmp_path,monkeypatch):
    env,rows,calls=fixture(tmp_path,monkeypatch)
    canonical=snapshots.validate_unified_envelope_snapshot_runtime
    checks=[]
    def validate(snapshot,actual_env):
        checks.append(snapshot)
        canonical(snapshot,actual_env)
    monkeypatch.setattr(snapshots,'validate_unified_envelope_snapshot_runtime',validate)
    numerical=snapshots.restore_unified_envelope_snapshot
    def wrong(snapshot,actual_env):
        return numerical(snapshot,actual_env).replace(obs={'state':jp.zeros_like(snapshot.observation)})
    monkeypatch.setattr(snapshots,'restore_unified_envelope_snapshot',wrong)
    with pytest.raises(ValueError,match='Actor observation/history'):
        restore.restore_preserved_snapshot_rows(rows,env)
    assert len(checks)==17 and len(calls)==1
