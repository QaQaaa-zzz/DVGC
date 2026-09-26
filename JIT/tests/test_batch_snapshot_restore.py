from types import SimpleNamespace
import jax
import jax.numpy as jp
import numpy as np
import pytest
from mujoco_playground._src import mjx_env
from jit_dvgc import batch_snapshot_restore as batch


def test_restore_validates_every_snapshot_before_dispatch(monkeypatch):
    seen=[]
    def validate(snapshot, env):
        seen.append(snapshot)
        if snapshot == 'bad': raise ValueError('identity mismatch')
    monkeypatch.setattr(batch, 'validate_unified_envelope_snapshot_runtime', validate)
    monkeypatch.setattr(batch, '_fused_restore', lambda env: pytest.fail('dispatched before validation'))
    with pytest.raises(ValueError, match='identity mismatch'):
        batch.restore_snapshot_batch(['good','bad'], object(), backend='fused')
    assert seen == ['good','bad']


def test_payload_preserves_history_rng_events_counters():
    fields=batch.RESTORE_FIELDS
    values={name: np.asarray([i], dtype=np.int32) for i,name in enumerate(fields)}
    values['up_events']={'episode_step':np.asarray(17,np.int32)}
    values['down_events']={'post_contact_ticks':np.asarray(9,np.int32)}
    source=SimpleNamespace(**values, xml_sha256='not traced')
    payload=batch.snapshot_restore_payload(source)
    assert set(payload)==set(fields)
    for name in fields:
        for actual, expected in zip(jax.tree.leaves(payload[name]),jax.tree.leaves(values[name])):
            np.testing.assert_array_equal(actual,expected)


def test_fused_uses_dynamic_payload_and_warp_stack(monkeypatch):
    monkeypatch.setattr(batch,'validate_unified_envelope_snapshot_runtime',lambda s,e:None)
    monkeypatch.setattr(batch,'snapshot_restore_payload',lambda s:{'x':jp.asarray(s)})
    def restore(payload, env):
        x=payload['x']
        return mjx_env.State(data=None,obs={'state':x[None]},reward=x,done=jp.array(0.),
                             metrics={},info={'rng':jax.random.key(4),'counter':x})
    monkeypatch.setattr(batch,'_restore_unified_envelope_payload',restore)
    result=batch.restore_snapshot_batch([2.,7.],SimpleNamespace(),backend='fused',fresh=False)
    np.testing.assert_array_equal(result.info['counter'],[2.,7.])
    assert result.info['rng'].shape==(2,)


def test_rejects_empty_batch_and_unknown_backend():
    with pytest.raises(ValueError,match='nonempty'):
        batch.restore_snapshot_batch([],object())
    with pytest.raises(ValueError,match='backend'):
        batch.restore_snapshot_batch([None],object(),backend='magic')


def test_parity_rejects_counter_and_physics_drift():
    from jit_dvgc.batch_snapshot_restore_validation import compare_trees
    reference={'qpos':np.asarray([1.],np.float32),'counter':np.asarray(1,np.int32)}
    assert compare_trees(reference,reference)['exact']
    assert not compare_trees(reference,{**reference,'counter':np.asarray(2,np.int32)})['passed']
    assert not compare_trees(reference,{**reference,'qpos':np.asarray([1.01],np.float32)})['passed']
