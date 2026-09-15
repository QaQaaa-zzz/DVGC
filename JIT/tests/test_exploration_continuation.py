"""Context serialization and bounded existence-label behavior without simulator steps."""
from types import SimpleNamespace as NS
import json

import jax
import numpy as np
import pytest

from jit_dvgc import exploration_continuation as m
from jit_dvgc import unified_envelope_snapshot as snapshots


@pytest.fixture
def captured(monkeypatch):
    monkeypatch.setattr(snapshots, 'compatibility_identity', lambda env: {'fixture': 'complete-context'})
    info = {k: np.asarray(0) for k in m.INFO_FIELDS}
    info.update(rng=jax.random.PRNGKey(19), last_action=np.array([.1, .2, .3, .4]),
        episode_step=np.asarray(17), phase_episode_step=np.asarray(5), episode_return=np.asarray(2.5),
        source_tick=np.asarray(17), active_phase=np.asarray(1), phase_transitioned=np.asarray(True),
        history=NS(frames=np.arange(24).reshape(3, 8), valid_count=np.asarray(3)),
        up_events=NS(**{k: np.asarray(17 if k == 'episode_step' else 0) for k in m.UP_EVENT_FIELDS}),
        down_events=NS(**{k: np.asarray(0) for k in m.DOWN_EVENT_FIELDS}))
    state = NS(data=NS(qpos=np.arange(7.), qvel=np.arange(6.), ctrl=np.arange(4.)),
               done=np.asarray(False), obs={'state': np.arange(24.)}, info=info)
    env = NS(_bundle=NS(xml_sha256='a' * 64))
    record = dict(formal_config_sha256='b' * 64, actor_sha256='c' * 64, payload_sha256='d' * 64, iteration=6)
    kw = dict(env=env, record=record, parent_trajectory='reached-prefix', parent_state_sha256='e' * 64)
    return state, kw


def test_flat_scan_context_matches_canonical_capture_and_is_copied(captured):
    state, kw = captured
    arrays = m.snapshot_arrays(state)
    assert all(isinstance(v, (np.ndarray, jax.Array)) for v in arrays.values())
    result = m.snapshot_from_arrays(arrays, **kw)
    expected = snapshots.capture_unified_envelope_snapshot(state, env=kw['env'],
        parent_trajectory=kw['parent_trajectory'], parent_state_sha256=kw['parent_state_sha256'],
        config_sha256='b' * 64, policy_actor_sha256='c' * 64, policy_payload_sha256='d' * 64, policy_iteration=6)
    assert snapshots.snapshot_context_sha256(result) == snapshots.snapshot_context_sha256(expected)
    state.info['history'].frames[:] = -1
    state.data.qpos[:] = -1
    assert result.observation_fifo[0, 0] == 0 and result.qpos[0] == 0
    assert result.episode_step == 17 and result.phase_episode_step == 5
    assert result.up_events['episode_step'] == 17


@pytest.mark.parametrize('key', ['info/done', 'info/expert_switching_used'])
def test_capture_rejects_terminal_and_switching(captured, key):
    state, kw = captured
    arrays = m.snapshot_arrays(state)
    arrays[key] = np.asarray(True)
    with pytest.raises(ValueError):
        m.snapshot_from_arrays(arrays, **kw)


def evaluator(monkeypatch, tmp_path, budget, outcomes):
    bank = dict(bank_sha256='f' * 64, max_ticks=4,
        task=dict(xml_sha256='a' * 64, success_criterion='first_valid_landing', continuation_start_semantics='fresh_continuation_v1'),
        members=[dict(name=name, roles=['evaluator'], policy=dict(actor_sha256='c'*64, payload_sha256='d'*64))
                 for name in ('first', 'second')])
    from jit_dvgc import probe_bank
    monkeypatch.setattr(probe_bank, 'load_probe_bank', lambda path: bank)
    obj = m.FrozenSuffixEvaluator('bank.json', ['first', 'second'], 4, tmp_path / 'labels', budget)
    calls = []
    def attempt(snapshot, name, directory):
        calls.append(name)
        label, cost = outcomes[name]
        obj.charged_interactions += cost
        if label == 'error':
            raise RuntimeError('dispatched failed step')
        return dict(label=label, outcome='mock_validated_endpoint')
    monkeypatch.setattr(obj, '_attempt', attempt)
    return obj, calls


def test_first_success_stops_and_receipts_bind_snapshot(monkeypatch, tmp_path, captured):
    obj, calls = evaluator(monkeypatch, tmp_path, 8, {'first': (1, 2)})
    snap = m.snapshot_from_arrays(m.snapshot_arrays(captured[0]), **captured[1])
    result = obj.evaluate(snap)
    assert result['label'] == 1 and result['witness'] == 'first'
    assert result['labels'] == {'first': 1, 'second': None} and calls == ['first']
    assert result['charged_interactions'] == 2
    assert result['receipt_sha256'] == m.canonical_sha256({k:v for k,v in result.items() if k != 'receipt_sha256'})
    assert list((tmp_path / 'labels').glob('*/snapshot/snapshot.pkl'))
    with pytest.raises(ValueError, match='already attempted'):
        obj.evaluate(snap)


@pytest.mark.parametrize('budget,outcomes,label,calls_expected,cost', [
    (3, {}, None, [], 0),
    (4, {'first': (0, 3)}, None, ['first'], 3),
    (8, {'first': (0, 3), 'second': (0, 4)}, 0, ['first', 'second'], 7),
    (8, {'first': (None, 1), 'second': (0, 2)}, None, ['first', 'second'], 3),
    (8, {'first': ('error', 1)}, None, ['first'], 1),
])
def test_budget_unknown_and_complete_bank_failure(monkeypatch, tmp_path, captured, budget, outcomes, label, calls_expected, cost):
    obj, calls = evaluator(monkeypatch, tmp_path, budget, outcomes)
    snap = m.snapshot_from_arrays(m.snapshot_arrays(captured[0]), **captured[1])
    result = obj.evaluate(snap)
    assert result['label'] == label and calls == calls_expected
    assert result['charged_interactions'] == cost <= budget
    if calls:
        assert result['attempts'][0]['reserved_interactions'] == 4


@pytest.mark.parametrize('contact,failure,expected', [(True, False, 1), (True, True, None), (False, True, 0)])
def test_actual_suffix_classifies_first_endpoint_and_stops(monkeypatch, tmp_path, captured, contact, failure, expected):
    obj, _ = evaluator(monkeypatch, tmp_path, 8, {})
    from jit_dvgc import unified_continuation_labels as labels
    from jit_dvgc import jump_evidence_runtime as runtime
    initial = {'done': False, 'down/valid_contact_seen': False, 'physical_failure': False,
               'timeout': False, 'end_code': 0, 'success': False}
    final = {**initial, 'done': failure, 'down/valid_contact_seen': contact, 'physical_failure': failure}
    calls = []
    def step(state, action):
        calls.append(action)
        return final
    monkeypatch.setattr(obj, '_runtime', lambda name: (None, None, step))
    monkeypatch.setattr(labels, 'fresh_unified_continuation_start', lambda snapshot, env: initial)
    monkeypatch.setattr(runtime, 'view', lambda state: state.copy())
    monkeypatch.setattr(runtime, 'action_for', lambda *args: np.zeros(4))
    directory = tmp_path / 'attempt'
    directory.mkdir()
    # Bypass only the fake attempt installed by evaluator(), exercising real stop/classify logic.
    result = m.FrozenSuffixEvaluator._attempt(obj, None, 'first', directory)
    assert result['label'] == expected
    assert len(calls) == obj.charged_interactions == 1
    trace = json.loads((directory / 'trace.json').read_text())
    assert len(trace['frames']) == 2 and len(trace['actions']) == 1
