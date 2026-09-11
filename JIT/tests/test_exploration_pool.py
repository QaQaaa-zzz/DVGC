"""Offline integrity and append-only history tests; no simulator is constructed."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from flax import serialization

from types import SimpleNamespace as NS
import jax
from jit_dvgc import exploration_pool as m
from jit_dvgc import exploration_continuation as continuation
from jit_dvgc import unified_envelope_snapshot as snapshots
from jit_dvgc.handoff_bank import pytree_sha256


@pytest.fixture
def captured(monkeypatch):
    monkeypatch.setattr(snapshots, 'compatibility_identity', lambda env: {'fixture': 'complete-context'})
    info = {k: np.asarray(0) for k in continuation.INFO_FIELDS}
    info.update(rng=jax.random.PRNGKey(19), last_action=np.array([.1, .2, .3, .4]),
        episode_step=np.asarray(17), phase_episode_step=np.asarray(5), episode_return=np.asarray(2.5),
        source_tick=np.asarray(17), active_phase=np.asarray(1), phase_transitioned=np.asarray(True),
        history=NS(frames=np.arange(24).reshape(3, 8), valid_count=np.asarray(3)),
        up_events=NS(**{k: np.asarray(17 if k == 'episode_step' else 0) for k in continuation.UP_EVENT_FIELDS}),
        down_events=NS(**{k: np.asarray(0) for k in continuation.DOWN_EVENT_FIELDS}))
    state = NS(data=NS(qpos=np.arange(7.), qvel=np.arange(6.), ctrl=np.arange(4.)),
               done=np.asarray(False), obs={'state': np.arange(24.)}, info=info)
    env = NS(_bundle=NS(xml_sha256='a' * 64))
    record = dict(formal_config_sha256='b' * 64, actor_sha256='c' * 64, payload_sha256='d' * 64, iteration=6)
    kw = dict(env=env, record=record, parent_trajectory='reached-prefix', parent_state_sha256='e' * 64)
    return state, kw


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def seal(doc, field='receipt_sha256'):
    doc.pop(field, None)
    doc[field] = m.canonical_sha256(doc)
    return doc


@pytest.fixture
def source(tmp_path, captured):
    state, kw = captured
    actor = {'params': {'kernel': np.ones((2, 4), dtype=np.float32)}}
    checkpoint = tmp_path / 'checkpoints/initial/state.msgpack'
    checkpoint.parent.mkdir(parents=True)
    from brax.training.acme.running_statistics import RunningStatisticsState
    normalizer = RunningStatisticsState(mean=np.zeros(2), std=np.ones(2), count=np.asarray(0.), summed_variance=np.zeros(2))
    checkpoint.write_bytes(serialization.to_bytes(dict(params=dict(policy=actor), normalizer=normalizer)))
    controller = dict(base_actor_sha256='c'*64, residual_actor_sha256=pytree_sha256(actor),
        normalizer_sha256=pytree_sha256(normalizer), residual_checkpoint=str(checkpoint), residual_checkpoint_sha256=m._sha(checkpoint),
        action_composition='clip(base+delta_limit*tanh_sample,-1,1)', delta_limit=[.1]*4)
    seal(controller, 'controller_sha256')
    write(checkpoint.with_name('identity.json'), dict(base_actor_sha256='c'*64, state_sha256=m._sha(checkpoint), delta_limit=[.1]*4))
    kw['record'].update(actor_sha256=controller['controller_sha256'], payload_sha256=m._sha(checkpoint))
    candidates=[]
    for e in range(2):
        state.info['rng'] = np.array([0, e], dtype=np.uint32)
        arrays = continuation.snapshot_arrays(state)
        snap = continuation.snapshot_from_arrays(arrays, **kw)
        context = snapshots.snapshot_context_sha256(snap)
        directory = tmp_path / 'suffixes' / context / 'snapshot'
        snapshots.save_unified_envelope_snapshot(directory, snap)
        prefix = tmp_path / f'prefix{e}.npz'
        np.savez(prefix, **{'snap/' + k: np.asarray(v)[None,None] for k,v in arrays.items()})
        candidate = dict(episode_index=0, tick=0, cell='same-cell', state_sha256=snapshots.physical_state_sha256(snap),
            context_sha256=context, label=None, witness=None, suffix_receipt_sha256=None, prefix_file=str(prefix),
            prefix_file_sha256=m._sha(prefix), controller=controller, generated_by_env_step_only=True,
            jump_start_state_sha256=kw['parent_state_sha256'])
        candidates.append(seal(candidate))
    write(tmp_path / 'declaration.json', dict(role='train',final_test_used=False,spec=dict(proposer='pi_fixture')))
    write(tmp_path / 'initial_candidates.json', candidates)
    return tmp_path, candidates


def result(entry, label, bank='a'):
    labels={'evaluator': label}
    attempts=[]
    if label is not None:
        attempts=[seal(dict(evaluator='evaluator', snapshot_context_sha256=entry['context_sha256'],
            label=label, status='completed', outcome='first_valid_landing' if label else 'horizon_exhausted_before_landing',
            end_flags={'physical_failure': False, 'down/valid_contact_seen': bool(label)}))]
    return seal(dict(state_sha256=entry['state_sha256'], snapshot_context_sha256=entry['context_sha256'],
        snapshot_identity_sha256=m._sha(Path(entry['snapshot_dir'])/'identity.json'), bank_sha256=bank*64,
        label=label, labels=labels, witness='evaluator' if label else None, attempts=attempts))


def test_distinct_contexts_same_cell_and_append_only_later_promotion(source):
    root,_=source
    pool=m.import_candidates(root, round_index=2)
    assert len(pool['entries'])==2
    assert len({e['state_sha256'] for e in pool['entries'].values()})==1
    key,entry=next(iter(pool['entries'].items()))
    frozen=copy.deepcopy(pool)
    unknown=m.add_evaluation(pool,key,result(entry,None),round_index=2)
    negative=m.add_evaluation(unknown,key,result(entry,0),round_index=2)
    assert negative['entries'][key]['status']=='pending'
    positive=m.add_evaluation(negative,key,result(entry,1,'b'),round_index=3)
    assert positive['entries'][key]['status']=='witnessed'
    assert [o['label'] for o in positive['entries'][key]['observations']]==[None,0,1]
    assert pool==frozen
    assert negative['entries'][key]['observations']==positive['entries'][key]['observations'][:2]
    assert m.import_candidates(root,positive,round_index=4)==positive
    assert len(m.select_pending(positive,5))==1


@pytest.mark.parametrize('what',['candidate','prefix','checkpoint','snapshot','actor'])
def test_import_rejects_tampering(source,what):
    root,candidates=source
    c=candidates[0]
    if what=='candidate':
        c['cell']='altered'
    elif what=='prefix':
        Path(c['prefix_file']).write_bytes(b'corrupted')
    elif what=='checkpoint':
        Path(c['controller']['residual_checkpoint']).write_bytes(b'corrupted')
    elif what=='snapshot':
        p=root/'suffixes'/c['context_sha256']/'snapshot/snapshot.pkl'
        p.write_bytes(b'corrupted')
    else:
        c['controller']['residual_actor_sha256']='0'*64
        seal(c['controller'],'controller_sha256');seal(c)
    write(root/'initial_candidates.json',candidates)
    with pytest.raises(ValueError):m.import_candidates(root)


def test_selfhashed_prefix_context_swap_rejected(source):
    root,candidates=source
    candidates[0]['prefix_file']=candidates[1]['prefix_file']
    candidates[0]['prefix_file_sha256']=candidates[1]['prefix_file_sha256']
    seal(candidates[0]);write(root/'initial_candidates.json',candidates)
    with pytest.raises(ValueError,match='prefix snapshot mismatch'):m.import_candidates(root)


def test_selfhashed_wrong_context_and_conflicted_positive_rejected(source):
    pool=m.import_candidates(source[0]);key,entry=next(iter(pool['entries'].items()))
    r=result(entry,1);r['snapshot_context_sha256']='0'*64;seal(r)
    with pytest.raises(ValueError,match='state/context'):m.add_evaluation(pool,key,r,round_index=0)
    r=result(entry,1);r['attempts'][0]['end_flags']['physical_failure']=True
    seal(r['attempts'][0]);seal(r)
    with pytest.raises(ValueError,match='unconflicted'):m.add_evaluation(pool,key,r,round_index=0)


def test_select_phase_balanced_deterministic_and_detached(source):
    pool=m.import_candidates(source[0]);values=list(pool['entries'].values())
    values[0]['phase']='upstream';values[1]['phase']='downstream';m._seal(pool)
    selected=m.select_pending(pool,2)
    assert [e['phase'] for e in selected]==['downstream','upstream']
    selected[0]['status']='tampered'
    assert all(e['status']=='pending' for e in pool['entries'].values())
    pool['entries'][selected[0]['key']]['status']='witnessed'
    with pytest.raises(ValueError,match='self-hash'):m.select_pending(pool,2)


@pytest.mark.parametrize('label',[None,0,1])
def test_import_retains_each_bank_outcome_and_verifies_saved_receipts(source,label):
    root,candidates=source
    pool=m.import_candidates(root)
    entry=next(e for e in pool['entries'].values() if e['context_sha256']==candidates[0]['context_sha256'])
    r=result(entry,label)
    directory=root/'suffixes'/entry['context_sha256']
    for attempt in r['attempts']:
        write(directory/attempt['evaluator']/'receipt.json',attempt)
    write(directory/'result.json',r)
    candidates[0].update(label=label,witness=r['witness'],suffix_receipt_sha256=r['receipt_sha256'])
    seal(candidates[0]);write(root/'initial_candidates.json',candidates)
    imported=m.import_candidates(root)
    observed=imported['entries'][entry['key']]
    assert observed['status']==('witnessed' if label==1 else 'pending')
    assert observed['observations'][0]['label']==label
    if label is not None:
        write(directory/'evaluator/receipt.json',{})
        with pytest.raises(ValueError,match='saved evaluator receipt'):
            m.import_candidates(root)
