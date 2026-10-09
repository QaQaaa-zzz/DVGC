"""Exact sampling equivalence against the frozen pre-index implementation."""
import numpy as np
import pytest
from jit_dvgc.generative_bridge.data import build_action_windows
from jit_dvgc.generative_bridge.feedback_data import MIX, realized_mix, sample_corpus
from jit_dvgc.generative_bridge.corpus_index import compile_corpus_index


def synthetic_corpus(ancestors=5, length=41):
    groups = {key: [] for key in MIX}
    for group_index, (group, traces) in enumerate(groups.items()):
        for ancestor in reversed(range(ancestors)):
            for trajectory in range(1 + ancestor % 3):
                n = length + trajectory * 7
                obs = np.arange((n+1)*76, dtype=np.float64).reshape(n+1,76)
                obs += ancestor * 100000 + group_index * 10000000
                traces.append({'metadata': {'root_episode_id': str(ancestor),
                    'root_id': f'{group}-{ancestor}-{trajectory}', 'onset': trajectory,
                    'trace_start_step': 8 if trajectory else None}, 'arrays': {
                    'actor_observation_before': obs[:-1], 'actor_observation_after': obs[1:],
                    'normalized_action_executed': np.sin(obs[:-1,:4]),
                    'valid_mask': np.ones(n,bool), 'done': np.arange(n)==n-1,
                    'success': np.arange(n)==n-1, 'failure': np.zeros(n,bool),
                    'timeout': np.zeros(n,bool), 'phase_before': np.arange(n)%2,
                    'action_origin': np.array(['bridge_prefix']*8+['source_tail']*(n-8))}})
    return {'groups': groups, 'realized_mix': realized_mix({k:len(v) for k,v in groups.items()})}


def assert_same(a, b):
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert a[0].dtype == b[0].dtype and a[1].dtype == b[1].dtype
    assert a[2:] == b[2:]


@pytest.mark.parametrize('missing', [None, 'history', 'teacher_new', 'actor_new'])
def test_fixed_seed_exact_sequence_and_rng_state(missing):
    corpus = synthetic_corpus()
    if missing:
        corpus['groups'][missing] = []
        corpus['realized_mix'] = realized_mix({k:len(v) for k,v in corpus['groups'].items()})
    compiled = compile_corpus_index(corpus)
    old_rng = np.random.default_rng(8309)
    new_rng = np.random.default_rng(8309)
    for count in [0, 1, 17, 256]:
        assert_same(legacy_sample_corpus(corpus,old_rng,count,return_metadata=True),
                    sample_corpus(compiled,new_rng,count,return_metadata=True))
        assert old_rng.bit_generator.state == new_rng.bit_generator.state


def test_explicit_snapshot_isolated_from_writable_corpus_and_no_implicit_cache():
    corpus = synthetic_corpus(ancestors=1)
    compiled = compile_corpus_index(corpus)
    expected = sample_corpus(compiled,np.random.default_rng(8),24,return_metadata=True)
    for traces in corpus['groups'].values():
        for t in traces:
            t['arrays']['normalized_action_executed'][:] = .75
            t['metadata']['root_id'] = 'changed'
    assert_same(expected,sample_corpus(compiled,np.random.default_rng(8),24,return_metadata=True))
    changed = sample_corpus(corpus,np.random.default_rng(8),24,return_metadata=True)
    assert not np.array_equal(expected[1],changed[1])
    assert all(m['root_id']=='changed' for m in changed[3])
    row = compiled.groups[0][0][1][0]
    with pytest.raises(ValueError):
        row.actions.setflags(write=True)


def test_invalid_trace_rejected_at_compile_and_not_rechecked_per_draw(monkeypatch):
    corpus = synthetic_corpus(ancestors=1)
    compiled = compile_corpus_index(corpus)
    import jit_dvgc.generative_bridge.corpus_index as module
    monkeypatch.setattr(module,'validate_trace',lambda _: (_ for _ in ()).throw(AssertionError('revalidated')))
    sample_corpus(compiled,np.random.default_rng(1),40)
    monkeypatch.undo()
    corpus['groups']['history'][0]['arrays']['normalized_action_executed'][0,0] = np.nan
    with pytest.raises(ValueError,match='normalized_action_executed'):
        compile_corpus_index(corpus)



def test_one_window_no_metadata_and_empty_corpus():
    corpus = synthetic_corpus(ancestors=1, length=16)
    for payload in (corpus, compile_corpus_index(corpus)):
        assert_same(legacy_sample_corpus(corpus,np.random.default_rng(92),30),
                    sample_corpus(payload,np.random.default_rng(92),30))
    for group in corpus['groups']:
        corpus['groups'][group] = []
    corpus['realized_mix'] = dict.fromkeys(MIX,0.)
    with pytest.raises(ValueError, match='empty corpus'):
        sample_corpus(compile_corpus_index(corpus),np.random.default_rng(92),1)


def legacy_sample_corpus(corpus, rng, batch_size, *, return_metadata=False):
    """Choose source, ancestor, trajectory, then window; loss is not reweighted."""
    names=list(MIX);probs=[corpus['realized_mix'][k] for k in names]
    if not any(probs):raise ValueError('empty corpus cannot be sampled')
    observations=[];actions=[];sources=[];metadata=[]
    for _ in range(batch_size):
        group=names[int(rng.choice(3,p=probs))];traces=corpus['groups'][group]
        ancestors=sorted({t['metadata']['root_episode_id'] for t in traces})
        ancestor=ancestors[int(rng.integers(len(ancestors)))]
        choices=[t for t in traces if t['metadata']['root_episode_id']==ancestor]
        trace=choices[int(rng.integers(len(choices)))]
        windows=build_action_windows(trace)
        i=int(rng.integers(len(windows['actions'])))
        observations.append(windows['observations'][i]);actions.append(windows['actions'][i]);sources.append(group)
        if return_metadata:
            start=int(windows['start_indices'][i]);m=trace['metadata']
            origins=sorted(set(trace['arrays']['action_origin'][start:start+16].tolist()))
            offset=m.get('trace_start_step')
            metadata.append({'source_group':group,'recency':'history' if group=='history' else 'new',
                'root_episode_id':ancestor,'root_id':m.get('root_id'),'onset':m.get('onset'),
                'window_start':start,'window_end_exclusive':start+16,
                'window_start_step':None if offset is None else int(offset)+start,
                'window_end_step_exclusive':None if offset is None else int(offset)+start+16,
                'segment':origins[0] if len(origins)==1 else 'mixed','action_origins':origins})
    result=(np.asarray(observations),np.asarray(actions),sources)
    return (*result,metadata) if return_metadata else result

