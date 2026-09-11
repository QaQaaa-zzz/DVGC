"""Arrival novelty is provisional; selectors operate only on existing frames."""
from copy import deepcopy
import numpy as np
import pytest
from jit_dvgc.exploration_training import select_candidate_ticks
from jit_dvgc.exploration_reward import arrival_reward_batch, continuation_reward_batch


def test_deferred_receipt_persists_unknown_without_runtime(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from jit_dvgc import exploration_continuation as m, probe_bank
    bank=dict(bank_sha256='a'*64,max_ticks=4,task=dict(xml_sha256='b'*64,
        success_criterion='first_valid_landing',continuation_start_semantics='fresh_continuation_v1'),
        members=[dict(name=name,roles=['evaluator']) for name in ('first','second')])
    monkeypatch.setattr(probe_bank,'load_probe_bank',lambda path:bank)
    obj=m.FrozenSuffixEvaluator('bank.json',['first','second'],4,tmp_path/'labels',0)
    monkeypatch.setattr(obj,'_runtime',lambda *a:pytest.fail('defer must not create runtime'))
    monkeypatch.setattr(m,'snapshot_context_sha256',lambda s:'c'*64)
    monkeypatch.setattr(m,'physical_state_sha256',lambda s:'d'*64)
    def save(path,snapshot):
        path.mkdir();(path/'identity.json').write_text('{}');(path/'snapshot.pkl').write_bytes(b'fixture')
    monkeypatch.setattr(m,'save_unified_envelope_snapshot',save)
    snapshot=SimpleNamespace(down_events={'valid_contact_seen':False},xml_sha256='b'*64)
    result=obj.defer(snapshot)
    assert result['label'] is None and result['witness'] is None
    assert result['labels']=={'first':None,'second':None}
    assert result['evaluation_status']=='deferred' and result['attempts']==[]
    assert result['charged_interactions']==obj.charged_interactions==0
    assert list((tmp_path/'labels').glob('*/snapshot/snapshot.pkl'))
    assert result['receipt_sha256']==m.canonical_sha256({k:v for k,v in result.items() if k!='receipt_sha256'})
    with pytest.raises(ValueError,match='already recorded'):obj.defer(snapshot)


def test_arrival_credit_does_not_turn_unknown_or_failure_into_witness():
    policy='a'*64
    ledger={'policy_sha256':policy,'cells':[]}
    candidates=[dict(episode_index=e,tick=0,cell='same',label=label,witness=None,
                    state_sha256=str(e+1)*64,context_sha256=str(e+3)*64,receipt_sha256='f'*64,
                    generated_by_env_step_only=True) for e,label in enumerate((None,0))]
    before=deepcopy(candidates)
    reward,updated,evidence=arrival_reward_batch(ledger,candidates,shape=(1,2),expected_policy_sha256=policy)
    np.testing.assert_array_equal(reward,[[.5,.5]])
    assert candidates==before and [r['label'] for r in evidence['candidates']]==[None,0]
    assert evidence['envelope_admission'] is False and updated['cells']==['same']
    strict,strict_ledger,_=continuation_reward_batch(ledger,candidates,shape=(1,2),expected_policy_sha256=policy)
    assert strict.sum()==0 and strict_ledger==ledger
    candidates[0]['generated_by_env_step_only']=False
    with pytest.raises(ValueError,match='forward provenance'):
        arrival_reward_batch(ledger,candidates,shape=(1,2),expected_policy_sha256=policy)


def fixture_frames(phases):
    n=len(phases)
    data=dict(mask=np.ones((n,1),bool),qpos=np.zeros((n,1,3)),qvel=np.zeros((n,1,3)),
              phase=np.array(phases)[:,None],terminal=np.zeros((n,1),bool),finite=np.ones((n,1),bool))
    data['qpos'][:,0,0]=np.arange(n)+1
    data['qvel'][:,0,2]=-1
    data['snap/up/apex_seen']=np.zeros((n,1),bool)
    data['snap/down/valid_contact_seen']=np.zeros((n,1),bool)
    spec=dict(max_candidates_per_episode=4,candidate_min_x=0,candidate_max_x=100,candidate_spacing=.5)
    return data,[str(i) for i in range(n)],spec


def test_two_phase_selector_spans_existing_bins_and_preserves_legacy():
    data,cells,spec=fixture_frames([0]*6+[1]*6)
    assert select_candidate_ticks(data,0,cells,[],spec)==[0,1,2,3]
    spec['candidate_selection']='phase_stratified_v1'
    assert select_candidate_ticks(data,0,cells,[],spec)==[0,5,6,11]
    data['terminal'][11,0]=True
    data['snap/up/apex_seen'][5,0]=True
    assert select_candidate_ticks(data,0,cells,['0'],spec)==[1,4,6,10]


def test_one_phase_and_sparse_phase_redistribute_quota():
    data,cells,spec=fixture_frames([0]*9)
    spec['candidate_selection']='phase_stratified_v1'
    assert select_candidate_ticks(data,0,cells,[],spec)==[0,2,5,8]
    data['phase'][8,0]=1
    assert select_candidate_ticks(data,0,cells,[],spec)==[0,3,7,8]
    assert select_candidate_ticks(data,0,cells,cells,spec)==[]


def test_frozen_critic_network_matches_original_ppo_cpu():
    import jax
    import jax.numpy as jp
    from jit_dvgc.ppo import make_network_factory
    from jit_dvgc.exploration_network import make_exploration_network_factory
    shape={'state':76,'privileged_state':106}
    original=make_network_factory()(shape,4)
    explorer=make_exploration_network_factory()(shape,4)
    params=original.value_network.init(jax.random.PRNGKey(8))
    obs={'state':jp.ones((2,76)), 'privileged_state':jp.arange(212,dtype=jp.float32).reshape(2,106)}
    from brax.training.acme import running_statistics
    normalizer=running_statistics.init_state({key:jp.zeros(size) for key,size in shape.items()})
    expected=original.value_network.apply(normalizer,params,obs)
    actual=explorer.value_network.apply(normalizer,params,obs)
    np.testing.assert_array_equal(actual,expected)
