"""CPU-only contracts for opt-in candidate training and unchanged witness panels."""
from copy import deepcopy
import pytest
from jit_dvgc.evidence_integrity import canonical_sha256
from jit_dvgc.jump_evidence_validation import write
from jit_dvgc.iterative_probe_training import (
    candidate_support_view, validate_candidate_support, make_config, load_config,
    fixed_train_panel_identity, CANDIDATE_SCHEMA,
)


def seal(value):
    value.pop('support_sha256', None)
    value['support_sha256'] = canonical_sha256(value)
    return value


@pytest.fixture
def sources(tmp_path):
    def row(key, phase, witnessed, trajectory):
        path = tmp_path / key
        path.mkdir()
        (path/'identity.json').write_text('{}')
        (path/'snapshot.pkl').write_bytes(b'fixture complete snapshot')
        return dict(key=key, phase=phase, witnessed=witnessed, snapshot=str(path),
                    trajectory_id=trajectory, sampling_weight=1., labels={'pi_0':int(witnessed)},
                    state_sha256=key, snapshot_context_sha256=key,
                    evidence_status='witnessed' if witnessed else 'pending')
    witness = seal(dict(schema='jit_iterative_witnessed_support_v1', role='train',
        final_test_used=False, inputs={}, entries=[row('wu','upstream',True,'w'),row('wd','downstream',True,'w')]))
    pending = [row('p1','upstream',False,'a'),row('p2','upstream',False,'a'),row('p3','upstream',False,'b')]
    return witness, pending


def test_pending_quota_and_group_balancing_without_false_witness(sources):
    witness, pending = sources
    before = deepcopy(witness)
    support = candidate_support_view(witness,pending,{})
    assert witness == before
    assert support['realized_pending_fraction_by_phase'] == {'upstream':.25,'downstream':0.}
    rows = {r['key']:r for r in support['entries']}
    assert rows['wu']['sampling_weight'] == .75
    assert rows['wd']['sampling_weight'] == 1.
    assert rows['p1']['sampling_weight'] == rows['p2']['sampling_weight'] == .0625
    assert rows['p3']['sampling_weight'] == .125
    assert all(r['witnessed'] is False and r['evidence_status']=='pending' for k,r in rows.items() if k.startswith('p'))
    assert len(support['inputs']) == 10


def test_round_robin_cap_retains_groups(sources):
    support = candidate_support_view(*sources,{},max_pending_per_phase=2)
    assert {r['key'] for r in support['entries'] if not r['witnessed']} == {'p1','p3'}
    assert validate_candidate_support(support) == support


@pytest.mark.parametrize('mutation', ['positive','role','duplicate','weight','missing_lock','quota'])
def test_invalid_candidate_support_rejected(sources,mutation):
    support = candidate_support_view(*sources,{})
    row = next(r for r in support['entries'] if not r['witnessed'])
    if mutation == 'positive': row['labels']['pi_1'] = 1
    if mutation == 'role': row['role'] = 'test'
    if mutation == 'duplicate': support['entries'].append(deepcopy(row))
    if mutation == 'weight': row['sampling_weight'] *= 2
    if mutation == 'missing_lock': support['inputs'].clear()
    if mutation == 'quota': support['realized_pending_fraction_by_phase']['downstream'] = .25
    with pytest.raises(ValueError): validate_candidate_support(seal(support))


def test_no_pending_is_not_silent_baseline(sources):
    with pytest.raises(ValueError,match='at least one pending'):
        candidate_support_view(sources[0],[],{})


def test_candidate_config_keeps_witnessed_panel_and_locks_snapshots(sources,tmp_path,jit_root):
    witness,pending = sources
    support=candidate_support_view(witness,pending,{})
    support_path=tmp_path/'candidate.json';write(support_path,support)
    panel_path=tmp_path/'witness.json';write(panel_path,witness)
    initializer=tmp_path/'frozen.json';write(initializer,{})
    output=tmp_path/'config.json'
    raw=make_config(support_path,initializer,jit_root/'configs/pi_unified_formal.json',output,
        'candidate',7,128000,123,checkpoints=[32000,64000,128000],panel_support_path=panel_path,pending_fraction=.25,panel_samples_per_phase=1)
    config=load_config(output)
    assert config.schema==CANDIDATE_SCHEMA
    assert config.ppo.requested_transitions==128000
    assert raw['fixed_train_panel']==fixed_train_panel_identity(witness,raw['checkpoint_evaluation'])
    assert all(r['entry']['witnessed'] for r in raw['fixed_train_panel']['selected_entries'])
    assert config.formal.checkpoint_transitions==(0,32000,64000,128000)
    from pathlib import Path
    (Path(pending[0]['snapshot'])/'snapshot.pkl').write_bytes(b'changed')
    with pytest.raises(ValueError,match='support input changed'): load_config(output)
