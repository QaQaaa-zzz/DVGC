import pytest
import numpy as np
from jit_dvgc.retention_next import budget_layout, qualify_method, choose_roots, baseline_class
from jit_dvgc.neighborhood import FrozenNeighborhood, FIELDS


def test_budget_charges_every_replay_lane_and_collection_padding():
    b=budget_layout('D1')
    assert b['teacher_search_and_replay']==40*2*3*17*400
    assert b['total']<=2000000
    assert budget_layout('V')['total']==800000


def test_mixed_source_or_winner_never_becomes_reliable_lesson():
    assert qualify_method([0,0,0],[1,1,1])=='verified_solution'
    assert qualify_method([0,1,0],[1,1,1])=='source_ambiguous'
    assert qualify_method([0,0,0],[1,0,1])=='winner_ambiguous'
    assert qualify_method([0,0,0],None)=='searched_no_solution'
    assert baseline_class([0,0,0])=='stable_failure'
    assert baseline_class([0,1,0])=='baseline_ambiguous'
    with pytest.raises(ValueError): baseline_class([0,None])


def test_selection_is_balanced_by_onset_and_excludes_unstable_baselines():
    rows=[dict(root_id=str(i),onset=(0,5,10,15)[i%4]) for i in range(30)]
    labels={str(i):[0,0,0] for i in range(30)};labels['0']=[0,1,0]
    chosen=choose_roots(rows,labels,16)
    assert len(chosen)==16 and '0' not in [r['root_id'] for r in chosen]
    assert [sum(r['onset']==o for r in chosen) for o in (0,5,10,15)]==[4]*4


def test_other_actor_labels_cannot_populate_current_pi0_flags():
    rows=[dict(coordinates=dict.fromkeys(FIELDS,0.),phase='upstream',snapshot_context_sha256='same',
               evaluated_actor_sha256='R73',label=1,evidence_kind='actor',data_role='train')]
    idx=FrozenNeighborhood(rows,'pi0',dict(evidence_scope='train_history_v1'))
    v=idx.query(np.zeros((1,12)),np.zeros(1,int))
    assert v[0,12:14].tolist()==[0.,0.]
    assert v[0,14]==1


def test_resume_budget_retains_failed_reservation_and_only_remaining_roots():
    from jit_dvgc.retention_next import resume_budget
    b=resume_budget(116258,12,489.315)
    assert b['total']==116258+12*2*3*17*400
    assert b['inherited_charge']==116258
    assert b['wall_seconds']<21600
    with pytest.raises(ValueError):resume_budget(1900000,12,489)
