import numpy as np
import pytest
from jit_dvgc.generative_bridge import production as p


def evidence(tmp_path,monkeypatch,labels=None):
    runner=p.ProductionRunner.__new__(p.ProductionRunner)
    runner.root=tmp_path
    runner.spec={'teacher_layout':'source_control_in_candidate_batch',
                 'teacher_colored_noise_candidates':0,
                 'teacher_replay_failure_policy':'reject_finite_candidate',
                 'source':dict(actor_sha256='a',normalizer_sha256='n',xml_sha256='m'),
                 'source_runtime':dict(horizon=400)}
    rows=[dict(candidate_id=i,root_id='r',snapshot_context_sha256='ctx',label=int(i==6),
        attempts=[dict(trace_lane=i,actor_sha256='a',normalizer_sha256='n',model_sha256='m',
                       snapshot_context_sha256='ctx',label=int(i==6),steps=2)]) for i in range(17)]
    repeated=[dict(r,label=0,attempts=[dict(r['attempts'][0],label=0)]) for r in rows]
    if labels is not None:repeated[labels[0]]['label']=labels[1]
    runner.evaluate=lambda *a,**k:repeated
    monkeypatch.setattr(p,'lane_arrays',lambda attempt:{'qpos':np.zeros((2,12)),'done':np.array([False,True]),
        'valid_contact':np.array([False,bool(attempt['label'])]),
        'physical_failure':np.array([False,not bool(attempt['label'])]),'timeout':np.array([False,False])})
    return runner,rows


def test_complete_negative_repeat_is_rejected_without_demo(tmp_path,monkeypatch):
    runner,rows=evidence(tmp_path,monkeypatch)
    replay=runner.replay_teacher(1,rows,np.zeros((17,16,4)),6,search_results=rows)
    rejected=p.reject_finite_replay({'root_id':'r','source_recheck_label':0},replay)
    assert rejected['teacher_status']=='replay_rejected'
    assert rejected['training_eligible'] and 'demo' not in rejected
    assert rejected['selected_replay_label']==0 and rejected['source_recheck_label']==0
    from jit_dvgc.generative_bridge.outcomes import root_outcome
    outcome=root_outcome({**rejected,'student_label':1,'student_adopted':True})
    assert outcome['teacher_found'] is None and not outcome['has_direct_teacher_demo']
    from jit_dvgc.generative_bridge.closed_loop import feedback_rows
    rows=feedback_rows([dict(root_id='r',label=0)],[dict(root_id='r',label=1)],{'r':rejected},{'adopted':True})
    assert rows[0]['initial_label']==0 and rows[0]['label']==1


@pytest.mark.parametrize('label',[None,1])
def test_unknown_or_changed_source_cannot_be_rejected_as_finite_failure(tmp_path,monkeypatch,label):
    runner,rows=evidence(tmp_path,monkeypatch,(0,label))
    replay=runner.replay_teacher(1,rows,np.zeros((17,16,4)),6,search_results=rows)
    assert p.reject_finite_replay({},replay) is None


def test_nonfinite_repeat_still_stops(tmp_path,monkeypatch):
    runner,rows=evidence(tmp_path,monkeypatch)
    monkeypatch.setattr(p,'lane_arrays',lambda attempt:{'qpos':np.array([[np.nan]])})
    with pytest.raises(ValueError,match='nonfinite'):
        runner.replay_teacher(1,rows,np.zeros((17,16,4)),6,search_results=rows)


def test_legacy_mode_does_not_silently_enable_rejection(tmp_path,monkeypatch):
    runner,rows=evidence(tmp_path,monkeypatch);runner.spec.pop('teacher_replay_failure_policy')
    replay=runner.replay_teacher(1,rows,np.zeros((17,16,4)),6,search_results=rows)
    assert p.reject_finite_replay({},replay) is None


def test_unknown_other_lane_is_not_silently_ignored(tmp_path,monkeypatch):
    runner,rows=evidence(tmp_path,monkeypatch,(3,None))
    replay=runner.replay_teacher(1,rows,np.zeros((17,16,4)),6,search_results=rows)
    assert p.reject_finite_replay({},replay) is None


def test_missing_trace_is_still_an_engineering_error(tmp_path,monkeypatch):
    runner,rows=evidence(tmp_path,monkeypatch);rows[2]['attempts'].clear()
    with pytest.raises(ValueError,match='missing completed'):
        runner.replay_teacher(1,rows,np.zeros((17,16,4)),6,search_results=rows)


@pytest.mark.parametrize('change',[{'actor_sha256':'wrong'},{'normalizer_sha256':'wrong'},
    {'model_sha256':'wrong'},{'snapshot_context_sha256':'wrong'},{'trace_lane':9},{'label':1},{'steps':3}])
def test_corrupt_replay_identity_or_length_still_stops(tmp_path,monkeypatch,change):
    runner,rows=evidence(tmp_path,monkeypatch);rows[3]['attempts'][0].update(change)
    with pytest.raises(ValueError):
        runner.replay_teacher(1,rows,np.zeros((17,16,4)),6,search_results=rows)


def test_partial_physical_trace_still_stops(tmp_path,monkeypatch):
    runner,rows=evidence(tmp_path,monkeypatch);original=p.lane_arrays
    monkeypatch.setattr(p,'lane_arrays',lambda a:{**original(a),'done':np.array([False,False])})
    with pytest.raises(ValueError,match='incomplete'):
        runner.replay_teacher(1,rows,np.zeros((17,16,4)),6,search_results=rows)
