from copy import deepcopy
import pytest
from jit_dvgc.generative_bridge.promotion import run_promotion_series


def plan(root):
    return dict(schema='jit_bridge_promotion_series_v1_2',execute=True,output=str(root),rounds=2,
        gates={k:True for k in ('source_validated','absorption','adopted_candidate','demo_control')},
        initial=dict(actor={'sha256':'P0'},generator={'sha256':'G0'},explorer={'sha256':'E0'},demo_bank={'root_ids':[]}),
        budgets=dict(max_physics=20,max_updates=20,max_wall_seconds=100,
            per_stage={s:{'physics':1,'updates':1} for s in ('collect_fresh','teacher_student','accept','update_explorer','update_generator')}))


def callbacks(log,fail=False):
    def collect(c):
        log.append(deepcopy(c))
        return dict(records=[dict(root_id=str(c['round']),episode_id=str(c['round']),round=c['round'],role='train',
            behavior=c['frozen'],mode='learned',raw_log_prob=-.2)],physics=1,updates=0)
    def teach(c):
        ids=c['previous']['demo_bank']['root_ids']+[str(c['round'])]
        return dict(candidate_actor={'sha256':'P'+str(c['round'])},demo_bank={'root_ids':ids},
            teacher_feedback=[dict(role='train',valid=True,root_id=str(c['round']))],
            student_feedback=[dict(role='train',valid=True,root_id=str(c['round']))],physics=1,updates=1)
    def accept(c):
        return dict(adopted=c['round']==2,outcomes=[dict(root_id=str(c['round']),source_label=0,student_label=1)],physics=1,updates=0)
    def explorer(c):
        assert len(c['onpolicy_records'])==1
        assert c['feedback'][0]['adopted_gain']==(c['round']==2)
        from jit_dvgc.generative_bridge.contracts import digest
        return dict(explorer={'sha256':'E'+str(c['round'])},physics=0,updates=1,
                    training_episode_ids=[r['episode_id'] for r in c['onpolicy_records']],behavior_sha256=digest(c['frozen']))
    def generator(c):
        if fail and c['round']==2:raise RuntimeError('generator failure')
        assert len(c['eligible_feedback'])==(2 if c['round']==2 else 1)
        return dict(generator={'sha256':'G'+str(c['round'])},physics=0,updates=1)
    return dict(collect_fresh=collect,teacher_student=teach,accept=accept,update_explorer=explorer,update_generator=generator)


def test_two_round_reject_then_accept_keeps_demos_and_frozen_source(tmp_path):
    log=[];result=run_promotion_series(plan(tmp_path/'run'),callbacks(log))
    assert log[1]['frozen']['actor']=={'sha256':'P0'}
    assert log[1]['frozen']['generator']=={'sha256':'G1'}
    assert result['actor']=={'sha256':'P2'}
    assert result['demo_bank']['root_ids']==['1','2']
    assert result['completed_round']==2


def test_failure_does_not_partially_promote(tmp_path):
    import json
    root=tmp_path/'run'
    with pytest.raises(RuntimeError,match='generator failure'):
        run_promotion_series(plan(root),callbacks([],True))
    committed=json.loads((root/'current_source.json').read_text())
    assert committed['completed_round']==1 and committed['actor']=={'sha256':'P0'}
    assert committed['explorer']=={'sha256':'E1'}
    assert committed['demo_bank']['root_ids']==['1']
    assert (root/'round_0002/teacher_student.json').exists()
    with pytest.raises(FileExistsError):run_promotion_series(plan(root),callbacks([]))


def test_requires_gates_and_explicit_execution(tmp_path):
    p=plan(tmp_path/'run');p['gates']['absorption']=False
    with pytest.raises(ValueError,match='gates'):run_promotion_series(p,callbacks([]))
    p['gates']['absorption']=True;p['execute']=False
    with pytest.raises(ValueError,match='execute'):run_promotion_series(p,callbacks([]))


def test_rejects_stale_collection(tmp_path):
    cb=callbacks([]);old=cb['collect_fresh']
    def stale(c):
        result=old(c);result['records'][0]['round']=0;return result
    cb['collect_fresh']=stale
    with pytest.raises(ValueError,match='current round'):run_promotion_series(plan(tmp_path/'run'),cb)
    assert not (tmp_path/'run/current_source.json').exists()


def test_random_and_unknown_cannot_enter_explorer_update(tmp_path):
    cb=callbacks([]);old=cb['collect_fresh']
    def random(c):
        result=old(c);result['records'][0].update(mode='uniform_random',raw_log_prob=None);return result
    cb['collect_fresh']=random
    def stale_update(c):
        from jit_dvgc.generative_bridge.contracts import digest
        assert c['onpolicy_records']==[]
        return dict(explorer={'sha256':'Ebad'},training_episode_ids=['old-root'],
                    behavior_sha256=digest(c['frozen']),physics=0,updates=1)
    cb['update_explorer']=stale_update
    with pytest.raises(ValueError,match='current on-policy'):
        run_promotion_series(plan(tmp_path/'run'),cb)
    assert not (tmp_path/'run/current_source.json').exists()


def test_stage_reservation_stops_before_publication(tmp_path):
    cb=callbacks([]);old=cb['teacher_student']
    def over_budget(c):
        result=old(c);result['physics']=2;return result
    cb['teacher_student']=over_budget
    with pytest.raises(ValueError,match='cost exceeds'):
        run_promotion_series(plan(tmp_path/'run'),cb)
    assert not (tmp_path/'run/current_source.json').exists()


def test_real_logical_episode_zero_is_valid(tmp_path):
    cb=callbacks([]);old=cb['collect_fresh']
    def zero_first(c):
        result=old(c)
        result['records'][0]['episode_id']=c['round']-1
        return result
    cb['collect_fresh']=zero_first
    result=run_promotion_series(plan(tmp_path/'run'),cb)
    assert result['completed_round']==2
