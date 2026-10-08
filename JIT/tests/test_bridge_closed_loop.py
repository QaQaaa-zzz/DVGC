import pytest
from jit_dvgc.generative_bridge.closed_loop import continuation_decision, feedback_rows, stress_conditions


def test_forgetting_does_not_veto_experimental_continuation():
    decision=continuation_decision({'actor_sha256':'a','normalizer_sha256':'n'},[1,1,1,1],True)
    assert decision['adopted'] and not decision['formal_adopted']
    assert decision['scope']=='experimental_continuation'


def test_unknown_nominal_or_incomplete_learner_is_not_adoption():
    assert not continuation_decision({},[1,None],True)['adopted']
    assert not continuation_decision({},[1,1],False)['adopted']
    assert not continuation_decision({},[],True)['adopted']


def test_feedback_student_success_does_not_relabel_teacher():
    original=[{'root_id':'x','label':0,'teacher_status':'searched_no_solution'},
              {'root_id':'y','label':0}]
    after=[{'root_id':'x','label':1},{'root_id':'y','label':1}]
    teachers={'y':{'teacher_status':'invalid','source_control_labels':[0,1]}}
    rows=feedback_rows(original,after,teachers,{'adopted':True})
    assert rows[0]['label']==1 and rows[0]['teacher_status']=='searched_no_solution'
    assert rows[0]['successor_adopted']
    assert rows[1]['source_recheck_label'] is None and rows[1]['initial_label'] is None
    assert original[0]['label']==0


def test_fixed_stress_panel_is_separate_from_train():
    panel=stress_conditions()
    assert len(panel)==9
    assert {r['amplitude'] for r in panel}=={.25,.4,.6}
    assert len({i for r in panel for i in r['episode_ids']})==288
    assert all(r['role']=='student_dev' and r['round']==0 for r in panel)


def test_stress_comparison_preserves_unknown_and_conditions():
    from jit_dvgc.generative_bridge.closed_loop import compare_stress
    baseline={'c':{'condition':{'amplitude':.4},'labels':[1,0,1,None]}}
    student={'c':{'condition':{'amplitude':.4},'labels':[0,1,None,1]}}
    result=compare_stress(baseline,student)['c']
    assert result['gained']==1 and result['lost']==1 and result['unknown']==2
    student['c']['condition']['amplitude']=.6
    with pytest.raises(ValueError,match='conditions'):
        compare_stress(baseline,student)


def test_child_overcharge_is_not_accepted(tmp_path):
    from jit_dvgc.generative_bridge.closed_loop import ClosedLoopRound
    from jit_dvgc.generative_bridge.protocol import atomic_json
    runner=ClosedLoopRound.__new__(ClosedLoopRound);runner.root=tmp_path
    runner.spec={};runner.costs=[];runner.child=lambda *args,**kwargs: {}
    atomic_json(tmp_path/'worker/status.json',{'phase':'completed','charged_interactions':11})
    with pytest.raises(ValueError,match='reservation'):
        runner.measured('worker','collect',{},10)


def test_notification_start_failure_is_recorded(tmp_path,monkeypatch):
    import jit_dvgc.generative_bridge.closed_loop as loop
    from jit_dvgc.generative_bridge.production import read
    def fail(*args):raise RuntimeError('notification failed')
    monkeypatch.setattr(loop,'start_notifications',fail)
    plan={'output':str(tmp_path),'prior_physics_charged':0,'budgets':{'max_physics':10}}
    with pytest.raises(RuntimeError,match='notification failed'):loop.run_closed_loop(plan)
    assert read(tmp_path/'status.json')['phase']=='failed'
