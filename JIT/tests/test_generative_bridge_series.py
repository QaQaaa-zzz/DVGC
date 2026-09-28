import pytest


def test_series_panels_are_predeclared_disjoint_train_roots():
    from jit_dvgc.generative_bridge.series import plan_panels
    rows=[dict(root_id=str(i),root_episode_id=str(i),label=0) for i in range(162)]
    panels={'original_pending':rows,'new_roots':rows[:32],
            'splits':{str(i):'generator_train' if i<130 else 'generator_dev' for i in range(162)}}
    result=plan_panels(panels,3,32,9)
    assert [len(x) for x in result]==[32]*3
    ids=[r['root_id'] for group in result for r in group]
    assert len(set(ids))==96 and not set(ids)&{str(i) for i in range(32)}
    assert all(panels['splits'][r['root_episode_id']]=='generator_train' for g in result for r in g)
    assert len(panels['original_pending'])==162
    with pytest.raises(ValueError,match='distinct TRAIN'):
        plan_panels(panels,4,32,9)


def test_all_previous_corpus_groups_become_history_without_erasing_adoption():
    from jit_dvgc.generative_bridge.series import corpus_history
    adopted={'metadata':{'origin_type':'adopted_actor'},'adoption':{'adopted':True,'actor_sha256':'exact'}}
    corpus={'groups':{'history':[{'id':'old'}],'teacher_new':[{'id':'teacher'}],'actor_new':[adopted]}}
    history=corpus_history(corpus)
    assert history==[{'id':'old'},{'id':'teacher'},adopted]
    assert history[-1]['adoption']['actor_sha256']=='exact'


def test_rejected_students_do_not_stop_declared_series(tmp_path,monkeypatch):
    from jit_dvgc.generative_bridge import series as s
    from jit_dvgc.generative_bridge.protocol import atomic_json
    monkeypatch.setattr(s,'implementation_identity',lambda p:'code')
    monkeypatch.setattr(s,'start_notifications',lambda m:None)
    seen=[]
    def prepare(spec,previous,index,started):
        root=tmp_path/f'round_{index+1:04d}';root.mkdir()
        seen.append((index,str(previous),started))
        atomic_json(root/'costs.json',[])
        return {'output':str(root)}
    monkeypatch.setattr(s,'prepare_next_round',prepare)
    class Runner:
        def __init__(self,spec):self.root=s.Path(spec['output'])
        def run(self):
            atomic_json(self.root/'result_matrix.json',{'valid':True})
            atomic_json(self.root/'costs.json',[{'charged_interactions':10,'charged_updates':2}])
            return {'actor_acceptance':{'adopted':False},'generator':{'path':self.root.name}}
    monkeypatch.setattr(s,'ProductionRunner',Runner)
    spec={'output':str(tmp_path),'previous':'previous','repository':'repo','implementation_commit':'code',
          'locks':{},'rounds':3,'prior_physics':7,'budgets':{'max_physics':30,'max_supervised_updates':6,'max_wall_seconds':100}}
    s.run_series(spec)
    status=s.read(tmp_path/'status.json')
    assert status['phase']=='completed' and status['completed_rounds']==3
    assert status['new_physics_charged_or_reserved']==30 and status['lifetime_physics_charged_or_reserved']==37
    assert [x[0] for x in seen]==[0,1,2] and len({x[2] for x in seen})==1
    assert seen[1][1]==str(tmp_path/'round_0001')
    with pytest.raises(ValueError,match='already started'):s.run_series(spec)


def test_series_startup_failure_is_recorded(tmp_path,monkeypatch):
    from jit_dvgc.generative_bridge import series as s
    from jit_dvgc.generative_bridge.protocol import atomic_json
    atomic_json(tmp_path/'status.json',{'phase':'prepared'})
    monkeypatch.setattr(s,'implementation_identity',lambda p:'changed')
    spec={'output':str(tmp_path),'previous':'parent','repository':'repo','implementation_commit':'expected',
          'locks':{},'rounds':3,'prior_physics':7,'budgets':{'max_physics':30,'max_supervised_updates':6,'max_wall_seconds':100}}
    with pytest.raises(ValueError,match='series code drift'):s.run_series(spec)
    assert s.read(tmp_path/'status.json')['phase']=='failed'
