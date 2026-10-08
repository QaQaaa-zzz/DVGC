import pytest


def test_locked_panels_leave_original_pending_intact_and_ancestors_disjoint():
    from jit_dvgc.generative_bridge.production import select_panels
    rows=[dict(index=i,label=0 if i<60 else 1,prefix_terminal=False,
        snapshot_context_sha256=str(i),root_episode_id='ancestor_'+str(i)) for i in range(300)]
    panels=select_panels(rows,seed=7,new_roots=32,bootstrap=32,core=64,protected=64)
    assert len(panels['original_pending'])==60
    assert len(panels['new_roots'])==32
    groups=[panels[k] for k in ('new_roots','bootstrap','core','protected')]
    ids=[{r['root_episode_id'] for r in group} for group in groups]
    assert all(not a&b for i,a in enumerate(ids) for b in ids[i+1:])
    assert all(panels['splits'][r['root_episode_id']]=='generator_train' for r in panels['new_roots'])
    assert {panels['splits'][r['root_episode_id']] for r in panels['bootstrap']}=={'generator_train','generator_dev'}
    assert rows[0]['label']==0 and 'attempts' not in rows[0]


def test_adoption_is_exact_student_independent_of_teacher_with_protected_threshold():
    from jit_dvgc.generative_bridge.production import acceptance_decision
    old={'new_roots':[0,0], 'core':[1,1], 'protected':[1]*64,'nominal':[1]}
    new={'new_roots':[1,0], 'core':[1,1], 'protected':[1]*63+[0],'nominal':[1]}
    identity={'actor_sha256':'student','normalizer_sha256':'norm'}
    r=acceptance_decision(old,new,identity)
    assert r['adopted'] and r['actor_sha256']=='student'
    new['core']=[1,0]
    assert not acceptance_decision(old,new,identity)['adopted']
    new['core']=[1,None]
    assert not acceptance_decision(old,new,identity)['adopted']


def test_wall_deadline_interrupts_blocked_wait_and_restores_signal():
    import signal,time
    from jit_dvgc.generative_bridge.production import wall_deadline
    prior=signal.getsignal(signal.SIGALRM)
    with pytest.raises(TimeoutError,match='whole pilot'):
        with wall_deadline(.02):time.sleep(.3)
    assert signal.getsignal(signal.SIGALRM)==prior
    assert signal.getitimer(signal.ITIMER_REAL)==(0.,0.)


def test_production_generator_retry_reconciles_cost_and_keeps_committed_student(tmp_path):
    from jit_dvgc.generative_bridge.production import ProductionRunner
    from jit_dvgc.generative_bridge.protocol import StageJournal,atomic_json
    from jit_dvgc.generative_bridge.contracts import file_sha
    runner=ProductionRunner.__new__(ProductionRunner)
    runner.root=tmp_path;runner.costs=[];runner.retry_generator=False
    runner.spec={'budgets':{'max_wall_seconds':100},'source_frozen_policy':'locked','seed':3}
    runner.journal=StageJournal(tmp_path/'stages',{'round':1})
    student_calls=[]
    student=lambda:student_calls.append(1) or {'actor':'committed'}
    runner.journal.stage('student_training',{},student)
    corpus={'new_data':True,'path':'locked','sha256':'locked'}
    dev=tmp_path/'dev.npz';dev.write_bytes(b'fixture')
    calls=[]
    def child(name,argv,maximum,*,updates):
        calls.append(updates);runner.costs.append({'stage':name,'charged_updates':updates,'charged_interactions':0})
        if len(calls)==1:
            atomic_json(tmp_path/'generator_incremental/attempt_0000/cost_receipt.json',{'charged_updates':3})
            raise RuntimeError('deliberate failed G attempt')
        checkpoint=tmp_path/'checkpoint';checkpoint.mkdir()
        (checkpoint/'state.msgpack').write_bytes(b'hashed-payload-fixture')
        atomic_json(checkpoint/'manifest.json',{'state_sha256':file_sha(checkpoint/'state.msgpack')})
        atomic_json(tmp_path/'incremental_result.json',{'status':'completed',
            'checkpoint_manifest':str(checkpoint/'manifest.json'),
            'checkpoint_manifest_sha256':file_sha(checkpoint/'manifest.json')})
    runner.child=child
    with pytest.raises(RuntimeError,match='deliberate'):
        runner.generator('incremental',corpus,dev,incumbent={'id':'old'})
    assert runner.costs[0]['charged_updates']==3
    with pytest.raises(RuntimeError,match='explicit --retry-generator'):
        runner.generator('incremental',corpus,dev,incumbent={'id':'old'})
    runner.retry_generator=True
    assert runner.generator('incremental',corpus,dev,incumbent={'id':'old'})['status']=='completed'
    runner.journal.stage('student_training',{},student)
    assert student_calls==[1] and calls==[2000,1997]
    assert sum(c['charged_updates'] for c in runner.costs)==2000


def test_v12_teacher_replay_keeps_source_control_in_same_lane():
    import numpy as np
    from jit_dvgc.generative_bridge.production import ProductionRunner
    runner=ProductionRunner.__new__(ProductionRunner)
    runner.spec={'teacher_layout':'source_control_in_32_world_batch'}
    rows=[{'candidate_id':i} for i in range(32)]
    def evaluate(name,candidates,**kw):
        assert kw['prefixes'].shape==(32,16,4)
        assert kw['source_only'].tolist()==[True]+[False]*31
        return [{**r,'label':int(r['candidate_id']==7)} for r in candidates]
    runner.evaluate=evaluate
    assert runner.replay_teacher(0,rows,np.zeros((32,16,4)),7)['candidate_id']==7


def test_reduced_teacher_replay_keeps_all_17_lanes_and_source_control(tmp_path):
    import numpy as np
    from jit_dvgc.generative_bridge.production import ProductionRunner
    runner=ProductionRunner.__new__(ProductionRunner);runner.root=tmp_path
    runner.spec={'teacher_layout':'source_control_in_candidate_batch','teacher_colored_noise_candidates':0}
    rows=[{'candidate_id':i,'label':int(i==7)} for i in range(17)]
    def evaluate(name,candidates,*,prefixes,source_only):
        assert [r['candidate_id'] for r in candidates]==list(range(17))
        assert prefixes.shape==(17,16,4)
        assert source_only.tolist()==[True]+[False]*16
        return rows
    runner.evaluate=evaluate
    selected=runner.replay_teacher(0,rows,np.zeros((17,16,4)),7,search_results=rows)
    assert selected['label']==1 and selected['candidate_id']==7
