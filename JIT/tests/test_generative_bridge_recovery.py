import pytest
from jit_dvgc.generative_bridge.protocol import atomic_json,StageJournal
from jit_dvgc.generative_bridge.contracts import file_sha


def test_rechecked_source_success_is_not_teacher_rescue():
    from jit_dvgc.generative_bridge.production import source_recheck_disposition
    r=source_recheck_disposition(1)
    assert r['teacher_status']=='not_scheduled' and r['training_eligible']
    assert r['source_recheck_label']==1 and not r['new_gain_eligible']
    assert source_recheck_disposition(0) is None
    with pytest.raises(ValueError,match='unknown'):
        source_recheck_disposition(None)


@pytest.mark.parametrize('recheck,final,quality',[(1,1,0.),(0,1,2.),(1,0,-.1),(0,0,-.1)])
def test_gain_uses_current_source_without_erasing_pending_failure(recheck,final,quality):
    from jit_dvgc.generative_bridge.rewards import feedback
    row=dict(initial_label=0,label=final,learning_attempted=True,successor_adopted=True,
             source_recheck_label=recheck,pulse_cells=['real'])
    reward,mask,ledger,parts=feedback([row],[])
    assert mask[0] and parts['quality'][0]==pytest.approx(quality)
    assert reward[0]==pytest.approx(quality+.02)
    assert row['successor_adopted'] is True


def test_recovery_import_checks_original_inputs_and_receipt(tmp_path):
    from jit_dvgc.generative_bridge.recovery import RecoveryJournal
    old=StageJournal(tmp_path/'old',{'version':'old'})
    old.stage('bootstrap',{'roots':[1]},lambda:{'done':True})
    path=old.path/'bootstrap.json'
    recovery={'previous_contract_identity':old.identity,
              'stages':{'bootstrap':{'path':str(path),'sha256':file_sha(path)}}}
    new=RecoveryJournal(tmp_path/'new',{'recovery':recovery})
    with pytest.raises(ValueError,match='inputs'):
        new.stage('bootstrap',{'roots':[2]},lambda:pytest.fail('must not execute'))
    assert new.stage('bootstrap',{'roots':[1]},lambda:pytest.fail('must not execute'))=={'done':True}
    assert path.read_text() and not list(old.path.glob('*.running.json'))
    other=RecoveryJournal(tmp_path/'other',{'recovery':recovery})
    atomic_json(path,{'tampered':True})
    with pytest.raises(ValueError,match='receipt'):
        other.stage('bootstrap',{'roots':[1]},lambda:pytest.fail('must not execute'))


def test_all_source_positive_produces_no_demo_without_sampling(tmp_path,monkeypatch):
    from jit_dvgc.generative_bridge.production import ProductionRunner
    from jit_dvgc.generative_bridge import worker,diffusion
    monkeypatch.setattr(worker,'generator_template',lambda *a:(None,None,None))
    monkeypatch.setattr(diffusion,'restore_state',lambda *a:None)
    monkeypatch.setattr(diffusion,'ddim_sample',lambda *a:pytest.fail('must not generate fake lessons'))
    runner=ProductionRunner.__new__(ProductionRunner)
    runner.root=tmp_path;runner.spec={'source_frozen_policy':'locked','seed':1}
    runner.source={'actor_sha256':'source'}
    roots=[dict(root_id=str(i),label=1,attempts=[{}]) for i in range(3)]
    runner.panels={'new_roots':roots,'original_pending':list(roots)}
    runner.evaluate=lambda *a:roots
    result=runner.teacher_search({'checkpoint_manifest':'/unused/manifest.json'})
    assert len(result)==3 and all(r['teacher_status']=='not_scheduled' and 'demo' not in r for r in result.values())
    assert len(runner.panels['original_pending'])==3
    assert not runner.teacher_in_progress


def test_replay_keeps_full_search_batch_lane_and_actions(tmp_path):
    import numpy as np
    from jit_dvgc.generative_bridge.production import ProductionRunner
    runner=ProductionRunner.__new__(ProductionRunner)
    rows=[{'candidate_id':i,'index':i,'root_id':'same'} for i in range(1,32)]
    actions=np.arange(31*16*4,dtype=np.float32).reshape(31,16,4)/2000
    calls=[]
    def evaluate(name,candidates,**kwargs):
        calls.append(name)
        assert candidates==rows
        np.testing.assert_array_equal(kwargs['prefixes'],actions)
        assert kwargs['source_only'].tolist()==[False]*31
        return [{**r,'label':1 if r['candidate_id']==11 else 0,'attempts':[{'trace_lane':i}]} for i,r in enumerate(rows)]
    runner.evaluate=evaluate
    selected=runner.replay_teacher(17,rows,actions,11)
    assert selected['label']==1 and selected['attempts'][0]['trace_lane']==10
    assert calls==['replay_batch_0017']  # Old single-lane failed cache cannot satisfy this stage.


def test_replay_rejects_incomplete_or_reordered_batch():
    import numpy as np
    from jit_dvgc.generative_bridge.production import ProductionRunner
    runner=ProductionRunner.__new__(ProductionRunner)
    rows=[{'candidate_id':i} for i in range(1,32)]
    runner.evaluate=lambda *a,**k:rows[::-1]
    with pytest.raises(ValueError,match='replay candidate ordering'):
        runner.replay_teacher(17,rows,np.zeros((31,16,4)),11)


def test_recovery_of_replay_failure_reuses_completed_teachers_and_costs(tmp_path,monkeypatch):
    from jit_dvgc.generative_bridge.recovery import prepare_recovery
    from jit_dvgc.generative_bridge.contracts import digest
    from jit_dvgc.generative_bridge import production
    import time,json
    old=tmp_path/'old';old.mkdir();output=tmp_path/'new'
    monkeypatch.setattr(production,'implementation_identity',lambda p:'new-code')
    monkeypatch.setattr(production,'implementation_files',lambda p:{})
    spec={'source':{'actor_sha256':'actor'},'locks':{},'budgets':{'max_wall_seconds':43200,'max_physics':698400},
          'recovery':{'previous':'first-generation','teachers':{}},'output':str(old)}
    atomic_json(old/'production.json',spec)
    atomic_json(old/'status.json',{'phase':'failed','error':"ValueError('teacher replay invalid; never use empty-demo fallback')"})
    started={'started_unix':time.time()-100};costs=[{'stage':'teacher','phase':'completed','charged_interactions':123,'charged_updates':0}]
    atomic_json(old/'started.json',started);atomic_json(old/'costs.json',costs)
    for name in ['panels.json','training_support.json','retention_support.json']:atomic_json(old/name,{'unchanged':True})
    journal=StageJournal(old/'stages',spec)
    corpus=old/'corpus.json';atomic_json(corpus,{'new_data':False,'admission':[],'groups':{'history':[],'teacher_new':[],'actor_new':[]}})
    dev=old/'dev.npz';dev.write_bytes(b'dev')
    bootstrap={'corpus':{'path':str(corpus),'sha256':file_sha(corpus),'new_data':False},'dev_fixture':str(dev),'dev_fixture_sha256':file_sha(dev)}
    checkpoint=old/'checkpoint';checkpoint.mkdir();(checkpoint/'state.msgpack').write_bytes(b'state')
    atomic_json(checkpoint/'manifest.json',{'state_sha256':file_sha(checkpoint/'state.msgpack')})
    generator={'checkpoint_manifest':str(checkpoint/'manifest.json'),'checkpoint_manifest_sha256':file_sha(checkpoint/'manifest.json')}
    source=old/'source.json';atomic_json(source,[{'root_id':'positive','label':1},{'root_id':'negative','label':0},{'root_id':'bad','label':0}])
    for name,result in [('semantic_smoke',{}),('bootstrap',bootstrap),('generator_pretrain',generator),('nominal',{}),('eval_teacher_source',{'path':str(source),'sha256':file_sha(source)})]:
        journal.stage(name,{},lambda r=result:r)
    teachers=old/'teachers';teachers.mkdir()
    atomic_json(teachers/'0000_result.json',{'root_id':'positive','teacher_status':'not_scheduled','source_recheck_label':1})
    proposal=teachers/'0001_proposals.npz';proposal.write_bytes(b'proposal')
    row={'root_id':'negative','source_actor_sha256':'actor','generator':generator,'teacher_status':'searched_no_solution','proposal_sha256':file_sha(proposal)}
    atomic_json(teachers/'0001_result.json',row)
    bad=teachers/'0002_result.json';atomic_json(bad,{**row,'root_id':'bad','teacher_status':'invalid'})
    # The invalid record is retained as evidence, not imported as a valid teacher.
    proposal_bad=teachers/'0002_proposals.npz';proposal_bad.write_bytes(b'proposal')
    result=prepare_recovery(old,output,tmp_path)
    assert set(result['recovery']['teachers'])=={'negative'}
    assert result['recovery']['rejected_teachers']['bad']['sha256']==file_sha(bad)
    assert json.loads((output/'started.json').read_text())==started
    assert json.loads((output/'costs.json').read_text())==costs
    assert (output/'training_support.json').read_bytes()==(old/'training_support.json').read_bytes()
    assert result['budgets']==spec['budgets']


@pytest.mark.parametrize('fault',['missing_spec_lock','wrong_output'])
def test_diagnostic_import_rejects_unbound_execution(tmp_path,fault):
    from jit_dvgc.generative_bridge.recovery import import_replay_diagnostic
    d=tmp_path/'diagnostic';d.mkdir()
    atomic_json(d/'spec.json',{})
    argv=['python','evaluate.py','--mode','evaluate','--spec',str(d/'spec.json'),'--output',str(d/'rollout')]
    if fault=='wrong_output':argv[-1]=str(d/'another-output')
    plan={'stages':[{'argv':argv}],'input_files':{} if fault=='missing_spec_lock' else {str(d/'spec.json'):file_sha(d/'spec.json')}}
    atomic_json(d/'plan.json',plan)
    atomic_json(d/'execution/status.json',{'phase':'completed','plan_sha256':file_sha(d/'plan.json'),
        'stages':[{'returncode':0,'argv':argv}]})
    with pytest.raises(ValueError,match='diagnostic (spec is not locked|executed a different command)'):
        import_replay_diagnostic(tmp_path/'old',d,{}, {},{}, {},[])
