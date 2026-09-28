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
