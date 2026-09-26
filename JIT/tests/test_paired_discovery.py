import copy
from pathlib import Path
import pytest
from jit_dvgc import paired_discovery as p
from jit_dvgc.jump_evidence_validation import read,write,file_sha


def data(name):
    points=[dict(candidate_id=str(i),state_sha256=str(i),phase='upstream',root_cell=str(i),full_cell=str(i),
        coordinates={'root_x_m':2.5+.1*i,'root_z_m':.3,'root_vz_mps':-.2}) for i in range(2)]
    labels={f'pi_{n}':[dict(candidate_id=x['candidate_id'],state_sha256=x['state_sha256'],phase=x['phase'],
        label=1,success_criterion='first_valid_landing',environment_interactions=2) for x in points] for n in range(5)}
    trails=[dict(trajectory_id=str(i),anchor_x_m=2.85,strength=.1,direction={'action_name':'hip','sign':1},
        environment_interactions=1,valid_landing=True,truncated=False,peak_root_z_m=.7) for i in range(32)]
    plan=dict(sources=dict.fromkeys(p.PHYSICS_SOURCES,'same'),proposer=name,members=[],bank_sha256='bank',physical_resolution={'fixed':True},
        frontier_profile={'round_index':0},horizon=400,label_seed=1,centerline='center',baseline='baseline')
    return dict(points=points,labels=labels,charged=52,catalog={'trajectory_receipts':trails,'environment_interactions':32},
                plan=plan,inputs={})


def test_cost_events_are_atomic_and_training_is_not_free():
    a,b=data('pi_2'),data('pi_4')
    b['points'][1].update(root_cell='2',full_cell='2')
    overhead,events=p.events(a)
    assert overhead==32 and events[0]['cost']==10
    metrics,curves,matched=p.summarize({'pi_2':a,'pi_4':b},{'0'},30,5)
    assert all(m['novel_root_cells']==1 and m['exclusive_novel_root_cells']==1 for m in metrics)
    assert curves['pi_4'][-1]['interactions']==52
    rows={(r['scenario'],r['proposer']):r for r in matched}
    assert rows['exploration_only','pi_4']['novel_root_cells']==1
    assert rows['plus_completed_probe_training','pi_4']['novel_root_cells']==0
    assert rows['plus_recorded_failed_smoke','pi_4']['training_surcharge']==35
    assert rows['plus_recorded_failed_smoke','pi_2']['training_surcharge']==0
    assert rows['plus_completed_probe_training','pi_2']['evaluator_count']==4
    assert rows['plus_completed_probe_training','pi_2']['common_budget']==48


@pytest.mark.parametrize('change',['labels','endpoint','cost','identity'])
def test_incomplete_or_invalid_evidence_rejected(change):
    d=data('pi_2')
    if change=='labels':d['labels']['pi_4'].pop()
    if change=='endpoint':d['labels']['pi_4'][0]['success_criterion']='recovery'
    if change=='cost':d['charged']=1
    if change=='identity':d['labels']['pi_4'][0]['state_sha256']='wrong'
    with pytest.raises(ValueError):p.events(d)


def test_identical_schedule_and_bank_required():
    a,b=data('pi_2'),data('pi_4');p.verify_pair(a,b)
    b['catalog']['trajectory_receipts'][0]['strength']=.2
    with pytest.raises(ValueError,match='schedule'):p.verify_pair(a,b)
    b=data('pi_4');b['plan']['bank_sha256']='other'
    with pytest.raises(ValueError,match='bank'):p.verify_pair(a,b)


def test_supervisor_reuses_pi4_and_runs_only_pi2(tmp_path,monkeypatch):
    repo=tmp_path;source=repo/'source';out=repo/'out';seed=repo/'seed'
    write(seed/'analysis_inputs.json',{})
    write(source/'seed_inputs.json',{str(seed/'analysis_inputs.json'):file_sha(seed/'analysis_inputs.json')})
    frozen=source/'frozen.json';write(frozen,{'test':True})
    done=source/'round_000/training_attempt_000/completion.json'
    write(done,{'policy':str(frozen),'charged_interactions':30,'artifacts':{str(frozen):file_sha(frozen)}})
    write(source/'round_000/discovery/request.json',{'budget':12000000})
    panels={n:data(n) for n in p.NAMES}
    for path in p.PHYSICS_SOURCES:
        f=repo/path;f.parent.mkdir(parents=True,exist_ok=True);f.write_text('# test physics')
        for d in panels.values():d['plan']['sources'][path]=file_sha(f)
    for d in panels.values():d['plan']['members']=[{'path':str(frozen),'policy':{'name':f'pi_{i}'}} for i in range(5)]
    monkeypatch.setattr(p,'panel',lambda child,recovery=False:panels['pi_4' if recovery else 'pi_2'])
    monkeypatch.setattr(p,'read_child',lambda child:([{'root_cell':'0','witnessed':True}],{},{}))
    calls=[]
    monkeypatch.setattr(p,'run_dense',lambda *a,**k:calls.append(k) or {'status':'completed'})
    monkeypatch.setattr(p,'render',lambda *a:None)
    monkeypatch.setattr(p,'bundle',lambda path:path/'results_to_send.zip')
    result=p.run(repo,out,source)
    assert result['status']=='completed',result
    assert len(calls)==1 and calls[0]['proposer']=='pi_2'
    assert calls[0]['budget']==12000000 and result['training_transitions']==0
    assert result['new_interactions']==52 and result['reused_pi4_discovery_interactions']==52
    assert result['known_failed_smoke_charge'] is None
    assert (out/'figures/matched_budget.csv').exists()


def test_cpu_entries_do_not_hide_visible_devices():
    root=Path(__file__).parents[1]/'cli'
    for name in ('analyze_envelope_campaign.py','run_envelope_campaign.py','compare_pi2_pi4_discovery.py'):
        source=(root/name).read_text()
        assert 'CUDA_VISIBLE_DEVICES' not in source
        assert 'JAX_PLATFORMS' in source and 'cpu' in source


def test_physics_drift_is_rejected_and_figures_export(tmp_path):
    a,b=data('pi_2'),data('pi_4')
    b['plan']['sources'][p.PHYSICS_SOURCES[0]]='changed'
    with pytest.raises(ValueError,match='implementation'):p.verify_pair(a,b)
    panels={'pi_2':a,'pi_4':data('pi_4')}
    _,curves,_=p.summarize(panels,{'0'},30,None)
    p.render(panels,curves,tmp_path)
    for ext in ('png','pdf','svg'):assert (tmp_path/f'paired_discovery.{ext}').stat().st_size>500


def test_no_new_ppo_view_does_not_get_free_pi4_witnesses():
    d=data('pi_2')
    for n in range(4):
        for row in d['labels'][f'pi_{n}']:row['label']=0
    overhead,ev=p.events(d,[f'pi_{n}' for n in range(4)])
    assert overhead==32 and sum(r['cost'] for r in ev)==16
    assert all(r['root_cell'] is None for r in ev)
    assert all(r['root_cell'] is not None for r in p.events(d)[1])
