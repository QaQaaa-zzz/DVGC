"""CPU orchestration contracts; these do not certify production GPU PPO."""
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import pytest
import jax
import jax.numpy as jp
from jit_dvgc import envelope_campaign as c
from jit_dvgc import iterative_probe_training as t
from jit_dvgc.jump_evidence_validation import read,write,file_sha


def rows(source='old',cell='a'):
    return [dict(key=f'{cell}-{p}',phase=p,trajectory_id=f'{source}-trajectory',
                 source=source,witnessed=True,root_cell=f'{cell}-{p}',snapshot='unused',
                 labels={'pi_0':1},coordinates={}) for p in ('upstream','downstream')]


def test_support_preserves_old_witnesses_excludes_failures_and_caps_groups():
    data=rows()+rows('recent','b')
    failed={**data[0],'key':'failure','witnessed':False}
    s=c.support_view(data+[failed],{},'recent')
    assert len(s['entries'])==4
    assert {r['sampling_weight'] for r in s['entries'] if r['source']=='recent'}=={2.}
    assert all(r['witnessed'] for r in s['entries'])
    assert not s['final_test_used']
    assert c.should_stop([20,4,0],2,5)
    assert not c.should_stop([4,10,0],2,5)
    assert not c.should_stop([0],2,5)


def test_fresh_clocks_keep_history_rng_and_events_under_jit():
    sample={'episode_step':jp.int32(29),'phase_episode_step':jp.int32(8),
            'episode_return':jp.float32(4),'events':{'episode_step':jp.int32(29),'apex_seen':jp.array(True)},
            'rng':jax.random.key(11),'observation_fifo':jp.arange(12).reshape(3,4),
            'down_events':{'airborne_seen':jp.array(True)},'qvel':jp.array([3.,-2.])}
    got=jax.jit(t.fresh_sample)(sample)
    assert int(got['episode_step'])==int(got['phase_episode_step'])==0
    assert bool(got['events']['apex_seen']) and bool(got['down_events']['airborne_seen'])
    assert jp.array_equal(got['observation_fifo'],sample['observation_fifo'])
    assert jp.array_equal(got['qvel'],sample['qvel'])
    assert jp.array_equal(jax.random.key_data(got['rng']),jax.random.key_data(sample['rng']))


def test_first_landing_changes_terminal_and_timeout_consistently():
    from flax import struct
    @struct.dataclass
    class State:
        done:object
        info:object
        metrics:object
    @struct.dataclass
    class Events:
        valid_contact_seen:object
    state=State(jp.float32(0),{'down_events':Events(jp.array(True)),'physical_failure':jp.array(False),
        'truncated':jp.array(True),'timeout':jp.array(True),'terminated':jp.array(False),'end_code':jp.int32(9)},
        {'terminal/descent_timeout':jp.float32(1)})
    got=jax.jit(t.first_landing_state)(state)
    assert bool(got.done) and bool(got.info['success']) and bool(got.info['terminated'])
    assert not bool(got.info['truncated']) and int(got.info['end_code'])==13
    assert float(got.metrics['terminal/success'])==1 and float(got.info['time_out'])==0
    failed=state.replace(info={**state.info,'physical_failure':jp.array(True)})
    assert not bool(t.first_landing_state(failed).info['success'])


@pytest.fixture
def campaign(tmp_path,monkeypatch):
    repo=tmp_path/'repo';(repo/'JIT').mkdir(parents=True)
    base=read(Path(__file__).parents[1]/'configs/pi_unified_formal.json')
    bootstrap=repo/'bootstrap.json';write(bootstrap,base)
    frozen=repo/'pi_2.json';write(frozen,{'mock':True})
    sentinel=repo/'evidence.json';write(sentinel,{'TRAIN':True})
    members=[{'path':str(frozen),'policy':{'name':f'pi_{i}','formal_config':str(bootstrap)}} for i in range(4)]
    plan={'baseline':str(repo/'baseline'),'members':members}
    discoveries=[];train_calls=[]
    def child(path):
        path=Path(path)
        data=rows(str(path), 'seed' if path.name!='discovery' else path.parent.name)
        return data,{str(sentinel):file_sha(sentinel)},plan
    monkeypatch.setattr(c,'read_child',child)
    monkeypatch.setattr(c,'bundle',lambda root:Path(root)/'results_to_send.zip')
    monkeypatch.setattr(c,'render_progress',lambda *args:None)
    from jit_dvgc import dense_tube,unified_policy_freeze
    def explore(repo,out,**kwargs):
        c.validate_profile(kwargs['profile']);discoveries.append(kwargs)
        write(out/'request.json',{'budget':kwargs['budget']})
        write(out/'cost_ledger.json',{'charged_interactions':100})
        return {'status':'completed'}
    monkeypatch.setattr(dense_tube,'run',explore)
    class Process:
        def __init__(self,command,**kwargs):
            config=read(command[command.index('--config')+1]);train_calls.append(config)
            assert kwargs['env']['JAX_PLATFORMS']=='cuda,cpu'
            assert kwargs['env']['CUDA_VISIBLE_DEVICES']=='0'
            root=Path(kwargs['env']['JIT_RUN_ROOT'])/config['run_declaration']['run_id']
            steps=config['ppo']['requested_transitions']
            write(root/'formal_report.json',{'status':'completed','completed_training_transitions':steps,'train_panel_interactions':20})
            write(root/'checkpoints'/f'transition_{steps}'/'identity.json',{'mock':True})
            (root/'checkpoints'/f'transition_{steps}'/'payload.pkl').write_bytes(b'test-only')
        def wait(self,timeout=None):return 0
        def poll(self):return 0
    monkeypatch.setattr(c.subprocess,'Popen',Process)
    def freeze(out,**kwargs):write(out/'frozen_unified_policy.json',{'policy':f"pi_{kwargs['iteration']}"})
    monkeypatch.setattr(unified_policy_freeze,'freeze_unified_policy',freeze)
    return repo,repo/'campaign',train_calls,discoveries


def test_two_round_loop_bank_growth_new_support_and_cache(campaign):
    repo,out,trains,discoveries=campaign
    result=c.run(repo,out,max_rounds=2,ppo_steps=3200)
    assert result['status']=='empirical_stagnation',result
    assert len(trains)==len(discoveries)==2
    assert [r['bank_size'] for r in result['rounds']]==[5,6]
    assert result['charged_interactions']==2*(3200+20+100)
    assert 'round_000' in trains[1]['initialization']['source_frozen_policy']
    assert any(r['source'].endswith('round_000/discovery') for r in read(out/'round_001/support.json')['entries'])
    assert not result['physical_boundary_proven']
    again=c.run(repo,out,max_rounds=2,ppo_steps=3200)
    assert again==result and len(trains)==2
    assert c.run(repo,out,max_rounds=3,ppo_steps=3200)['status']=='engineering_error'


def test_no_training_when_budget_cannot_cover_reservation(campaign):
    repo,out,trains,_=campaign
    result=c.run(repo,out,budget=4000,ppo_steps=3200)
    assert result['status']=='budget_exhausted' and not trains
    assert result['charged_interactions']==0


def test_training_failure_charged_then_fresh_retry(campaign,monkeypatch):
    repo,out,trains,_=campaign
    original=c.subprocess.Popen
    class Failure:
        def __init__(self,*a,**k):pass
        def wait(self,timeout=None):return 9
        def poll(self):return 9
    monkeypatch.setattr(c.subprocess,'Popen',Failure)
    result=c.run(repo,out,max_rounds=1,ppo_steps=3200)
    assert result['status']=='engineering_error'
    assert result['charged_interactions']==4800
    monkeypatch.setattr(c.subprocess,'Popen',original)
    result=c.run(repo,out,max_rounds=1,ppo_steps=3200)
    assert result['status']=='round_limit_reached',result
    assert result['charged_interactions']==4800+3200+20+100
    assert (out/'round_000/training_attempt_000/exit.json').exists()
    assert (out/'round_000/training_attempt_001/completion.json').exists()


def test_training_config_drift_and_unwitnessed_support_rejected(campaign):
    repo,out,_,_=campaign
    result=c.run(repo,out,max_rounds=1,ppo_steps=3200)
    assert result['status']=='round_limit_reached',result
    p=out/'round_000/training_attempt_000/config.json'
    config=t.load_config(p)
    assert config.ppo.requested_transitions==3200 and config.ppo.num_parallel_envs==128
    from jit_dvgc.unified_formal import load_unified_policy_formal_config
    assert load_unified_policy_formal_config(p).schema==t.SCHEMA
    support=Path(config.raw['support']);data=read(support);data['entries'][0]['witnessed']=False
    write(support,data)
    with pytest.raises(ValueError,match='input changed'):t.load_config(p)


def test_progress_exports_observed_points_and_cost_table(tmp_path):
    data=rows()
    for r in data:r['coordinates']={'root_x_m':2.8,'root_z_m':.4}
    records=[dict(round=0,policy='pi_4',novel_root_cells=2,campaign_union_root_cells=4,
                  charged_interactions=5000,bank_size=5,candidate_count=2)]
    c.render_progress(tmp_path,data,records,2)
    assert '5000' in (tmp_path/'figures/coverage_cost.csv').read_text()
    for ext in ('png','pdf','svg'):
        assert (tmp_path/'figures'/f'campaign_progress.{ext}').stat().st_size>500


def test_growing_bank_supports_five_policy_plot(tmp_path):
    from jit_dvgc.analysis.policy_envelopes import summarize,render_comparison
    from jit_dvgc.analysis.policy_envelopes import project_row
    from jit_dvgc.analysis.capability_tube import FULL_PHYSICAL_FIELDS
    points=[]
    for i,phase in enumerate(('upstream','downstream')):
        row=dict(candidate_id=str(i),state_sha256=str(i),phase=phase,parent_group_id='g')
        coords=dict.fromkeys(FULL_PHYSICAL_FIELDS,0.)
        coords.update(root_x_m=2.5+i*.1,root_z_m=.3+i*.1)
        points.append(project_row(row,coords,str(i)))
    labels={f'pi_{i}':[dict(candidate_id=p['candidate_id'],state_sha256=p['state_sha256'],phase=p['phase'],
        label=1,success_criterion='first_valid_landing',environment_interactions=2) for p in points] for i in range(5)}
    result=summarize(points,labels,role='train')
    render_comparison(points,result,tmp_path,title_prefix='SYNTHETIC ONLY | ')
    assert (tmp_path/'pi_4_projections.png').stat().st_size>500
    assert len(result['metrics'])==6


def test_child_publication_is_suppressed_inside_campaign(tmp_path,monkeypatch):
    from jit_dvgc import result_publishing as p
    monkeypatch.setattr(p,'__file__',str(tmp_path/'repo/JIT/src/jit_dvgc/result_publishing.py'))
    root=tmp_path/'repo/JIT/runs/campaign/test';root.mkdir(parents=True)
    (root/'execution.lock').touch();write(root/'seed_inputs.json',{})
    called=[];monkeypatch.setattr(p,'publish_safely',lambda repo,out:called.append(out))
    p.auto_publish(root/'round_000/discovery');assert not called
    p.auto_publish(root);assert called==[root]


def test_production_training_wrapper_accepts_new_support_preflight(campaign,monkeypatch):
    repo,out,_,_=campaign
    assert c.run(repo,out,max_rounds=1,ppo_steps=3200)['status']=='round_limit_reached'
    config=out/'round_000/training_attempt_000/config.json'
    from jit_dvgc.training import formal
    monkeypatch.setattr(formal,'_tube_points',lambda artifact:[{} for _ in artifact.entries])
    called=[]
    monkeypatch.setattr(formal,'_run_unified_formal',lambda *a,**k:called.append(a) or {'status':'mock_trained'})
    result=formal.preflight_unified_formal_tube(config)
    assert result['entry_count']==4 and result['training_transitions']==0
    assert not result['production_training_smoke_verified']
    assert formal.run_unified_formal(config,'test')['status']=='mock_trained'
    assert len(called)==1


def test_all_proposers_reuse_then_grow_and_resume(campaign,monkeypatch):
    from jit_dvgc import campaign_bank as bank
    repo,out,trains,discoveries=campaign
    monkeypatch.setattr(bank,'inherited_bank',lambda repo:(rows('inherited','i'),{},[str(repo/'pi_2.json')],195_551))
    exports=[];monkeypatch.setattr(bank,'export_round',lambda *args:exports.append(args))
    result=c.run(repo,out,max_rounds=2,ppo_steps=3200,all_proposers=True)
    assert result['status'] in ('round_limit_reached','empirical_stagnation'),result
    assert [d['proposer'] for d in discoveries]==['pi_0','pi_1','pi_3','pi_5',
        'pi_0','pi_1','pi_2','pi_3','pi_4','pi_5','pi_6']
    assert len(trains)==2 and len(exports)==4
    assert result['completed_training_transitions']==6400
    assert result['charged_interactions']==2*3220+1100
    assert result['inherited_recorded_charge']==195_551
    assert result['total_including_inherited_recorded_charge']==203_091
    assert len(result['frozen_new_policies'])==2
    assert c.run(repo,out,max_rounds=2,ppo_steps=3200,all_proposers=True)==result
    assert len(trains)==2


def test_all_proposer_budget_stops_before_training(campaign,monkeypatch):
    from jit_dvgc import campaign_bank as bank
    repo,out,trains,discoveries=campaign
    monkeypatch.setattr(bank,'inherited_bank',lambda repo:(rows('inherited','i'),{},[str(repo/'pi_2.json')],100))
    result=c.run(repo,out,max_rounds=1,budget=8000,ppo_steps=3200,all_proposers=True)
    assert result['status']=='budget_exhausted',result
    assert not trains and not discoveries


def test_round_exports_replot_data_and_untested_labels(tmp_path):
    from jit_dvgc.campaign_bank import export_round
    a=tmp_path/'pi_2';b=tmp_path/'pi_4'
    data=[]
    for path,name,label in ((a,'pi_2','pi_0'),(b,'pi_4','pi_4')):
        write(path/'plan.json',dict(proposer=name,plan_sha256='plan',bank_sha256='bank',physical_resolution={'x':.1}))
        write(path/'analysis_inputs.json',{'catalog':str(path/'catalog.json')})
        write(path/'catalog.json',{'trajectory_receipts':[]})
        rs=rows(str(path),name)
        for r in rs:r.update(coordinates={'root_x_m':2.85,'root_z_m':.5,'root_vz_mps':1.},labels={label:1})
        data+=rs
    export_round(tmp_path/'round',data,set(),120,30)
    import csv
    points=list(csv.DictReader((tmp_path/'round/figures/all_points.csv').open()))
    assert points[0]['evaluator_pi_4']==''
    assert points[-1]['evaluator_pi_0']==''
    assert len(points)==4
    for name in ('pi_2_envelope','pi_4_envelope','all_proposers'):
        for ext in ('png','pdf','svg'):assert (tmp_path/f'round/figures/{name}.{ext}').stat().st_size>500
    manifest=read(tmp_path/'round/figures/figure_manifest.json')
    assert manifest['slice_width_m']==.05 and not manifest['hulls_filled']
    export_round(tmp_path/'round',data,set(),9999,30)
    assert read(tmp_path/'round/figures/figure_manifest.json')['charged_new_interactions']==120


def test_all_proposer_failed_labels_never_train(campaign,monkeypatch):
    from jit_dvgc import campaign_bank as bank,dense_tube
    repo,out,trains,_=campaign
    monkeypatch.setattr(bank,'inherited_bank',lambda repo:(rows('inherited','i'),{},[str(repo/'pi_2.json')],100))
    def fail(repo,out,**kwargs):
        write(out/'cost_ledger.json',{'charged_interactions':400})
        return {'status':'engineering_error'}
    monkeypatch.setattr(dense_tube,'run',fail)
    result=c.run(repo,out,max_rounds=1,ppo_steps=3200,all_proposers=True)
    assert result['status']=='engineering_error' and result['charged_interactions']==400
    assert not trains
