from pathlib import Path
from types import SimpleNamespace
import copy
import json
import numpy as np
import jax.numpy as jp
import pytest
from jit_dvgc.dense_tube import choose_backend
from jit_dvgc.jump_evidence_validation import read,write


def test_backend_selection_requires_matching_endpoints_before_speed():
    serial={'identities':{'policy':'a'},'seconds':10.,'labels':[{'candidate_id':'x','label':1,'environment_interactions':10}]}
    device=copy.deepcopy(serial);device['seconds']=2.
    assert choose_backend(serial,device)['backend']=='device'
    device['labels'][0]['environment_interactions']=11
    assert choose_backend(serial,device)['backend']=='serial'
    assert choose_backend(serial,None)['backend']=='serial'
    device=copy.deepcopy(serial);device['seconds']=20.
    assert choose_backend(serial,device)['backend']=='serial'


@pytest.mark.parametrize('extended,cap,expected_steps,reason',[
    (False,32,9,'task_terminated_before_landing'),
    (True,32,12,'first_valid_landing'),
    (True,2,4,'candidate_cap'),
])
def test_dense_collector_saves_multiple_real_frames_and_exact_action_prefixes(tmp_path,monkeypatch,extended,cap,expected_steps,reason):
    from jit_dvgc.acquisition import causal_jump as c
    import jit_dvgc.frontier_label_shard_runner as runner
    def state(tick):
        x=2.5+.06*tick
        return SimpleNamespace(data=SimpleNamespace(qpos=jp.array([x,0.,.2]),qvel=jp.array([2.,0.,-1. if tick>=5 else 1.])),
            obs={'state':jp.array([float(tick)])},done=jp.array(tick>=(12 if extended else 9)),
            info={'reset_from_soft_tube':jp.array(False),'reset_from_jump_start':jp.array(True),'expert_switching_used':jp.array(False),
                  'active_phase':jp.array(int(tick>=5)), 'up_events':SimpleNamespace(apex_seen=jp.array(tick>=5)),
                  'down_events':SimpleNamespace(valid_contact_seen=jp.array(extended and tick>=12))})
    def step(s,a):return state(int(s.obs['state'][0])+1)
    artifact=SimpleNamespace(entries=[],manifest={'manifest_sha256':'b'*64})
    env=SimpleNamespace(_bundle=SimpleNamespace(xml_sha256='x'),tube_pool=SimpleNamespace(artifact=artifact),
                        _reset_jump_start_unified=lambda key:state(0),step=step)
    start=c.physical_state_sha256_from_state(state(0))
    center={'jump_start_state_sha256':start,'centerline_sha256':'c'*64,'x_min_m':2.5,'effective_centerline_max_x_m':3.1}
    monkeypatch.setattr(c,'load_nominal_jump_centerline',lambda p:center)
    monkeypatch.setattr(c.jax,'jit',lambda f:f)
    monkeypatch.setattr(c.jax,'block_until_ready',lambda x:x)
    monkeypatch.setattr(runner,'_build_memory_stable_step',lambda e:e.step)
    def capture(s,**kwargs):
        t=int(s.obs['state'][0]);return SimpleNamespace(qpos=np.asarray(s.data.qpos),qvel=np.asarray(s.data.qvel),episode_step=t,phase_episode_step=t)
    monkeypatch.setattr(c,'capture_unified_envelope_snapshot',capture)
    monkeypatch.setattr(c,'snapshot_context_sha256',lambda s:f'{s.episode_step:064x}')
    monkeypatch.setattr(c,'physical_state_sha256',lambda s:f'{s.episode_step+100:064x}')
    def save(p,s):write(p/'mock.json',{'step':s.episode_step})
    monkeypatch.setattr(c,'save_unified_envelope_snapshot',save)
    policy={'iteration':0,'name':'pi_0','policy_role':'envelope_expansion_authority','xml_sha256':'x',
            'actor_sha256':'a'*64,'payload_sha256':'b'*64,'formal_config_sha256':'c'*64}
    result=c.collect_jump_start_connected_candidates([{'phase':'upstream','x_target_m':2.8,'proposal_family_index':0,
        'state_sha256':start,'parent_group_id':'train_trajectory'}],tmp_path/'dense',env=env,policy=lambda obs,key:jp.zeros(4),
        policy_record=policy,frozen_manifest_sha256='d'*64,nominal_centerline=tmp_path/'center.json',protocol_seed=1,
        strengths=[.1],action_names=['hip'],signs=[1],lookbacks_m=[.2],max_forward_ticks=20,
        evidence_mode='probe_bank_arrivals_v1',probe_bank_sha256='e'*64,logical_role='train',start_contract_sha256='f'*64,
        sampling_mode='trajectory_slices_v2',slice_spacing_m=.05,max_candidates_per_attempt=cap,sampling_max_x_m=8. if extended else None)
    rows=result['entries']
    if cap>2: assert len(rows)>3 and {r['phase'] for r in rows}=={'upstream','downstream'}
    assert result['environment_interactions']==expected_steps
    receipt=result['trajectory_receipts'][0]
    assert receipt['stop_reason']==reason
    assert receipt['environment_interactions']==expected_steps
    assert receipt['truncated']==(reason=='candidate_cap')
    if extended and cap>2:
        assert max(r['x'] for r in rows)>center['effective_centerline_max_x_m']
        assert receipt['valid_landing']
        assert read(tmp_path/'dense/protocol.json')['sampling_max_x_m']==8.
    for r in rows:
        t=r['trajectory_step'];prov=r['jump_start_reachability']
        assert len(r['perturbation']['nominal_actions'])==t
        assert len(r['perturbation']['effective_deltas'])==t
        assert prov['environment_transitions_from_jump_start']==t
        assert abs(r['x']-r['x_target_m'])<=.025001
        assert r['episode_step']==t
        assert prov['qpos_qvel_injection_used'] is False
    if cap>2: assert rows[-1]['jump_start_reachability']['unperturbed_suffix_transitions']>0
    assert len({r['trajectory_id'] for r in rows})==1
    assert read(tmp_path/'dense/protocol.json')['target_x_half_width_m']==.025


def test_coverage_delta_does_not_credit_rebinning_or_duplicate_points():
    from jit_dvgc.analysis.dense_coverage import coverage_rows
    a=[{'root_cell':'a','full_cell':'a'}]
    b=[{'root_cell':'a','full_cell':'a'},{'root_cell':'b','full_cell':'b'}]
    old={'policy_names':['pi_0'],'resolution':{'x':.1},'masks':{'pi_0':[True],'union':[True]}}
    new={'policy_names':['pi_0'],'resolution':{'x':.1},'masks':{'pi_0':[True,True],'union':[True,True]}}
    rows=coverage_rows(a,old,b,new)
    assert rows[0]['root_cell_novel']==1 and rows[0]['root_cell_cumulative']==2
    new['resolution']={'x':.05}
    with pytest.raises(ValueError,match='resolution'):coverage_rows(a,old,b,new)


def test_failure_packages_and_reuse_refuses_artifact_changes(tmp_path,monkeypatch):
    from jit_dvgc import dense_tube as d
    class Failed:
        def __init__(self,*a,**k):pass
        def wait(self,timeout=None):return 2
        def poll(self):return 2
    monkeypatch.setattr(d.subprocess,'Popen',Failed)
    assert d.run(tmp_path,tmp_path/'run')['status']=='engineering_error'
    assert (tmp_path/'run/results_to_send.zip').exists()


@pytest.mark.parametrize('serial_only',[False,True])
def test_supervisor_falls_back_preserves_cost_and_resumes(tmp_path,monkeypatch,serial_only):
    from jit_dvgc import dense_tube as d
    from jit_dvgc.evidence_integrity import canonical_sha256
    calls=[];output=tmp_path/'run'
    class Process:
        def __init__(self,command,**kwargs):
            get=lambda flag:command[command.index(flag)+1]
            kind=get('--worker');dest=Path(get('--destination'));calls.append(kind);self.code=0
            if kind=='prepare':
                plan={'repo':str(tmp_path),'request':read(output/'request.json'),'sources':{},'input_files':{},
                      'horizon':400,'names':['pi_0','pi_1','pi_2','pi_3']}
                plan['plan_sha256']=canonical_sha256(plan);write(output/'plan.json',plan)
                result={'environment_interactions':0}
            elif kind=='benchmark':
                if get('--policy')=='pi_1' and get('--backend')=='device':self.code=2;return
                result={'environment_interactions':3,'identities':{'policy':get('--policy')},
                        'seconds':1. if get('--backend')=='device' else 2.,
                        'labels':[{'candidate_id':'a','label':1,'environment_interactions':3}]}
            elif kind=='acquire':
                write(dest/'result/catalog.json',{'candidate_count':3});result={'environment_interactions':5}
            elif kind=='project':
                write(dest/'projected.json',[]);result={'environment_interactions':0,'candidate_count':3}
            elif kind=='label':result={'environment_interactions':7}
            elif kind=='merge':result={'environment_interactions':0}
            else:
                write(output/'figures/figure_metadata.json',{'ok':True})
                result={'environment_interactions':0,'figures':str(output/'figures')}
            write(dest/'worker_report.json',{'status':'completed',**result})
        def wait(self,timeout=None):return self.code
        def poll(self):return self.code
    monkeypatch.setattr(d.subprocess,'Popen',Process)
    profile=None
    if serial_only:
        from jit_dvgc.frontier_exploration import allocate
        old={'status':'completed','scope':'TRAIN','final_test_used':False,
             'metrics':[{'proposer':n,'charged_interactions':10,'novel_vs_previous_train_union':1} for n in ('pi_0','pi_1','pi_2','pi_3')]}
        old['report_sha256']=canonical_sha256(old)
        profile=allocate(old)['profiles']['pi_0']
    result=d.run(tmp_path,output,profile=profile)
    assert result['status']=='completed'
    assert result['backend_decisions']['pi_1']['backend']=='serial'
    assert result['backend_decisions']['pi_0']['backend']==('serial' if serial_only else 'device')
    assert result['charged_interactions']==(5+4*7 if serial_only else 7*3+6400+5+4*7)
    if serial_only: assert 'benchmark' not in calls
    count=len(calls)
    assert d.run(tmp_path,output,profile=profile)['status']=='completed' and len(calls)==count
    write(output/'figures/figure_metadata.json',{'changed':True})
    assert d.run(tmp_path,output,profile=profile)['status']=='engineering_error'
    assert len(calls)==count
