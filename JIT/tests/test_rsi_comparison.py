import numpy as np
from jit_dvgc.rsi_comparison import episode_results


def test_early_failure_and_timeout_remain_in_comparison_denominator():
    shape = (4, 3)
    tape = {k: np.zeros(shape, bool) for k in ('prefix_mask', 'success', 'physical_failure', 'terminal', 'mask')}
    tape['prefix_mask'][:, 0] = True
    tape['prefix_mask'][:2, 1] = True
    tape['prefix_mask'][:, 2] = True
    tape['success'][3, 0] = True; tape['terminal'][3, 0] = True
    tape['physical_failure'][1, 1] = True; tape['terminal'][1, 1] = True
    tape['time'] = np.broadcast_to(np.arange(1, 5)[:, None] * .02, shape)
    tape['qpos'] = np.zeros((4, 3, 3))
    rows = episode_results(tape, 0)
    assert len(rows) == 3 and sum(r['success'] for r in rows) == 1
    assert rows[1]['control_steps'] == 2 and rows[1]['pulse_applied_steps'] == 0
    assert rows[2]['horizon_exhausted'] is True


def test_conflicting_success_and_failure_is_not_success():
    tape = {k: np.ones((1, 1), bool) for k in ('prefix_mask', 'success', 'physical_failure', 'terminal', 'mask')}
    tape.update(time=np.array([[.02]]), qpos=np.zeros((1, 1, 3)))
    row = episode_results(tape, 0)[0]
    assert row['conflict'] and not row['success']


def test_environment_timeout_is_counted_separately_from_rollout_horizon():
    from jit_dvgc.constants import END_TIMEOUT
    tape = {k: np.zeros((1, 1), bool) for k in ('success', 'physical_failure', 'mask')}
    tape.update(prefix_mask=np.ones((1, 1), bool), terminal=np.ones((1, 1), bool),
                time=np.array([[8.]]), qpos=np.zeros((1, 1, 3)), end_code=np.array([[END_TIMEOUT]]))
    row = episode_results(tape, 0)[0]
    assert row['environment_timeout'] and not row['horizon_exhausted'] and not row['success']


def test_report_retains_baseline_counts_and_all_failed_episodes(tmp_path):
    from jit_dvgc.constants import END_TIMEOUT
    from jit_dvgc.rsi_comparison import report
    from jit_dvgc.jump_evidence_validation import write, read
    write(tmp_path/'spec.json',dict(root_qpos_address=0,episodes=2,baseline='initial',training_steps=3200,role='development'))
    for method in ('baseline','fresh_rsi'):
        for condition,n in [('nominal',1),('random',2)]:
            p=tmp_path/'evaluation'/f'{method}_{condition}';p.mkdir(parents=True)
            tape={k:np.zeros((2,n),bool) for k in ('success','physical_failure','terminal','mask')}
            tape.update(prefix_mask=np.ones((2,n),bool),end_code=np.full((2,n),END_TIMEOUT),
                        qpos=np.zeros((2,n,3)),time=np.full((2,n),.02),
                        front_wheel_clearance=np.zeros((2,n)),rear_wheel_clearance=np.zeros((2,n)))
            for k in ('delta','requested_delta','effective_delta'):tape[k]=np.zeros((2,n,4))
            tape['terminal'][-1]=True
            np.savez_compressed(p/'prefixes.npz',**tape)
            write(p/'status.json',dict(charged_interactions=2*n))
    summary=report(tmp_path)
    assert summary['baseline']['successes']==0 and summary['baseline']['episodes']==2
    assert summary['baseline']['environment_timeouts']==2
    assert summary['baseline_policy']=='initial'
    assert summary['aligned_exclusions']=={'baseline':2,'fresh_rsi':2}
    assert read(tmp_path/'comparison/summary.json')['baseline']['episodes']==2


def test_report_compares_three_methods_and_rejects_unpaired_draws(tmp_path):
    import pytest
    from jit_dvgc.rsi_comparison import report
    from jit_dvgc.jump_evidence_validation import write
    methods=[dict(key=m,label=m,evaluation_dir=str(tmp_path/'evaluation')) for m in ('baseline','fresh_rsi','phase_u')]
    write(tmp_path/'spec.json',dict(root_qpos_address=0,episodes=2,baseline='initial',training_steps=8000000,role='development',comparison_methods=methods))
    for method in methods:
        for condition,n in [('nominal',1),('random',2)]:
            p=tmp_path/'evaluation'/f'{method["key"]}_{condition}';p.mkdir(parents=True)
            tape={k:np.zeros((2,n),bool) for k in ('success','physical_failure','mask')}
            tape.update(prefix_mask=np.ones((2,n),bool),terminal=np.ones((2,n),bool),end_code=np.zeros((2,n),int),
                        qpos=np.zeros((2,n,3)),time=np.full((2,n),.02),
                        front_wheel_clearance=np.zeros((2,n)),rear_wheel_clearance=np.zeros((2,n)))
            for k in ('delta','requested_delta','effective_delta'):tape[k]=np.zeros((2,n,4))
            if method['key']=='phase_u':tape['success'][-1,0]=True
            np.savez_compressed(p/'prefixes.npz',**tape)
            write(p/'status.json',dict(charged_interactions=2*n))
    summary=report(tmp_path)
    assert summary['phase_u']['successes']==1
    assert summary['evaluation_charged_interactions']==18
    p=tmp_path/'evaluation/phase_u_random/prefixes.npz'
    tape=dict(np.load(p));tape['delta'][0,0,0]=1
    np.savez_compressed(p,**tape)
    with pytest.raises(AssertionError):report(tmp_path,tmp_path/'unpaired')


def test_phase_policy_override_forbidden_for_learning():
    import pytest
    from jit_dvgc.rsi_comparison import load_phase_evaluation_policy
    with pytest.raises(ValueError,match='fixed_random'):
        load_phase_evaluation_policy({'controller_mode':'learned_residual'},None,None)


def test_phase_policy_preserves_payload_and_rejects_runtime_drift(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import pytest
    from jit_dvgc import rsi_comparison as comparison, iterative_probe_training
    from jit_dvgc.handoff_bank import pytree_sha256
    from jit_dvgc.jump_evidence_validation import write,file_sha
    config_path=tmp_path/'phase.json';write(config_path,{'model':'fixture'})
    checkpoint=tmp_path/'checkpoint';checkpoint.mkdir();(checkpoint/'payload.pkl').write_bytes(b'fixture')
    (checkpoint/'identity.json').write_text('{}')
    payload=SimpleNamespace(identity=SimpleNamespace(xml_sha256='xml',actor_frame_fields=('a',),actor_task_fields=('b',),action_order=('c',)),training_transitions=4988928,
        actor_params={'w':np.array([3.])},critic_params={'w':np.array([4.])},observation_normalizer={'mean':np.array([5.])})
    monkeypatch.setattr(iterative_probe_training,'load_phase_initializer',lambda _:payload)
    phase=dict(name='phase_u',source_phase_config=str(config_path),source_checkpoint=str(checkpoint),
               input_files={str(config_path):file_sha(config_path)})
    spec=dict(controller_mode='fixed_random',full_episode_rollout=True,phase_policy=phase)
    config=SimpleNamespace(up_config_sha256=comparison.canonical_sha256({'model':'fixture'}))
    member={'name':'baseline','policy':{'xml_sha256':'xml'}}
    restored,record=comparison.load_phase_evaluation_policy(spec,config,member)
    assert restored is payload
    assert record['policy']['normalizer_sha256']==pytree_sha256(payload.observation_normalizer)
    assert member['name']=='baseline' and 'actor_sha256' not in member['policy']
    config.up_config_sha256='wrong'
    with pytest.raises(ValueError,match='runtime differs'):comparison.load_phase_evaluation_policy(spec,config,member)
    config_path.write_text('{}')
    with pytest.raises(ValueError,match='input drift'):comparison.load_phase_evaluation_policy(spec,config,member)


def test_zero_amplitude_nominal_window_is_not_an_applied_disturbance():
    tape={k:np.zeros((1,1),bool) for k in ('success','physical_failure')}
    tape.update(prefix_mask=np.ones((1,1),bool),terminal=np.ones((1,1),bool),mask=np.ones((1,1),bool),
                time=np.array([[.02]]),qpos=np.zeros((1,1,3)),effective_delta=np.zeros((1,1,4)))
    row=episode_results(tape,0)[0]
    assert row['pulse_applied_steps']==0
    assert row['pulse_window_steps']==1


def test_descendant_runtime_compatibility_allows_only_declared_training_changes(tmp_path):
    import copy
    import pytest
    from jit_dvgc.rsi_comparison import validate_phase_runtime_compatibility
    historical={'schema':'jit_phase_u_formal_v4','phase':'propulsion_ascent',
                **{key:{'v':1} for key in ('model','action','reset','events','physical_limits','reward','training_wrapper')},
                'ppo':{'seed':1,'requested_transitions':24576,'num_evals':2,'episode_horizon':400},
                'formal':{'checkpoint_transitions':[0,24576],'fixed_evaluation_transitions':[24576],'resume_semantics':'fresh_only'},
                'action_order':['a'],'actor_frame_fields':['b'],'actor_task_fields':['c']}
    descendant=copy.deepcopy(historical)
    descendant['ppo'].update(seed=2,requested_transitions=49152,num_evals=3)
    descendant['formal'].update(checkpoint_transitions=[0,49152],fixed_evaluation_transitions=[49152],resume_semantics='parameter_warm_start_optimizer_reset')
    descendant['initialization']={'actor':'warm_start_frozen_development'}
    descendant['training_reference']={'resolved_config':'unused','sha256':'unused'}
    descendant['run_declaration']={'run_id':'descendant'}
    validate_phase_runtime_compatibility(descendant,historical)
    for field in ('model','action','reset','events','physical_limits','reward','training_wrapper','action_order','actor_frame_fields','actor_task_fields','phase'):
        changed=copy.deepcopy(descendant);changed[field]='changed'
        with pytest.raises(ValueError,match='runtime'):validate_phase_runtime_compatibility(changed,historical)
    for field in ('episode_horizon','held_out_seeds'):
        changed=copy.deepcopy(descendant);changed['ppo'][field]='changed'
        with pytest.raises(ValueError,match='runtime'):validate_phase_runtime_compatibility(changed,historical)
    changed=copy.deepcopy(descendant);changed['unexpected_runtime_knob']=1
    with pytest.raises(ValueError,match='runtime'):validate_phase_runtime_compatibility(changed,historical)


def test_phase_descendant_loader_checks_locked_historical_contract_and_keeps_identity(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import pytest
    from jit_dvgc import rsi_comparison as comparison, iterative_probe_training
    from jit_dvgc.jump_evidence_validation import write,file_sha
    historical={'schema':'jit_phase_u_formal_v4','phase':'propulsion_ascent',
                **{key:{'v':1} for key in ('model','action','reset','events','physical_limits','reward','training_wrapper')},
                'ppo':{'seed':1},'formal':{}}
    descendant={**historical,'ppo':{'seed':2}}
    hp=tmp_path/'historical.json';cp=tmp_path/'descendant.json';write(hp,historical);write(cp,descendant)
    checkpoint=tmp_path/'checkpoint';checkpoint.mkdir();(checkpoint/'payload.pkl').write_bytes(b'fixture');write(checkpoint/'identity.json',{})
    payload=SimpleNamespace(identity=SimpleNamespace(xml_sha256='xml',actor_frame_fields=('a',),actor_task_fields=('b',),action_order=('c',)),training_transitions=14991360,
        actor_params={'w':np.array([3.])},critic_params={'w':np.array([4.])},observation_normalizer={'mean':np.array([5.])})
    monkeypatch.setattr(iterative_probe_training,'load_phase_initializer',lambda _:payload)
    phase=dict(name='descendant',source_phase_config=str(cp),source_checkpoint=str(checkpoint),
        runtime_compatibility=dict(schema='jit_phase_u_runtime_compatibility_v1',historical_config=str(hp)),
        input_files={str(p):file_sha(p) for p in (hp,cp,checkpoint/'identity.json',checkpoint/'payload.pkl')})
    spec=dict(controller_mode='fixed_random',full_episode_rollout=True,phase_policy=phase)
    config=SimpleNamespace(up_config_sha256=comparison.canonical_sha256(historical),up_config_path=str(hp))
    member={'name':'baseline','policy':{'xml_sha256':'xml','actor_frame_fields':['a'],'actor_task_fields':['b'],'action_order':['c']}}
    restored,record=comparison.load_phase_evaluation_policy(spec,config,member)
    assert restored is payload
    assert record['policy']['source_config_sha256']==comparison.canonical_sha256(descendant)
    descendant['reward']={'changed':True};write(cp,descendant);phase['input_files'][str(cp)]=file_sha(cp)
    with pytest.raises(ValueError,match='runtime'):comparison.load_phase_evaluation_policy(spec,config,member)
