"""Approved teacher-recovery student contract, isolated from legacy defaults."""
from types import SimpleNamespace as NS
import jax
import jax.numpy as jp
import numpy as np
import pytest


def test_ppo_coefficient_scales_real_actor_and_critic_gradients(monkeypatch):
    from brax.training.agents.ppo import losses
    from jit_dvgc.generative_bridge.student import make_joint_student_trainer
    def base(params,*args,**kw):
        p=jp.square(params.policy-3);v=jp.square(params.value-5)
        return p+v,dict(policy_loss=p,entropy_loss=jp.asarray(0.),v_loss=v)
    monkeypatch.setattr(losses,'compute_ppo_loss',base)
    norm=NS(count=jp.array(0.))
    net=NS(policy_network=NS(apply=lambda n,p,o:jp.ones((len(o['state']),4))*p),
           parametric_action_distribution=NS(mode=lambda x:x))
    def trainer(**kwargs):
        def objective(params):return losses.compute_ppo_loss(params,norm,NS(reward=jp.ones(2)),jax.random.PRNGKey(4),net)
        params=losses.PPONetworkParams(policy=jp.array(2.),value=jp.array(4.))
        (value,metrics),grad=jax.value_and_grad(objective,has_aux=True)(params)
        # Keep target pi0=0, hence gradient 4; base actor and critic each -.2.
        assert float(grad.policy)==pytest.approx(3.8)
        assert float(grad.value)==pytest.approx(-.2)
        assert float(value)==pytest.approx(4.2)
        assert float(metrics['ppo_loss'])==2.
        assert float(metrics['weighted_ppo_loss'])==pytest.approx(.2)
        assert float(metrics['ppo_coefficient'])==pytest.approx(.1)
        assert float(metrics['actual/gradient_norm_ppo'])==pytest.approx(.2)
        assert float(metrics['actual/gradient_norm_keep'])==4.
    make_joint_student_trainer(trainer,None,retention=(np.ones((2,76)),np.ones(2)/2),
        ppo_coefficient=.1,keep_coefficient=1.,actual_gradient_audit=True)(
        restore_params=(norm,jp.array(0.),jp.array(4.)))


def pulse():
    return dict(pulse_scope='full_task_start_only',onset_selection='uniform_complete_start',
        onsets=[0,5,10,15],steps=3,probability=1.,delta_limit=[.25]*4,
        logical_episode_rng=dict(master_seed=5,role='student_ppo',round=1,slot_ids=[0]))


def test_checked_four_onset_pulses_apply_relative_three_draws_and_real_clip():
    from jit_dvgc.iterative_probe_training import validate_training_action_pulse,initialize_training_action_pulse,apply_training_action_pulse
    config=pulse();validate_training_action_pulse(config)
    seen=set()
    for seed in range(120):
        info=initialize_training_action_pulse({},jax.random.PRNGKey(seed),config,full_task_start=True)
        onset=int(info['training_pulse_onset']);seen.add(onset)
        draws=np.asarray(info['training_pulse_draws'])
        assert np.max(np.abs(draws))<=.25
        for tick in range(19):
            actual,info,metrics=apply_training_action_pulse(info,jp.ones(4)*.95,config)
            wanted=draws[tick-onset] if onset<=tick<onset+3 else np.zeros(4)
            np.testing.assert_allclose(info['training_pulse_request'],wanted)
            np.testing.assert_allclose(actual,np.clip(.95+wanted,-1,1),atol=1e-7)
    assert seen=={0,5,10,15}
    snapshot=initialize_training_action_pulse({},jax.random.PRNGKey(3),config,full_task_start=False)
    for _ in range(20):
        _,snapshot,_=apply_training_action_pulse(snapshot,jp.zeros(4),config)
        np.testing.assert_array_equal(snapshot['training_pulse_request'],0)
    # Multi-onset semantics require explicit opt-in; old full-start contract remains t0.
    config.pop('onset_selection')
    with pytest.raises(ValueError,match='0/1/2'):validate_training_action_pulse(config)


def inventory():
    row=dict(role='TRAIN',verified_success=True,snapshot='/snapshot',snapshot_context_sha256='hash',snapshot_time=.125,snapshot_time_source={'path':'/trace','sha256':'hash'})
    return dict(schema='jit_teacher_recovery_reset_v1',role='TRAIN',
        complete_start=dict(probability=.4,qpos=[0.]*7,qvel=[0.]*6),
        current_teacher_verified=dict(probability=.3,entries=[{**row,'round':2}]),
        historical_success=dict(probability=.3,entries=[{**row,'round':1}]),round=2)


def test_three_reset_groups_fail_closed_and_have_declared_mass():
    from jit_dvgc.retention_reset import validate_inventory,group_indices
    inv=inventory();assert validate_inventory(inv) is inv
    groups=np.asarray(group_indices(jax.random.PRNGKey(5),30000,probabilities=[.4,.3,.3]))
    np.testing.assert_allclose(np.bincount(groups)/len(groups),[.4,.3,.3],atol=.015)
    for name in ('current_teacher_verified','historical_success'):
        damaged=inventory();damaged[name]['entries']=[]
        with pytest.raises(ValueError,match='empty'):validate_inventory(damaged)
        damaged=inventory();damaged[name]['entries'][0]['verified_success']=False
        with pytest.raises(ValueError,match='verified'):validate_inventory(damaged)
    damaged=inventory();damaged['current_teacher_verified']['entries'][0]['round']=1
    with pytest.raises(ValueError,match='round'):validate_inventory(damaged)


def test_real_brax_pi0_critic_fresh_adam_and_frozen_stats_survive_two_rounds(tmp_path):
    import functools,json,pickle
    from brax.envs.fast import Fast
    from brax.training.agents.ppo import train,networks
    from jit_dvgc.generative_bridge.student import make_joint_student_trainer
    from jit_dvgc.generative_bridge.learner_continuation import LearnerHooks,instrument_trainer
    from jit_dvgc.generative_bridge.actual_update import adam_update_count
    from jit_dvgc.handoff_bank import pytree_sha256
    kwargs=dict(environment=Fast(),num_timesteps=16,num_envs=2,episode_length=4,unroll_length=2,
        batch_size=2,num_minibatches=1,num_updates_per_batch=1,num_evals=2,run_evals=False,
        learning_rate=3e-5,normalize_observations=True,
        network_factory=functools.partial(networks.make_ppo_networks,
            policy_hidden_layer_sizes=(8,),value_hidden_layer_sizes=(8,)))
    _,pi0,_=train.train(**kwargs)
    first=LearnerHooks(tmp_path/'first',contract='frozen',trainer_sha='pinned',parent=None)
    wrapped=make_joint_student_trainer(instrument_trainer(train.train,first.initialize,first.save),
        None,retention=None,keep_coefficient=0,freeze_actor_normalizer=True,
        demo_clock='completed_transitions',ppo_coefficient=.1)
    wrapped(**{**kwargs,'environment':Fast(),'restore_params':pi0,'restore_value_fn':True})
    zero=json.loads((tmp_path/'first/transition_000000/manifest.json').read_text())
    with open(zero['state_path'],'rb') as f:initial=pickle.load(f)['training_state']
    assert pytree_sha256(initial.params.value)==pytree_sha256(pi0[2])
    assert int(adam_update_count(initial.optimizer_state))==0
    assert all(np.all(np.asarray(x)==0) for x in jax.tree.leaves(initial.optimizer_state))
    m=json.loads(__import__('pathlib').Path(first.latest_receipt()['path']).read_text())
    with open(m['state_path'],'rb') as f:saved=pickle.load(f)['training_state']
    assert pytree_sha256(saved.normalizer_params)==pytree_sha256(pi0[0])
    second=LearnerHooks(tmp_path/'second',contract='frozen',trainer_sha='pinned',parent=first.latest_receipt())
    make_joint_student_trainer(instrument_trainer(train.train,second.initialize,second.save),
        None,retention=None,keep_coefficient=0,freeze_actor_normalizer=True,
        demo_clock='completed_transitions',ppo_coefficient=.1)(**{**kwargs,'environment':Fast(),
            'restore_params':(saved.normalizer_params,saved.params.policy,saved.params.value),'restore_value_fn':True})
    m2=json.loads(__import__('pathlib').Path(second.latest_receipt()['path']).read_text())
    assert m2['normalizer_sha256']==m['normalizer_sha256']
    assert m2['lifetime_transitions']==32
    initialized=json.loads((tmp_path/'second/initialization.json').read_text())
    assert initialized['optimizer_sha256']==m['optimizer_sha256']
    assert initialized['rng_sha256']==m['rng_sha256']


def test_exact_demo_minibatch_is_half_current_half_history(tmp_path,monkeypatch):
    from brax.training.agents.ppo import losses
    from jit_dvgc.generative_bridge.student import make_joint_student_trainer
    from jit_dvgc.generative_bridge.contracts import file_sha
    path=tmp_path/'demo.npz'
    np.savez(path,observations=np.ones((3,76),np.float32),actions=np.zeros((3,4),np.float32),weights=[.5,.25,.25])
    demo=dict(schema='jit_bridge_demo_v1_2',count=3,path=str(path),sha256=file_sha(path),
        sample_roots=['new','old1','old2'],new_roots=['new'],sample_origins=['bridge_prefix']*3)
    usage={};norm=NS(count=jp.array(0.))
    monkeypatch.setattr(losses,'compute_ppo_loss',lambda *a,**k:(jp.array(1.),{}))
    net=NS(policy_network=NS(apply=lambda n,p,o:jp.ones((len(o['state']),4))*p),parametric_action_distribution=NS(mode=lambda x:x))
    def trainer(**k):
        losses.compute_ppo_loss(losses.PPONetworkParams(policy=jp.array(.1),value=jp.array(.2)),
            norm,None,jax.random.PRNGKey(5),net)
    make_joint_student_trainer(trainer,demo,retention=None,keep_coefficient=0,
        demo_coefficient_start=1.,demo_coefficient_end=1.,demo_batch_size=256,
        usage_sink=usage,exact_demo_fresh_history=True)(restore_params=(norm,jp.array(0.),None))
    assert usage['demo_samples_by_root']['new']==128
    assert sum(usage['demo_samples_by_root'].get(k,0) for k in ('old1','old2'))==128
    demo['new_roots']=[]
    with pytest.raises(ValueError,match='both'):
        make_joint_student_trainer(trainer,demo,retention=None,keep_coefficient=0,exact_demo_fresh_history=True)


def test_recovery_training_contract_constants_and_runtime_lr_fail_closed():
    from jit_dvgc.generative_bridge.student import recovery_training_contract,validate_recovery_training_contract
    c=recovery_training_contract();assert c['student_transitions']==32000
    assert c['student_learning_rate']==3e-5
    assert c['ppo_coefficient']==.1 and c['demo_coefficient']==c['retention_coefficient']==1
    validate_recovery_training_contract(c)
    c['retention_batch_size']=256
    with pytest.raises(ValueError,match='contract'):validate_recovery_training_contract(c)


def test_snapshot_time_is_read_from_hash_locked_original_control_clock(tmp_path):
    import json
    from jit_dvgc.retention_reset import original_snapshot_time
    from jit_dvgc.retention_repair import sha
    path=tmp_path/'identity.json';path.write_text(json.dumps({'episode_step':17}))
    row=dict(snapshot_time=.34,snapshot_time_source=dict(path=str(path),sha256=sha(path),
        control_step_field='episode_step',control_dt=.02))
    assert original_snapshot_time(row)==pytest.approx(.34)
    row['snapshot_time']=0.
    with pytest.raises(ValueError,match='original'):original_snapshot_time(row)
    row['snapshot_time']=.34;path.write_text(json.dumps({'episode_step':0}))
    with pytest.raises(ValueError,match='changed'):original_snapshot_time(row)


def test_full_learner_contract_locks_new_loss_weights_but_allows_round_data_change():
    from copy import deepcopy
    from jit_dvgc.generative_bridge.learner_continuation import learner_contract
    from jit_dvgc.generative_bridge.student import recovery_training_contract
    raw=dict(ppo=dict(learning_rate=3e-5,seed=1,requested_transitions=32000),inputs={},
        reward_mode='original_all_phases',reward_contract='original',snapshot_reset_contract='preserve',
        recovery_training_contract=recovery_training_contract(),generative_bridge_student=dict(
            ppo_coefficient=.1,demo_coefficient_start=1.,demo_coefficient_end=1.,retention_coefficient=1.,
            demo_batch_size=256,retention_batch_size=512,freeze_actor_normalizer=True,
            demo_clock='completed_transitions',exact_demo_fresh_history=True,demo_manifest={'path':'round1'}))
    changed=deepcopy(raw);changed['generative_bridge_student']['demo_manifest']={'path':'round2'}
    changed['ppo']['seed']=2
    assert learner_contract(raw)==learner_contract(changed)
    changed['generative_bridge_student']['ppo_coefficient']=1.
    assert learner_contract(raw)!=learner_contract(changed)


def test_recovery_first_round_hook_rejects_nonfresh_adam(tmp_path):
    from jit_dvgc.generative_bridge.learner_continuation import LearnerHooks
    from .test_ppo_learner_continuation import state_after_updates
    _,state=state_after_updates()
    hook=LearnerHooks(tmp_path/'bad',contract='c',trainer_sha='s',parent=None,require_fresh_adam=True)
    with pytest.raises(ValueError,match='fresh Adam'):hook.initialize(state,jax.random.PRNGKey(1))


def test_recovery_constructor_and_terminal_ppo_reset_keep_snapshot_context(tmp_path,monkeypatch):
    import json
    from jit_dvgc.retention_reset import FourPoolReset
    from jit_dvgc.retention_repair import sha
    from jit_dvgc.ppo import wrap_for_jit_training
    from .test_retention_t1_reset import saved_state,NoPhysicsEnv
    import jit_dvgc.unified_envelope_snapshot as snapshots
    import jit_dvgc.handoff_snapshot as handoff
    states=[saved_state(i) for i in (0,2,3)]
    identity={'physical':'unchanged'};snaps={};inv=inventory()
    inv['complete_start'].update(qpos=np.asarray(states[0].data.qpos),qvel=np.asarray(states[0].data.qvel))
    for group,name in ((1,'current_teacher_verified'),(2,'historical_success')):
        state=states[group];clock=int(state.info['episode_step'])
        source=tmp_path/f'identity{group}.json';source.write_text(json.dumps({'episode_step':clock}))
        path=str(tmp_path/f'snapshot{group}')
        snaps[path]=NS(compatibility_identity=identity,observation=np.asarray(state.obs['state']),
            observation_fifo=np.asarray(state.info['history'].frames),state=state)
        inv[name]['entries'][0].update(snapshot=path,snapshot_context_sha256='context',snapshot_time=clock*.02,
            snapshot_time_source=dict(path=str(source),sha256=sha(source),control_step_field='episode_step',control_dt=.02))
    monkeypatch.setattr(snapshots,'load_unified_envelope_snapshot',lambda p:snaps[str(p)])
    monkeypatch.setattr(snapshots,'snapshot_context_sha256',lambda snap:'context')
    # Real snapshot restoration omits MJX time: use zero here and prove its source.
    monkeypatch.setattr(snapshots,'restore_unified_envelope_snapshot',lambda snap,env:snap.state.replace(
        data=snap.state.data.replace(time=jp.asarray(0.))))
    monkeypatch.setattr(handoff,'compatibility_identity',lambda env:identity)
    class StartEnv(NoPhysicsEnv):
        def _reset_jump_start_unified(self,rng,initial_qpos,initial_qvel):
            return states[0].replace(info={**states[0].info,'rng':rng})
    adapter=FourPoolReset(StartEnv(),inv)
    wrapped=wrap_for_jit_training(adapter.install(),episode_length=400)
    state=jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(44),128))
    result=jax.jit(wrapped.step)(state,jp.ones((128,4)))
    groups=np.asarray(result.info['retention_reset_group']);assert set(groups)=={0,1,2}
    for lane,group in enumerate(groups):
        expected=states[group]
        np.testing.assert_array_equal(result.data.qpos[lane],expected.data.qpos)
        np.testing.assert_array_equal(result.obs['state'][lane],expected.obs['state'])
        for name in ('history','up_events','down_events','episode_step','phase_episode_step','active_phase','last_action'):
            for a,b in zip(jax.tree.leaves(result.info[name]),jax.tree.leaves(expected.info[name])):
                np.testing.assert_array_equal(a[lane],b)
        assert bool(result.info['reset_from_jump_start'][lane]) == bool(group==0)
        if group:
            assert float(result.data.time[lane])==pytest.approx(int(expected.info['episode_step'])*.02)
            np.testing.assert_array_equal(jax.random.key_data(result.info['rng'][lane]),jax.random.key_data(expected.info['rng']))


def test_runtime_recovery_declaration_restores_pi0_critic_and_checks_actual_lr():
    from jit_dvgc.generative_bridge.student import recovery_training_contract,apply_recovery_runtime_contract
    c=recovery_training_contract()
    student=dict(ppo_coefficient=.1,demo_coefficient_start=1.,demo_coefficient_end=1.,retention_coefficient=1.,
        demo_batch_size=256,retention_batch_size=512,freeze_actor_normalizer=True,
        demo_clock='completed_transitions',exact_demo_fresh_history=True)
    raw={'recovery_training_contract':c};kwargs=dict(learning_rate=3e-5,restore_value_fn=False)
    apply_recovery_runtime_contract(raw,student,kwargs)
    assert kwargs['restore_value_fn'] is True
    kwargs['learning_rate']=1e-4
    with pytest.raises(ValueError,match='learning rate'):apply_recovery_runtime_contract(raw,student,kwargs)
    kwargs['learning_rate']=3e-5;student['retention_coefficient']=.2
    with pytest.raises(ValueError,match='loss'):apply_recovery_runtime_contract(raw,student,kwargs)
