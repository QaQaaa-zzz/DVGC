"""Behavioral contracts for the finite, paired PPO absorption pilot."""
import importlib
import json
from pathlib import Path

import numpy as np
import pytest


def module():
    assert importlib.util.find_spec('jit_dvgc.retention_t1') is not None, 'T1 coordinator missing'
    return importlib.import_module('jit_dvgc.retention_t1')


def test_budget_reserves_all_three_repeats_and_engineering_before_dispatch():
    m = module()
    b = m.budget(1122852)
    assert b['total'] == 700200
    assert b['D2_cumulative_maximum'] == 1823052
    assert b['engineering_ppo'] + b['reset_GPU'] == 6600
    assert b['student_DEV'] == 153600
    assert b['source_DEV_repeats'] == 51200
    with pytest.raises(ValueError, match='cap'):
        m.budget(1300000)
    ledger = m.CostLedger(b)
    first = ledger.reserve('train_keep', 'PPO', 32000)
    ledger.complete(first, {'phase': 'completed', 'charged_interactions': 32000})
    ledger.reserve('train_demo', 'PPO', 32000)
    with pytest.raises(ValueError, match='category'):
        ledger.reserve('extra_update', 'PPO', 1)
    # An unsuccessful worker keeps its entire reservation.
    assert ledger.charged == 64000


def test_t0_gate_is_required_and_unknown_is_not_passing(tmp_path):
    m = module()
    p = tmp_path / 'T0_gate.json'
    p.write_text(json.dumps({'schema': 'jit_retention_T0_saved_diagnostics_v1',
                            'phase': 'blocked', 't1_allowed': False}))
    with pytest.raises(ValueError, match='T0'):
        m.require_t0(p)
    p.write_text(json.dumps({'schema': 'jit_retention_T0_saved_diagnostics_v1',
                            'phase': 'passed', 't1_allowed': True,
                            'simulation_steps': 0, 'supervised_updates': 0,
                            'blocking_failures': []}))
    assert m.require_t0(p)['t1_allowed'] is True


def labels(success=0):
    return {g: [1] * success + [0] * (16 - success) for g in 'ABCD'}


def fixture_results():
    roots = [f'TRAIN-42-{i}' for i in (68,37,10,18,26,30,55,63)]
    source = [labels(8) for _ in range(3)]
    def arm(ids, full):
        return {'checkpoint_transition': 32000,
                'TRAIN': [{r: int(int(r.rsplit('-',1)[1]) in ids) for r in roots} for _ in range(3)],
                'SOLVER_DEV': {'solver-'+str(i): int(i < 3) for i in range(7)},
                'DEV': [full.copy() for _ in range(3)]}
    results = {'PPO_keep': arm({68,37,10,26}, labels(9)),
               'PPO_demo_keep': arm({68,37,10,26,30}, labels(10))}
    required = [r for r in roots if int(r.rsplit('-',1)[1]) in (30,68,37,10,26)]
    return source, results, required


def test_mechanism_gate_uses_fixed_endpoint_stable_roots_and_paired_old_losses():
    m = module()
    source, results, required = fixture_results()
    out = m.mechanism_decision(source, results, required)
    assert out['t1_passed'] is True
    assert out['arms']['PPO_demo_keep']['stable_train_success'] == 5
    # A single success flip at an important root does not qualify as absorption.
    results['PPO_demo_keep']['TRAIN'][2][required[0]] = 0
    out = m.mechanism_decision(source, results, required)
    assert not out['t1_passed']
    assert out['arms']['PPO_demo_keep']['TRAIN_classification'][required[0]] == 'ambiguous'


def test_three_dev_repeats_are_mandatory_even_without_a_warning():
    m = module()
    source, results, required = fixture_results()
    results['PPO_demo_keep']['DEV'] = results['PPO_demo_keep']['DEV'][:1]
    out = m.mechanism_decision(source, results, required)
    assert out['evaluation_status'] == 'partial' and out['t1_passed'] is None


def test_guard_keeps_two_confirmed_drops_even_if_third_recovers():
    m = module()
    source, results, required = fixture_results()
    for repeat in (0, 1):
        results['PPO_demo_keep']['DEV'][repeat] = labels(7)
    out = m.mechanism_decision(source, results, required)
    assert out['arms']['PPO_demo_keep']['confirmed_regression'] == ['B', 'D']
    assert not out['t1_passed']


def test_clock_sorting_does_not_treat_callback_arrival_as_training_progress():
    m = module()
    rows = [{'actual/completed_training_transitions': 6400., 'actual/optimizer_updates': 16.},
            {'actual/completed_training_transitions': 3200., 'actual/optimizer_updates': 1.},
            {'actual/completed_training_transitions': 3200., 'actual/optimizer_updates': 8.}]
    ordered = m.ordered_updates(rows)
    assert [r['actual/optimizer_updates'] for r in ordered] == [1., 8., 16.]
    with pytest.raises(ValueError, match='finite'):
        m.ordered_updates([{'actual/completed_training_transitions': float('nan')}])


def test_actual_update_counter_is_read_from_adam_not_callback_order():
    import jax.numpy as jp
    import optax
    from jit_dvgc.generative_bridge import actual_update
    assert hasattr(actual_update, 'adam_update_count'), 'actual optimizer counter missing'
    opt = optax.chain(optax.clip_by_global_norm(.75), optax.adam(7.5e-6))
    params = {'x': jp.ones(2)}
    state = opt.init(params)
    for _ in range(3):
        _, state = opt.update(params, state, params)
    assert int(actual_update.adam_update_count(state)) == 3


def test_paired_contract_changes_only_demo_and_has_independent_128k_schedule():
    m = module()
    assert hasattr(m, 'training_contract'), 'T1 training contract missing'
    p = {'seed': 1010257101, 'models': {'pi0': {'frozen_policy': '/pi0'}},
         'demo_manifest': '/locked/demo', 'keep_receipt': '/locked/keep',
         'reset_inventory': '/locked/four_pools'}
    a = m.training_contract(p, 'PPO_keep')
    b = m.training_contract(p, 'PPO_demo_keep')
    changed = {k for k in a if a[k] != b[k]}
    assert changed == {'arm', 'demo_coefficient_start', 'demo_coefficient_end'}
    assert b['demo_schedule_transitions'] == 128000
    assert b['ppo']['requested_transitions'] == 32000
    assert b['checkpoints'] == [0, 16000, 32000]
    assert b['critic_initialization'] == 'source_pi0_then_trainable'
    assert b['normalizer'] == 'entire_source_state_frozen_including_count'
    assert b['ppo']['learning_rate'] == 7.5e-6
    for seed in (1011501, 1011502, 1011503):
        with pytest.raises(ValueError, match='paper'):
            m.training_contract({**p, 'seed': seed}, 'PPO_keep')


def test_evaluation_schedule_has_no_best_search_or_conditional_repeat():
    m = module()
    assert hasattr(m, 'evaluation_jobs'), 'fixed T1 evaluation schedule missing'
    jobs = m.evaluation_jobs()
    assert len([x for x in jobs if x['kind'] == 'DEV' and x['arm'] in m.ARMS]) == 6
    assert len([x for x in jobs if x['kind'] == 'TRAIN']) == 6
    assert len([x for x in jobs if x['kind'] == 'SOLVER_DEV']) == 2
    assert len([x for x in jobs if x['arm'] == 'pi0']) == 2
    assert {x['transition'] for x in jobs if x['arm'] in m.ARMS} == {32000}
    assert sum(x['reservation'] for x in jobs) == 626400


def test_ambiguous_old_ability_debt_cannot_disappear_from_the_gate():
    m = module()
    source, results, required = fixture_results()
    changed = copy_labels = {g:list(v) for g,v in results['PPO_demo_keep']['DEV'][2].items()}
    changed['A'][0] = 0
    results['PPO_demo_keep']['DEV'][2] = copy_labels
    out = m.mechanism_decision(source, results, required)
    assert out['arms']['PPO_demo_keep']['stable_lost'] == []
    assert out['arms']['PPO_demo_keep']['old_ambiguous'] == ['A_00']
    assert not out['t1_passed']


def test_update_contract_rejects_nonfinite_and_wrong_weighted_gradient_sum():
    m = module()
    assert hasattr(m, 'validate_actual_update'), 'numerical update contract missing'
    tolerance = {'gradient_sum_atol':1e-5,'gradient_sum_rtol':1e-3}
    row = {'actual/completed_training_transitions':3200.,'actual/optimizer_updates':1.,
           'actual/actor_gradient_norm':11.,'actual/actor_gradient_sum_norm':11.0002,
           'actual/actor_gradient_norm_sum_difference':.0002}
    m.validate_actual_update(row,tolerance)
    with pytest.raises(ValueError,match='gradient sum'):
        m.validate_actual_update({**row,'actual/actor_gradient_norm_sum_difference':.1},tolerance)
    with pytest.raises(ValueError,match='nonfinite'):
        m.validate_actual_update({**row,'actual/actor_gradient_norm':float('inf')},tolerance)


def test_real_cpu_ppo_composition_preserves_stats_and_serialized_clock(tmp_path):
    assert importlib.util.find_spec('jit_dvgc.retention_t1_runtime') is not None, 'T1 runtime missing'
    import functools
    import pickle
    import jax
    import jax.numpy as jp
    from brax.envs.fast import Fast
    from brax.training.agents.ppo import train, networks
    from brax.training.acme import running_statistics as rs
    from jit_dvgc.generative_bridge.learner_continuation import LearnerHooks
    from jit_dvgc.handoff_bank import pytree_sha256
    from jit_dvgc.ppo_numerics import guard_ppo_updates
    from jit_dvgc.retention_t1_runtime import compose_trainer

    class ObservedFast(Fast):
        def reset(self, rng):
            s = super().reset(rng)
            return s.replace(obs={'state':jp.zeros(76),'privileged_state':jp.zeros(106)})
        def step(self, state, action):
            s = super().step(state,action[:3])
            return s.replace(obs={'state':jp.full(76,s.reward),'privileged_state':jp.full(106,s.reward)})
        @property
        def action_size(self): return 4

    factory = functools.partial(networks.make_ppo_networks,policy_hidden_layer_sizes=(8,),
                                value_hidden_layer_sizes=(8,),policy_obs_key='state',value_obs_key='privileged_state')
    net = factory({'state':76,'privileged_state':106},4,preprocess_observations_fn=rs.normalize)
    norm = jax.tree.map(jp.asarray,rs.init_state({'state':jp.zeros(76),'privileged_state':jp.zeros(106)}))
    params = (norm,net.policy_network.init(jax.random.PRNGKey(1)),net.value_network.init(jax.random.PRNGKey(2)))
    data = tmp_path/'data.npz'
    np.savez(data,observations=np.ones((2,76),np.float32),actions=np.full((2,4),.2,np.float32),weights=np.array([.5,.5]))
    from jit_dvgc.retention_repair import sha
    manifest = dict(schema='jit_bridge_demo_v1_2',path=str(data),sha256=sha(data),count=2,sample_roots=['r','r'],sample_origins=['bridge_prefix','source_tail'])
    hooks = LearnerHooks(tmp_path/'learner',contract='test',trainer_sha='test',parent=None,preserve_stage_clock=True)
    rows=[]
    contract = module().training_contract({'seed':1010257101,'models':{'pi0':{'frozen_policy':'test'}},
        'demo_manifest':'test','keep_receipt':'test','reset_inventory':'test'},'PPO_demo_keep')
    wrapped = compose_trainer(train.train,hooks,contract,manifest,(np.zeros((2,76),np.float32),np.array([.5,.5])),
                              params[:2],rows.append)
    with guard_ppo_updates(tmp_path):
        wrapped(environment=ObservedFast(),num_timesteps=8,num_envs=2,episode_length=4,
            unroll_length=2,batch_size=2,num_minibatches=1,num_updates_per_batch=1,num_evals=3,
            run_evals=False,normalize_observations=True,network_factory=factory,
            restore_params=params,restore_value_fn=True,learning_rate=7.5e-6,max_grad_norm=.75)
    jax.effects_barrier()
    receipt=json.loads(Path(hooks.latest_receipt()['path']).read_text())
    with open(receipt['state_path'],'rb') as f: saved=pickle.load(f)
    assert receipt['local_transitions']==8
    assert pytree_sha256(saved['training_state'].normalizer_params)==pytree_sha256(norm)
    ordered=module().ordered_updates(rows)
    assert [int(r['actual/optimizer_updates']) for r in ordered]==[1,2]
    assert [int(r['actual/completed_training_transitions']) for r in ordered]==[4,8]
    assert ordered[-1]['effective_lambda_demo']==pytest.approx(.2-.15*8/128000)
    # Audit the actual serialized full learner, not just callback counters.
    from jit_dvgc.retention_repair import write
    from jit_dvgc.retention_t1_runtime import verify_saved_training,first_loss_gate
    audit_contract={k:v for k,v in contract.items() if k!='ppo'}
    write(tmp_path/'training_contract.json',audit_contract)
    write(tmp_path/'actual_updates.json',ordered)
    first_loss_gate(ordered[0],audit_contract,tmp_path/'first_loss_gate.json')
    for step,values in ((0,params),(8,(saved['training_state'].normalizer_params,saved['training_state'].params.policy,saved['training_state'].params.value))):
        path=tmp_path/f'inference_{step:06d}.pkl'
        with path.open('wb') as stream:pickle.dump(values,stream)
        manifest=tmp_path/f'learner/transition_{step:06d}/manifest.json'
        write(tmp_path/f'checkpoint_{step:06d}.json',dict(inference=str(path),inference_sha256=sha(path),
              learner=dict(path=str(manifest),sha256=sha(manifest))))
    policy={name:pytree_sha256(value) for name,value in zip(('normalizer_sha256','actor_sha256','critic_sha256'),params)}
    result=verify_saved_training(tmp_path,policy,8,2)
    assert result['phase']=='passed' and result['checks']['fresh_Adam_at_zero']
    assert result['checks']['full_state_restore_exact'] and result['checks']['critic_changed']


def test_first_loss_gate_uses_actual_optimizer_clock_and_existing_limit(tmp_path):
    from jit_dvgc.retention_t1_runtime import first_loss_gate
    contract = {'first_behavior_KL_limit':.05}
    path = tmp_path/'first.json'
    row = {'actual/optimizer_updates':2.,'audit/installed_behavior_kl':.2,'audit/installed_self_kl':0.}
    assert first_loss_gate(row,contract,path) is None and not path.exists()
    row['actual/optimizer_updates']=1.
    with pytest.raises(FloatingPointError):first_loss_gate(row,contract,path)
    assert json.loads(path.read_text())['status']=='diagnostic_stop'
    row['audit/installed_behavior_kl']=.02
    assert first_loss_gate(row,contract,path)['status']=='passed'
    row['audit/installed_self_kl']=float('nan')
    with pytest.raises(FloatingPointError):first_loss_gate(row,contract,path)


@pytest.mark.parametrize('repeats',[(0,1),(0,2),(1,2)])
def test_guard_any_two_of_three_same_cells_rejects(repeats):
    source,results,required=fixture_results()
    for index in repeats:results['PPO_demo_keep']['DEV'][index]=labels(7)
    out=module().mechanism_decision(source,results,required)
    assert out['arms']['PPO_demo_keep']['confirmed_regression']==['B','D']
    assert not out['t1_passed']


@pytest.mark.parametrize('bad',[None,2,float('nan')])
@pytest.mark.parametrize('role',['TRAIN','SOLVER_DEV'])
def test_unknown_root_labels_fail_closed(bad,role):
    source,results,required=fixture_results()
    target=results['PPO_demo_keep'][role][0] if role=='TRAIN' else results['PPO_demo_keep'][role]
    target[next(reversed(target))]=bad
    with pytest.raises(ValueError):module().mechanism_decision(source,results,required)


@pytest.mark.parametrize('role',['TRAIN','SOLVER_DEV'])
def test_substituted_root_identity_rejected(role):
    source,results,required=fixture_results()
    targets=results['PPO_demo_keep'][role] if role=='TRAIN' else [results['PPO_demo_keep'][role]]
    for target in targets:target['unrelated']=target.pop(next(reversed(target)))
    with pytest.raises(ValueError):module().mechanism_decision(source,results,required)
