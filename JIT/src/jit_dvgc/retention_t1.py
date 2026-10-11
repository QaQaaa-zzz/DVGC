"""One finite paired PPO pilot; no automatic continuation, E/G update or adoption.

Prepare/audit/report are CPU/file operations. Only ``run`` dispatches physics,
after the saved-data T0 gate, immutable input locks and one-shot claim pass.
"""
from pathlib import Path
import copy
import json
import math
import os
import subprocess
import time

import numpy as np

from .retention_repair import read, write, sha
from .retention_repair_execution import PY

SCHEMA = 'jit_retention_T1_paired_ppo_v1'
ARMS = ('PPO_keep', 'PPO_demo_keep')
PAPER_SEEDS = (1011501, 1011502, 1011503)


def budget(prior_charged):
    result = dict(PPO=64000, snapshot_first=204000, student_DEV=153600,
                  TRAIN_repeats=217600, TRAIN_panels=3200,
                  engineering_ppo=6400, reset_GPU=200, source_DEV_repeats=51200)
    result['total'] = sum(result.values())
    result.update(prior_charged=int(prior_charged), hard_cap=2000000,
                  D2_cumulative_maximum=int(prior_charged) + result['total'],
                  wall_seconds=21600, formal_transitions_per_arm=32000,
                  engineering_transitions_per_arm=3200)
    if prior_charged < 0 or result['D2_cumulative_maximum'] > result['hard_cap']:
        raise ValueError('T1 cumulative physics cap exceeded')
    return result


class CostLedger:
    """Reserve before dispatch; failed attempts retain the entire reservation."""
    def __init__(self, declaration, rows=None):
        self.budget = declaration
        self.rows = list(rows or [])

    @property
    def charged(self):
        return sum(r['charged'] for r in self.rows)

    def reserve(self, name, category, maximum):
        if any(r['stage'] == name for r in self.rows):
            raise ValueError('no implicit retry or duplicate stage')
        if type(maximum) is not int or maximum <= 0:
            raise ValueError('positive integer reservation required')
        used = sum(r['charged'] for r in self.rows if r['category'] == category)
        if category not in self.budget or used + maximum > self.budget[category]:
            raise ValueError('physics category reservation exceeds declaration')
        if (self.charged + maximum > self.budget['total'] or
                self.charged + maximum + self.budget['prior_charged'] > self.budget['hard_cap']):
            raise ValueError('physics cap before dispatch')
        row = dict(stage=name, category=category, reservation=maximum,
                   charged=maximum, phase='running')
        self.rows.append(row)
        return row

    def complete(self, row, receipt):
        count = receipt.get('charged_interactions')
        if (receipt.get('phase') != 'completed' or type(count) is not int or
                not 0 <= count <= row['reservation']):
            raise ValueError('incomplete or over-budget child receipt')
        row.update(phase='completed', charged=count)
        for field in ('active_interactions', 'padding_interactions', 'completed_training_transitions',
                      'actual_optimizer_updates', 'wall_seconds'):
            if field in receipt:
                row[field] = receipt[field]


def require_t0(path):
    gate = read(path)
    if (gate.get('schema') != 'jit_retention_T0_saved_diagnostics_v1' or
            gate.get('phase') != 'passed' or gate.get('t1_allowed') is not True or
            gate.get('simulation_steps') != 0 or gate.get('supervised_updates') != 0 or
            gate.get('blocking_failures') != []):
        raise ValueError('passing zero-physics T0 gate required')
    return gate


def training_contract(plan, arm):
    if arm not in ARMS:
        raise ValueError('unknown T1 arm')
    if plan['seed'] in PAPER_SEEDS:
        raise ValueError('pilot seed must be separate from paper seeds')
    demo = arm == 'PPO_demo_keep'
    return dict(arm=arm, source_frozen_policy=plan['models']['pi0']['frozen_policy'],
                seed=plan['seed'], initializer='original_pi0_actor_and_critic_fresh_Adam_RNG',
                critic_initialization='source_pi0_then_trainable',
                normalizer='entire_source_state_frozen_including_count',
                gradient_sum_atol=1e-5,gradient_sum_rtol=1e-3,first_behavior_KL_limit=.05,
                first_loss_contract_source='generative_bridge/campaign.py student v1.2 max_first_behavior_kl=0.05',
                first_loss_stop='actual Adam count1; asynchronous callback stop is not a zero-update guarantee',
                checkpoints=[0,16000,32000], fixed_endpoint=32000,
                demo_manifest=plan['demo_manifest'], keep_receipt=plan['keep_receipt'],
                reset_inventory=plan['reset_inventory'],
                demo_coefficient_start=.2 if demo else 0., demo_coefficient_end=.05 if demo else 0.,
                demo_schedule_transitions=128000, demo_clock='completed_transitions',
                keep_coefficient=.2, demo_batch_size=256, retention_batch_size=256,
                training_action_pulse=None, reset_probabilities=[.25]*4,
                ppo=dict(requested_transitions=32000,num_parallel_envs=128,unroll_length=25,
                         batch_size=16,num_minibatches=8,num_updates_per_batch=1,
                         episode_horizon=400,learning_rate=7.5e-6,clipping_epsilon=.1,
                         discounting=.995,gae_lambda=.97,entropy_cost=.001,
                         reward_scaling=.1,max_grad_norm=.75),
                joint_clip='unchanged shared Actor+critic global norm',
                retention_guard='either arm: same B or D cell drop >5pp in any2 of fixed3 paired repeats rejects; no extra evaluation',
                performance_gate_during_training=False, automatic_extension=False)


def evaluation_jobs():
    jobs = [dict(arm='pi0',kind='DEV',repeat=i,transition=0,
                 category='source_DEV_repeats',reservation=25600) for i in (1,2)]
    for arm in ARMS:
        for repeat in range(3):
            jobs.append(dict(arm=arm,kind='DEV',repeat=repeat,transition=32000,
                             category='student_DEV',reservation=25600))
            jobs.append(dict(arm=arm,kind='TRAIN',repeat=repeat,transition=32000,
                             category='snapshot_first' if repeat == 0 else 'TRAIN_repeats',
                             reservation=54400))
        jobs.append(dict(arm=arm,kind='SOLVER_DEV',repeat=0,transition=32000,
                         category='snapshot_first',reservation=47600))
    return jobs


def _validate_data(plan):
    from .retention_reset import validate_inventory
    inventory = validate_inventory(read(plan['reset_inventory']))
    demo, keep = read(plan['demo_manifest']), read(plan['keep_receipt'])
    policy = plan['models']['pi0']['policy']
    if (demo.get('schema') != 'jit_bridge_demo_v1_2' or demo.get('role') != 'TRAIN' or
            demo.get('count') != 753 or demo.get('old_count') != 525 or demo.get('new_count') != 228 or
            demo.get('source_normalizer_sha256') != policy['normalizer_sha256'] or
            demo.get('source_actor_sha256') != policy['actor_sha256']):
        raise ValueError('locked753 TRAIN demo and original pi0 coordinates required')
    if keep.get('role') != 'train' or keep.get('full_success') is not True or keep.get('DEV_used') is not False:
        raise ValueError('successful TRAIN-only keep required')
    for manifest, fields, count in ((demo,('observations','actions'),753),
                                    (keep,('actor_observation_before',),6289)):
        if sha(manifest['path']) != manifest['sha256']:
            raise ValueError('training dataset drift')
        with np.load(manifest['path'],allow_pickle=False) as data:
            for field in fields:
                if len(data[field]) != count or not np.isfinite(data[field]).all():
                    raise ValueError('finite exact TRAIN frame count required')
            weights = data['weights']
            if weights.shape != (count,) or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
                raise ValueError('frozen per-frame sampling weights required')
    if len(demo.get('sample_roots', [])) != 753 or len(demo.get('sample_origins', [])) != 753:
        raise ValueError('per-frame demonstration provenance required')
    train, solver = read(plan['train_roots']), read(plan['solver_roots'])
    if len(train) != 8 or len(solver) != 7:
        raise ValueError('fixed TRAIN8 and solver7 required')
    if any(x['root']['role'] != 'TRAIN' for x in train) or any(x['root']['role'] != 'SOLVER_DEV' for x in solver):
        raise ValueError('root role drift')
    if {x['root']['root_id'] for x in train} & {x['root']['root_id'] for x in solver}:
        raise ValueError('TRAIN and solver overlap')
    if any(r.get('role') != 'TRAIN' for name in ('teacher_recoverable','pi0_success_snapshots') for r in inventory[name]['entries']):
        raise ValueError('reset data leakage')
    return demo, keep, inventory


def prepare(parent, t0_gate, output, repository, tensorboard_port=6032, prior_failed_plan=None):
    """Freeze the new pilot without physics, training, processes or GPU imports."""
    parent, t0_gate = Path(parent).resolve(), Path(t0_gate).resolve()
    require_t0(t0_gate)
    old = read(parent); oldroot = Path(old['output']); code = Path(repository).resolve()
    from .retention_b import DEMO_SCHEMA, _history_locks
    if old.get('schema') != DEMO_SCHEMA or read(oldroot/'status.json').get('phase') != 'completed':
        raise ValueError('completed self-visit BC evidence required; never resume BC')
    prior = old['budget']['prior_charged'] + sum(x['charged'] for x in read(oldroot/'costs.json'))
    if read(oldroot/'status.json')['charged_interactions'] != prior - old['budget']['prior_charged']:
        raise ValueError('parent measured cost mismatch')
    failed_receipts = []
    if prior_failed_plan is not None:
        previous_path = Path(prior_failed_plan).resolve()
        previous = read(previous_path); previous_root = Path(previous['output'])
        previous_status = read(previous_root/'status.json')
        previous_costs = read(previous_root/'costs.json')
        failed_cost = sum(row['charged'] for row in previous_costs)
        if (previous.get('schema') != SCHEMA or Path(previous['parent']).resolve() != parent
                or previous_status.get('phase') != 'failed'
                or previous_status.get('formal_training_transitions') != 0
                or previous_status.get('engineering_training_transitions') != 0
                or previous_status.get('charged_interactions') != failed_cost
                or previous['budget']['prior_charged'] != prior):
            raise ValueError('prior failed engineering receipt mismatch')
        prior += failed_cost
        failed_receipts = [str(previous_path), str(previous_root/'status.json'), str(previous_root/'costs.json')]
    locks, history = _history_locks(old['locks'],old,code)
    root = Path(output).resolve()
    if root.exists():
        raise ValueError('new immutable output required')
    source_d2 = read(old['source_plan'])
    p = dict(schema=SCHEMA,stage='T1_paired_PPO',output=str(root),code=str(code),
             code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip(),
             parent=str(parent),t0_gate=str(t0_gate),t0_gate_sha256=sha(t0_gate),
             models=copy.deepcopy(old['models']),seed=source_d2['TRAIN_seed'],
             seed_role='D2 fixed pilot seed; distinct from paper seeds1011501/1011502/1011503',
             demo_manifest=old['demo_manifest'],keep_receipt=old['keep_receipt'],
             reset_inventory=old['reset_inventory'],train_roots=old['train_roots'],
             solver_roots=old['solver_roots'],batches=old['batches'],
             reused_zero=old['reused_zero'],budget=budget(prior),
             tensorboard_port=int(tensorboard_port),automatic_next_stage=False,published_policy=None,
             source_report=str(oldroot/'self_visit_demo_report_0003'),
             source_D2_plan=old['source_plan'],E_updates=0,G_updates=0,
             evaluation_jobs=evaluation_jobs(),historical_code_locks=history,
             preservation='old observations/actions/weights unchanged; no DEV training or new teacher search')
    demo, keep, inventory = _validate_data(p)
    suffixes = {int(x['root']['root_id'].rsplit('-',1)[1]):x['root']['root_id'] for x in read(p['train_roots'])}
    p['required_train_roots'] = [suffixes[i] for i in (30,68,37,10,26)]
    p['contracts'] = {arm:training_contract(p,arm) for arm in ARMS}
    files = [parent,t0_gate,oldroot/'status.json',oldroot/'costs.json',p['source_D2_plan'],
             p['demo_manifest'],p['keep_receipt'],demo['path'],keep['path'],p['reset_inventory'],
             p['train_roots'],p['solver_roots'],p['reused_zero']['zero_checkpoint'],
             inventory['random_complete']['path']]
    for group in ('pi0_success_snapshots','teacher_recoverable'):
        for row in inventory[group]['entries']:
            files.extend([Path(row['snapshot'])/'snapshot.pkl',Path(row['snapshot'])/'identity.json'])
    for batch in p['batches']:
        spec = read(batch['spec']); files.extend([batch['spec'],spec['initial_state_bank'],spec['frozen_request_table']['path']])
    for name in p['reused_zero']['DEV'].values():
        files.extend(Path(name)/f for f in ('status.json','prefixes.npz'))
    for model in p['models'].values():
        pol = model['policy']
        files.extend([model['frozen_policy'],pol['formal_config'],Path(pol['checkpoint'])/'payload.pkl',Path(pol['checkpoint'])/'identity.json'])
    for f in files:
        locks[str(Path(f).resolve())] = sha(f)
    source_names = ('retention_t1.py','retention_t1_runtime.py','retention_reset.py','ppo.py','ppo_numerics.py',
                    'generative_bridge/student.py','generative_bridge/student_clock.py',
                    'generative_bridge/actual_update.py','generative_bridge/learner_continuation.py',
                    'pulse_exploration_runtime.py','retention_b.py','checkpoint.py','unified_formal.py',
                    'unified_env.py','continuation/device_rollout.py','error_notifications.py')
    files.extend(failed_receipts)
    for receipt in failed_receipts: locks[receipt] = sha(receipt)
    p['prior_failed_receipts'] = failed_receipts
    for name in source_names:
        f = code/'JIT/src/jit_dvgc'/name; locks[str(f)] = sha(f)
    cli = code/'JIT/cli/run_retention_t1.py'; locks[str(cli)] = sha(cli)
    p['locks'] = locks
    root.mkdir(parents=True,exist_ok=False)
    write(root/'plan.json',p); write(root/'budget_dry_run.json',p['budget'])
    write(root/'source_lock.json',dict(locks=locks,historical_code_locks=history))
    for arm, contract in p['contracts'].items():
        write(root/f'{arm}_contract.json',contract)
    write(root/'status.json',dict(phase='prepared',stage=p['stage'],evaluation_status='not_started',
          charged_interactions=0,formal_training_transitions=0,engineering_training_transitions=0,
          E_updates=0,G_updates=0,automatic_next_stage=False,published_policy=None))
    return root/'plan.json'


def audit(path):
    p = read(path)
    if (p.get('schema') != SCHEMA or p.get('automatic_next_stage') is not False or
            p.get('published_policy') is not None or p.get('E_updates') != 0 or p.get('G_updates') != 0):
        raise ValueError('finite T1 scope drift')
    if p['budget'] != budget(p['budget']['prior_charged']) or p['evaluation_jobs'] != evaluation_jobs():
        raise ValueError('T1 budget or fixed evaluation schedule drift')
    if sha(p['t0_gate']) != p['t0_gate_sha256']:
        raise ValueError('T0 gate changed')
    require_t0(p['t0_gate'])
    for f, expected in p['locks'].items():
        if sha(f) != expected:
            raise ValueError('T1 input/code drift '+f)
    for arm in ARMS:
        expected = training_contract(p,arm)
        if p['contracts'][arm] != expected or read(Path(p['output'])/f'{arm}_contract.json') != expected:
            raise ValueError('paired training contract drift')
    _validate_data(p)
    return dict(phase='passed',schema=SCHEMA,budget=p['budget'],T0='passed',
                simulation_steps=0,training_updates=0,pilot_seed=p['seed'],
                checkpoint_selection='fixed32000, no best',automatic_next_stage=False)


def ordered_updates(rows):
    for row in rows:
        if not all(math.isfinite(float(v)) for v in row.values()):
            raise ValueError('nonfinite actual update telemetry')
        if 'actual/completed_training_transitions' not in row or 'actual/optimizer_updates' not in row:
            raise ValueError('actual loss clock and optimizer count required')
    return sorted(rows, key=lambda r: (r['actual/completed_training_transitions'], r['actual/optimizer_updates']))


def validate_actual_update(row, contract):
    ordered_updates([row])
    tolerance = contract['gradient_sum_atol'] + contract['gradient_sum_rtol'] * max(
        abs(row['actual/actor_gradient_norm']),abs(row['actual/actor_gradient_sum_norm']))
    if row['actual/actor_gradient_norm_sum_difference'] > tolerance:
        raise ValueError('actual weighted actor gradient sum violates numerical tolerance')
    if 'ppo' in contract:
        update=row['actual/optimizer_updates'];clock=row['actual/completed_training_transitions']
        if update!=int(update) or not 1<=update<=80 or clock!=3200*math.ceil(update/8):
            raise ValueError('actual loss/Adam clock contract failed')
        scale=contract['demo_coefficient_start']+(contract['demo_coefficient_end']-contract['demo_coefficient_start'])*clock/contract['demo_schedule_transitions']
        if abs(row['effective_lambda_demo']-scale)>1e-6 or row['audit/normalizer_count_delta_from_initializer']!=0:
            raise ValueError('demo schedule or frozen normalizer clock changed')
        expected=row['ppo_loss']+scale*row['demo_action_mse']+contract['keep_coefficient']*row['retention_action_mse']
        if not math.isclose(row['total_loss'],expected,rel_tol=1e-5,abs_tol=1e-6):
            raise ValueError('actual total loss component sum failed')


def _classify(values):
    if len(values) != 3 or any(v not in (0, 1) for v in values):
        return 'unknown'
    return 'stable_success' if values == [1, 1, 1] else 'stable_failure' if values == [0, 0, 0] else 'ambiguous'


def _full_classification(repeats):
    if len(repeats) != 3 or any(set(r) != set('ABCD') for r in repeats):
        raise ValueError('three complete DEV repeats required')
    if any(len(r[g]) != 16 or any(v not in (0, 1) for v in r[g]) for r in repeats for g in 'ABCD'):
        raise ValueError('complete finite binary DEV64 labels required')
    return {f'{g}_{i:02d}': _classify([r[g][i] for r in repeats]) for g in 'ABCD' for i in range(16)}


def mechanism_decision(source_repeats, results, required_train_roots, *, train_root_ids=None, solver_root_ids=None):
    partial = dict(evaluation_status='partial', t1_passed=None, automatic_next_stage=False,
                   published_policy=None, endpoint_transition=32000)
    if len(source_repeats) != 3 or any(a not in results for a in ARMS):
        return partial
    if any(len(results[a].get('DEV', [])) != 3 or len(results[a].get('TRAIN', [])) != 3
           or len(results[a].get('SOLVER_DEV', {})) != 7 for a in ARMS):
        return partial
    baseline = _full_classification(source_repeats)
    expected_train=set(train_root_ids if train_root_ids is not None else results[ARMS[0]]['TRAIN'][0])
    expected_solver=set(solver_root_ids if solver_root_ids is not None else results[ARMS[0]]['SOLVER_DEV'])
    arms = {}
    from .retention_b import warning_cells
    for name in ARMS:
        r = results[name]
        if r.get('checkpoint_transition') != 32000:
            raise ValueError('T1 fixed32000 endpoint required; no best selection')
        roots = list(r['TRAIN'][0])
        if (len(roots) != 8 or set(roots) != expected_train or
                any(set(x) != expected_train or any(v not in (0,1) for v in x.values()) for x in r['TRAIN'])):
            raise ValueError('three complete matching TRAIN8 repeats required')
        if set(r['SOLVER_DEV']) != expected_solver or any(v not in (0,1) for v in r['SOLVER_DEV'].values()):
            raise ValueError('complete matching binary solver7 labels required')
        train = {root: _classify([x[root] for x in r['TRAIN']]) for root in roots}
        dev = _full_classification(r['DEV'])
        gained = [k for k in dev if baseline[k] == 'stable_failure' and dev[k] == 'stable_success']
        lost = [k for k in dev if baseline[k] == 'stable_success' and dev[k] == 'stable_failure']
        warnings = [warning_cells(s, c) for s, c in zip(source_repeats, r['DEV'])]
        confirmed = [g for g in ('B','D') if sum(g in row for row in warnings) >= 2]
        arms[name] = dict(stable_train_success=sum(v == 'stable_success' for v in train.values()),
                          TRAIN_classification=train, DEV_classification=dev,
                          stable_gained=gained, stable_lost=lost, stable_net=len(gained)-len(lost),
                          old_ambiguous=[k for k in dev if baseline[k] == 'stable_success' and dev[k] == 'ambiguous'],
                          mixed_cases=[k for k in dev if dev[k] == 'ambiguous' or baseline[k] == 'ambiguous'],
                          solver_success=sum(v == 1 for v in r['SOLVER_DEV'].values()),
                          warning_cells_by_repeat=warnings, confirmed_regression=confirmed)
    control, demo = (arms[a] for a in ARMS)
    checks = dict(TRAIN_at_least_5=demo['stable_train_success'] >= 5,
                  TRAIN_gain_over_control=demo['stable_train_success'] >= control['stable_train_success'] + 1,
                  required_roots_preserved=all(demo['TRAIN_classification'].get(r) == 'stable_success' for r in required_train_roots),
                  solver_at_least_3=demo['solver_success'] >= 3,
                  solver_not_below_control=demo['solver_success'] >= control['solver_success'],
                  DEV_net_above_control=demo['stable_net'] > control['stable_net'],
                  DEV_old_losses_not_worse=len(demo['stable_lost']) <= len(control['stable_lost']),
                  DEV_old_ambiguous_debt_not_worse=len(demo['old_ambiguous']) <= len(control['old_ambiguous']),
                  retention_guard=not demo['confirmed_regression'] and not control['confirmed_regression'])
    return dict(evaluation_status='complete', t1_passed=all(checks.values()), checks=checks,
                arms=arms, source_DEV_classification=baseline, endpoint_transition=32000,
                automatic_next_stage=False, published_policy=None,
                scope='one-seed mechanism screen; not paper significance or policy adoption',
                mixed_semantics='mixed labels remain ambiguous and are excluded from stable N01/N10; old ambiguous debt is listed separately')


def report(path):
    """Derive a new report from saved receipts; never reverify by running a worker."""
    audit(path)
    plan = read(path)
    root = Path(plan['output'])
    state = read(root/'status.json')
    evaluation = read(root/'evaluation_results.json') if (root/'evaluation_results.json').exists() else {}
    missing, verification = [], {}
    for arm in ARMS:
        directory = root/'training'/arm
        status_path, receipt_path = directory/'status.json', directory/'verification.json'
        if not status_path.exists() or not receipt_path.exists():
            missing.append(f'{arm}: missing completed training status or saved verification')
            continue
        status, receipt = read(status_path), read(receipt_path)
        verification[arm] = dict(path=str(receipt_path), sha256=sha(receipt_path),
                                 phase=receipt.get('phase'), checks=receipt.get('checks', {}))
        if (status.get('phase') != 'completed' or status.get('completed_training_transitions') != 32000
                or receipt.get('phase') != 'passed' or receipt.get('completed_training_transitions') != 32000
                or not receipt.get('checks') or any(v is not True for v in receipt['checks'].values())):
            missing.append(f'{arm}: fixed32000 learner verification not passed')
    arms = evaluation.get('arms', {})
    source = evaluation.get('source_DEV', [])
    for arm in ARMS:
        results = arms.get(arm, {})
        for kind, repeats, count in (('TRAIN', 3, 8), ('DEV', 3, 64)):
            values = results.get(kind, [])
            if len(values) != repeats:
                missing.append(f'{arm}: {kind} requires three complete repeats')
                continue
            if kind == 'TRAIN' and any(len(r) != count or any(v not in (0, 1) for v in r.values()) for r in values):
                missing.append(f'{arm}: incomplete or nonbinary TRAIN labels')
        solver = results.get('SOLVER_DEV', {})
        if len(solver) != 7 or any(v not in (0, 1) for v in solver.values()):
            missing.append(f'{arm}: seven finite SOLVER_DEV labels required')
    if len(source) != 3:
        missing.append('pi0: three complete DEV repeats required')
    decision = dict(evaluation_status='partial', t1_passed=None, automatic_next_stage=False,
                    published_policy=None, endpoint_transition=32000)
    if not missing:
        try:
            decision = mechanism_decision(source, arms, plan['required_train_roots'],
                train_root_ids=[row['root']['root_id'] for row in read(plan['train_roots'])],
                solver_root_ids=[row['root']['root_id'] for row in read(plan['solver_roots'])])
        except (ValueError, KeyError, TypeError) as error:
            missing.append('evaluation completeness: ' + str(error))
    rows = read(root/'costs.json') if (root/'costs.json').exists() else []
    charged = sum(row['charged'] for row in rows)
    split_known = [r for r in rows if 'active_interactions' in r and 'padding_interactions' in r]
    summary = dict(**decision, schema='jit_retention_T1_report_v1',
        execution_status=state.get('phase', 'unknown'), missing_or_invalid_artifacts=missing,
        saved_training_verification=verification, new_physics_control_steps=0, new_training_updates=0,
        final_test_used=False, report_method='saved files only; no worker, optimizer, restore or evaluation invoked',
        costs=dict(charged=charged, prior_charged=plan['budget']['prior_charged'],
                   D2_cumulative=plan['budget']['prior_charged']+charged,
                   stage_cap=plan['budget']['total'], D2_cap=plan['budget']['hard_cap'],
                   active_known=sum(r['active_interactions'] for r in split_known),
                   padding_known=sum(r['padding_interactions'] for r in split_known),
                   charged_without_active_padding_split=sum(r['charged'] for r in rows if r not in split_known),
                   failed_or_unfinished_reserved_charged=sum(r['charged'] for r in rows if r.get('phase') != 'completed')),
        claim_boundary='One fixed endpoint, one pilot seed, finite saved conditions; not paper significance, policy adoption or automatic extension.')
    ordinal = 1
    while (root/f'report_{ordinal:04d}').exists():
        ordinal += 1
    output = root/f'report_{ordinal:04d}'
    output.mkdir(exist_ok=False)
    summary['report_directory'] = str(output)
    write(output/'summary.json', summary)
    write(output/'source_receipts.json', dict(plan=str(Path(path).resolve()), plan_sha256=sha(path),
        status=str(root/'status.json'), status_sha256=sha(root/'status.json'),
        evaluation_results_sha256=sha(root/'evaluation_results.json') if (root/'evaluation_results.json').exists() else None,
        costs_sha256=sha(root/'costs.json') if (root/'costs.json').exists() else None))
    return summary


def dry_run(path):
    result=audit(path)
    result.update(worker_processes=0,simulation_steps=0,training_updates=0,
                  formal_stages=[dict(arm=a,transitions=32000) for a in ARMS],
                  engineering_stages=['reset_GPU200','engineering_PPO_keep3200','engineering_PPO_demo_keep3200'],
                  evaluation_jobs=evaluation_jobs(),source_DEV_first='reuse frozen pi0 first',
                  data_counts=dict(demo_old=525,demo_new=228,keep=6289))
    return result


class Run:
    def __init__(self,path):
        self.path=Path(path).resolve();self.p=read(self.path);self.root=Path(self.p['output'])
        self.code=Path(self.p['code']);self.ledger=CostLedger(self.p['budget']);self.started=time.time()
        self.formal={a:0 for a in ARMS};self.engineering={a:0 for a in ARMS}
        self.env=dict(os.environ,PYTHONPATH=str(self.code/'JIT/src'),JAX_PLATFORMS='cuda,cpu',
            XLA_PYTHON_CLIENT_PREALLOCATE='false',JIT_AUTO_PUBLISH='0',OMP_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1',JAX_COMPILATION_CACHE_DIR='/home/qy/DVGC/JIT/runs/jax_cache')

    def status(self,phase,stage,**extra):
        write(self.root/'costs.json',self.ledger.rows)
        write(self.root/'status.json',dict(phase=phase,stage=stage,pid=os.getpid(),updated_unix=time.time(),
            started_unix=self.started,deadline_unix=self.started+self.p['budget']['wall_seconds'],
            charged_interactions=self.ledger.charged,maximum_new_interactions=self.p['budget']['total'],
            cumulative_charged=self.p['budget']['prior_charged']+self.ledger.charged,
            formal_training_transitions=sum(self.formal.values()),formal_by_arm=self.formal,
            engineering_training_transitions=sum(self.engineering.values()),engineering_by_arm=self.engineering,
            E_updates=0,G_updates=0,automatic_next_stage=False,published_policy=None,**extra))

    def remaining(self):
        left=self.started+self.p['budget']['wall_seconds']-time.time()
        if left<=0:raise TimeoutError('T1 independent6h wall budget exhausted')
        return left

    def child(self,name,category,maximum,command,output,request=None):
        from .retention_b import bounded_worker
        output=Path(output)
        if output.exists():raise ValueError('no overwrite or implicit child retry')
        row=self.ledger.reserve(name,category,maximum)
        self.status('running',name)
        dispatch=dict(phase='running',name=name,category=category,reservation=maximum,output=str(output))
        if request is not None:dispatch.update(request=str(Path(request).resolve()),request_sha256=sha(request))
        write(self.root/'active_dispatch.json',dispatch)
        with (self.root/(name+'.log')).open('x') as stream:
            bounded_worker(command,cwd=self.code,env=self.env,stdout=stream,timeout=self.remaining())
        receipt=read(output/'status.json')
        self.ledger.complete(row,receipt)
        write(self.root/'active_dispatch.json',{**dispatch,'phase':'completed','receipt_sha256':sha(output/'status.json')})
        self.status('running',name)
        return output

    def worker(self,kind,category,maximum,arm=None):
        name=kind+('_'+arm if arm else '')
        output=self.root/('training' if kind=='train' else kind)/(arm or 'check')
        req=self.root/(name+'_request.json')
        write(req,dict(plan=str(self.path),plan_sha256=sha(self.path),kind=kind,arm=arm,output=str(output)))
        return self.child(name,category,maximum,[PY,str(self.code/'JIT/cli/run_retention_t1.py'),
            'worker','--request',str(req)],output,req)


def _reference(plan,arm):
    node=read(Path(plan['output'])/'training'/arm/'checkpoint_032000.json')
    if node['transition']!=32000 or sha(node['inference'])!=node['inference_sha256']:
        raise ValueError('fixed endpoint inference identity drift')
    policy=plan['models']['pi0']['policy']
    if node['normalizer_sha256']!=policy['normalizer_sha256']:raise ValueError('evaluation normalizer drift')
    return dict(path=node['inference'],sha256=node['inference_sha256'],
                source_actor_sha256=policy['actor_sha256'],normalizer_sha256=policy['normalizer_sha256'])


def _evaluate_dev(run,job):
    from .retention_b import full_results
    from .retention_repair_execution import verify
    name=f"{job['arm']}_DEV_repeat{job['repeat']}"
    batch=copy.deepcopy(run.p['batches'][0]);spec=read(batch['spec'])
    if job['arm']!='pi0':spec['warmup_initializer']=_reference(run.p,job['arm'])
    else:spec.pop('warmup_initializer',None)
    sp=run.root/(name+'_spec.json');write(sp,spec)
    out=run.child(name,job['category'],job['reservation'],[PY,str(run.code/'JIT/cli/run_pulse_exploration.py'),
        '--mode','collect','--spec',str(sp),'--output',str(run.root/name)],run.root/name)
    batch.update(name=name,spec=str(sp));verify(batch,run.root)
    result=full_results(run.p,out)
    write(out/'evaluation_receipt.json',dict(**result,role='DEV',repeat=job['repeat'],arm=job['arm'],
        prefixes_sha256=sha(out/'prefixes.npz'),spec_sha256=sha(sp),status_sha256=sha(out/'status.json')))
    return result['labels']


def _evaluate_roots(run,job):
    role=job['kind'];entries=read(run.p['train_roots'] if role=='TRAIN' else run.p['solver_roots'])
    labels={};receipts=[];reference=_reference(run.p,job['arm'])
    for ordinal,entry in enumerate(entries):
        root=entry['root'];lane=entry['lane'];name=f"{job['arm']}_{role}_repeat{job['repeat']}_root{ordinal:02d}"
        rowpath=run.root/(name+'_rows.json')
        write(rowpath,[{**root,'index':i,'candidate_id':i} for i in range(17)])
        model=run.p['models']['pi0']
        spec=dict(bank=model['bank'],proposer=model['proposer'],order=[model['proposer']],horizon=400,
            budget=6800,candidates=str(rowpath),full_matrix=True,record_actor_preobservations=True,
            record_retention_diagnostics=True,reward_mode='original_all_phases',
            success_criterion='stable_forward_recovery',preserve_snapshot_episode_context=True,
            suffix_rng_count=17,suffix_rng_indices=list(range(17)),warmup_initializer=reference)
        sp=run.root/(name+'_spec.json');write(sp,spec)
        out=run.child(name,job['category'],6800,[PY,str(run.code/'JIT/cli/run_pulse_exploration.py'),
            '--mode','evaluate','--spec',str(sp),'--output',str(run.root/name)],run.root/name)
        results=read(out/'results.json')
        if len(results)!=17 or any(x.get('label') not in (0,1) for x in results):
            raise ValueError('incomplete binary full17 snapshot evaluation')
        chosen=results[lane]
        checked_traces={}
        endpoint=read(run.root/'training'/job['arm']/'checkpoint_032000.json')
        for r in results:
            attempt=r['attempts'][0]
            if (attempt['actor_sha256']!=endpoint['actor_sha256'] or
                attempt['normalizer_sha256']!=endpoint['normalizer_sha256'] or
                attempt['snapshot_context_sha256']!=root['snapshot_context_sha256']):
                raise ValueError('snapshot endpoint actor/normalizer/context drift')
            trace=attempt['trace']
            if trace not in checked_traces:
                if sha(trace)!=attempt['trace_sha256']:raise ValueError('snapshot trace drift')
                with np.load(trace,allow_pickle=False) as archive:checked_traces[trace]={k:archive[k] for k in archive.files}
            tape=checked_traces[trace];idx=attempt['trace_lane'];mask=tape['mask'][:,idx].astype(bool)
            if not mask.any() or np.any(mask[1:]&~mask[:-1]) or int(mask.sum())!=attempt['steps']:
                raise ValueError('noncontiguous or incomplete snapshot trace')
            for field in ('qpos','qvel','action','reward','actor_observation_before'):
                if not np.isfinite(tape[field][:,idx][mask]).all():raise ValueError('nonfinite snapshot evaluation')
            last=np.flatnonzero(mask)[-1]
            success=bool(tape['valid_contact'][last,idx]) and not bool(tape['physical_failure'][last,idx])
            if success!=(attempt['label']==1) or attempt['label']!=r['label']:
                raise ValueError('snapshot saved trace/label endpoint mismatch')
        labels[root['root_id']]=chosen['label']
        receipts.append(dict(root_id=root['root_id'],role=role,lane=lane,label=chosen['label'],
            all_lane_labels=[r['label'] for r in results],results=str(out/'results.json'),
            results_sha256=sha(out/'results.json'),spec_sha256=sha(sp),attempt=chosen['attempts'][0]))
    write(run.root/f"{job['arm']}_{role}_repeat{job['repeat']}_receipt.json",receipts)
    return labels


def tensorboard_receipt(root,port,required_runs=(),timeout=30):
    """Require HTTP scalar samples for the exact current run, not merely event files."""
    import urllib.request
    import urllib.parse
    deadline=time.monotonic()+timeout;last=None
    while time.monotonic()<deadline:
        try:
            base=f'http://localhost:{port}'
            with urllib.request.urlopen(base+'/',timeout=3) as response:code=response.status
            with urllib.request.urlopen(base+'/data/plugin/scalars/tags',timeout=3) as response:tags=json.load(response)
            samples={}
            for run_name in required_runs:
                samples[run_name]={}
                for tag in ('training/reward_per_transition','total_loss'):
                    query=urllib.parse.urlencode(dict(run=run_name,tag=tag))
                    with urllib.request.urlopen(base+'/data/plugin/scalars/scalars?'+query,timeout=3) as response:
                        values=json.load(response)
                    if not values or not any(v[1]>0 and math.isfinite(v[2]) for v in values):
                        raise ValueError('current run reward/loss scalar not loaded: '+run_name+'/'+tag)
                    samples[run_name][tag]=dict(count=len(values),last=values[-1])
            result=dict(phase='passed',url=base,http_status=code,required_runs=list(required_runs),
                reward_scalars=samples,tags=tags,checked_unix=time.time())
            write(Path(root)/'tensorboard_receipt.json',result);return result
        except Exception as error:
            last=repr(error);time.sleep(.5)
    raise RuntimeError('T1 TensorBoard HTTP/reward verification failed '+str(last))


def run(path):
    audit(path);p=read(path);root=Path(p['output'])
    if read(root/'status.json')['phase']!='prepared':raise ValueError('no implicit T1 resume or retry')
    # Refuse an occupied port: do not silently use/stop another project's service.
    import socket
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',p['tensorboard_port']))
    with (root/'execution.claim').open('x') as stream:
        json.dump(dict(plan=str(Path(path).resolve()),plan_sha256=sha(path),pid=os.getpid(),started_unix=time.time()),stream)
    r=Run(path);r.status('running','launch')
    write(root/'ACTIVE_RUN.json',dict(execution=str(root/'status.json'),lineage=str(root/'status.json'),name='T1 paired finite PPO'))
    write(root/'TRAINING_ACTIVE_RUN.json',dict(execution=str(root/'training/status.json'),name='T1 training stage'))
    (root/'training').mkdir(exist_ok=False)
    write(root/'training/status.json',dict(phase='prepared',stage='T1_training'))
    children={}
    try:
        with (root/'watcher.log').open('x') as stream:
            children['watcher']=subprocess.Popen([PY,str(r.code/'JIT/cli/watch_run_errors.py'),
                '--active-run',str(root/'ACTIVE_RUN.json'),'--active-run',str(root/'TRAINING_ACTIVE_RUN.json'),
                '--state-dir',str(root/'notifications')],cwd=r.code,env=r.env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        with (root/'tensorboard.log').open('x') as stream:
            children['tensorboard']=subprocess.Popen([PY,'-m','tensorboard.main','--logdir',str(root/'tensorboard'),
                '--port',str(p['tensorboard_port']),'--host','0.0.0.0','--reload_interval','1'],
                cwd=r.code,env=r.env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        write(root/'launch.json',dict(pid=os.getpid(),started_unix=r.started,plan=str(Path(path).resolve()),
            plan_sha256=sha(path),watcher_pid=children['watcher'].pid,tensorboard_pid=children['tensorboard'].pid,
            tensorboard_url=f"http://localhost:{p['tensorboard_port']}"))
        tensorboard_receipt(root,p['tensorboard_port'],timeout=min(30,r.remaining()))
        r.worker('reset','reset_GPU',200)
        for arm in ARMS:
            out=r.worker('engineering','engineering_ppo',3200,arm)
            if read(out/'verification.json')['phase']!='passed':raise ValueError('engineering PPO gate failed')
            r.engineering[arm]=3200
            tensorboard_receipt(root,p['tensorboard_port'],required_runs=['engineering_'+a for a in ARMS if r.engineering[a]],timeout=min(30,r.remaining()))
        write(root/'training/status.json',dict(phase='running',stage='T1_training',completed_training_transitions=0))
        for arm in ARMS:
            out=r.worker('train','PPO',32000,arm)
            if read(out/'verification.json')['phase']!='passed':raise ValueError('formal learner verification failed')
            r.formal[arm]=32000
            tensorboard_receipt(root,p['tensorboard_port'],required_runs=[a for a in ARMS if r.formal[a]],timeout=min(30,r.remaining()))
            write(root/'training/status.json',dict(phase='completed' if all(r.formal.values()) else 'running',
                stage='T1_training',completed_training_transitions=sum(r.formal.values()),by_arm=r.formal,
                updated_unix=time.time(),selection='fixed32000 each arm'))
        from .retention_b import full_results
        evaluation=dict(source_DEV=[full_results(p,p['reused_zero']['DEV']['pi0'])['labels']],
            arms={a:dict(checkpoint_transition=32000,DEV=[],TRAIN=[],SOLVER_DEV={}) for a in ARMS})
        write(root/'evaluation_results.json',evaluation)
        for job in p['evaluation_jobs']:
            labels=_evaluate_dev(r,job) if job['kind']=='DEV' else _evaluate_roots(r,job)
            if job['arm']=='pi0':evaluation['source_DEV'].append(labels)
            elif job['kind']=='SOLVER_DEV':evaluation['arms'][job['arm']]['SOLVER_DEV']=labels
            else:evaluation['arms'][job['arm']][job['kind']].append(labels)
            write(root/'evaluation_results.json',evaluation)
        for arm in ARMS:r.worker('panel','TRAIN_panels',1600,arm)
        decision=mechanism_decision(evaluation['source_DEV'],evaluation['arms'],p['required_train_roots'],
            train_root_ids=[x['root']['root_id'] for x in read(p['train_roots'])],
            solver_root_ids=[x['root']['root_id'] for x in read(p['solver_roots'])])
        write(root/'mechanism_decision.json',decision)
        if decision['evaluation_status']!='complete':raise ValueError('incomplete fixed evaluation schedule')
        r.status('completed','T1_complete',evaluation_status='complete',t1_passed=decision['t1_passed'],wall_seconds=time.time()-r.started)
        report(path)
    except BaseException as error:
        r.status('failed','T1',error=repr(error),wall_seconds=time.time()-r.started)
        current=read(root/'training/status.json')
        if current['phase']!='completed':write(root/'training/status.json',{**current,'phase':'failed','error':repr(error)})
        raise
