"""Prepare two isolated, bounded pulse-mechanism arms without simulation."""
from __future__ import annotations

import math
from pathlib import Path
import re

from .exploration_loop import read, write
from .gated_execution import _validate
from .probe_bank import _file_sha, lock_probe_bank
from .pulse_exploration import budget_contract


EVENTS = ('start', 'liftoff', 'apex', 'descent')


def prepare_mechanism(*, initial_frozen_policy, task_template_bank,
                      jump_start_state_sha256, output, seed, num_envs=128,
                      rounds=24, amplitudes=(.10, .15),
                      max_actual_interactions=2_500_000, recovery_seconds=.5,
                      repo=None, python='/home/qy/mujoco_playground/.venv/bin/python'):
    """Lock configuration and source code; write a sequential CPU supervisor plan.

    The task template contributes named geometry/start identities only. Its
    policies, support, labels and execution results never enter either arm.
    Each arm creates its own new nominal support at execution time.
    """
    from .unified_policy_freeze import load_frozen_unified_manifest

    root = Path(output).resolve()
    if root.exists():
        raise FileExistsError(root)
    repo = Path(repo).resolve() if repo is not None else Path(__file__).resolve().parents[3]
    if not Path(python).is_absolute():
        raise ValueError('absolute Python executable required')
    for key, value in [('seed', seed), ('num_envs', num_envs), ('rounds', rounds),
                       ('max_actual_interactions', max_actual_interactions)]:
        if type(value) is not int or value < (0 if key == 'seed' else 1):
            raise ValueError(f'positive integer {key} required')
    amplitudes = list(amplitudes)
    if (not amplitudes or any(type(a) not in (int, float) or not math.isfinite(a)
                              or not 0 < a <= 1 for a in amplitudes)
            or len(set(amplitudes)) != len(amplitudes)):
        raise ValueError('distinct finite amplitudes in (0, 1] required')
    if not re.fullmatch('[a-f0-9]{64}', str(jump_start_state_sha256)):
        raise ValueError('declared fixed jump-start state SHA-256 required')
    if (type(recovery_seconds) not in (int, float) or not math.isfinite(recovery_seconds)
            or recovery_seconds <= 0):
        raise ValueError('positive finite recovery duration required')

    source_path = Path(initial_frozen_policy).resolve()
    template_path = Path(task_template_bank).resolve()
    source = load_frozen_unified_manifest(source_path)['policy']
    config_path = Path(source['formal_config']).resolve()
    config = read(config_path)
    inputs = {str(p): _file_sha(p) for p in (source_path, template_path, config_path)}

    def resolve_input(path):
        path = Path(path)
        return path.resolve() if path.is_absolute() else (repo / path).resolve()

    phase_configs = {}
    for phase in ('up', 'down'):
        path = resolve_input(config['inputs'][phase + '_config_path'])
        inputs[str(path)] = _file_sha(path)
        phase_configs[phase] = read(path)
        for key in ('xml_path', 'reference_path'):
            asset = phase_configs[phase].get('model', {}).get(key)
            if asset:
                resolved = resolve_input(asset)
                inputs[str(resolved)] = _file_sha(resolved)
    descent = phase_configs['down']['descent']
    # Control period is the unchanged JIT task contract, not a CLI override.
    from .constants import CTRL_DT
    recovery_ticks = recovery_seconds / CTRL_DT
    if (not math.isclose(recovery_ticks, round(recovery_ticks), abs_tol=1e-9)
            or config.get('success_criterion') != 'stable_forward_recovery'
            or descent.get('continuous_stability') is not True
            or descent.get('recovery_ticks') != round(recovery_ticks)
            or descent.get('min_post_contact_forward_progress') != 0):
        raise ValueError('frozen source recovery contract differs from declared continuous duration')
    checkpoint = Path(source['checkpoint']).resolve()
    for name in ('identity.json', 'payload.pkl'):
        path = checkpoint / name
        inputs[str(path)] = _file_sha(path)
    if source.get('source_formal_report'):
        path = Path(source['source_formal_report']).resolve()
        inputs[str(path)] = _file_sha(path)

    inherited_task = read(template_path)['task']
    task = {key: inherited_task[key] for key in (
        'xml_sha256', 'start_contract_sha256', 'centerline_sha256', 'resolution_sha256')}
    if task['xml_sha256'] != source['xml_sha256']:
        raise ValueError('task template and initial policy model differ')
    task.update(success_criterion='stable_forward_recovery',
                continuation_start_semantics='fresh_continuation_v1',
                recovery_seconds=recovery_seconds,
                down_config_file_sha256=inputs[str(resolve_input(config['inputs']['down_config_path']))])
    source_locks = {str(path.resolve()): _file_sha(path)
                    for base in (repo / 'JIT/src', repo / 'JIT/cli')
                    for path in sorted(base.rglob('*.py'))}
    # XML can reference meshes/plugins; lock its asset closure as in the existing
    # preparation workflow. These are identities, never historical run labels.
    for base in (repo / 'assets', repo / 'JIT/configs'):
        for path in sorted(base.rglob('*')):
            if path.is_file():
                source_locks[str(path.resolve())] = _file_sha(path)
    if not source_locks:
        raise ValueError('runtime source locks required')

    event_schedule = list(EVENTS) * len(amplitudes)
    delta_schedule = [[float(a)] * 4 for a in amplitudes for _ in EVENTS]
    common = dict(schema='jit_short_pulse_delayed_quality_v1',
        proposer=source['name'], order=[source['name']], role='TRAIN', final_test_used=False,
        repo=str(repo), python=python, seed=seed, rounds=rounds, num_envs=num_envs, round_index=0,
        horizon=400, pulse_steps=3, delta_limit=[float(amplitudes[0])] * 4,
        pulse_start_schedule=[0], pulse_event_schedule=event_schedule, pulse_descent_clearance=.10,
        delta_limit_schedule=delta_schedule, iteration_mode='current_policy_only_v1',
        training_mode='fixed_policy', quality_mode='current_policy',
        success_criterion='stable_forward_recovery', reward_mode='original_all_phases',
        recovery_seconds=recovery_seconds, jump_start_state_sha256=jump_start_state_sha256,
        bootstrap_config=str(config_path), policy_steps=128000,
        pending_fraction=.5, minimum_retention=.9, retention_samples_per_phase=8,
        seed_stride=2, learning_rate=3e-5, epochs=4, minibatch_size=128,
        gamma=1., gae_lambda=1., clip=.2, target_kl=.01,
        max_grad_norm=1., value_coefficient=.5, entropy_coefficient=.001,
        reward_weights=dict(novelty=.25, success=1., failure=1.),
        maximum_actual_interactions=max_actual_interactions,
        gate=dict(kind='gpu_idle'), wait_timeout_seconds=72 * 3600,
        stage_timeout_seconds=7200, source_locks=source_locks)
    budgets = budget_contract(common, 1)
    common['maximum_interactions'] = budgets['maximum_interactions']
    arm_reservation = min(max_actual_interactions, budgets['maximum_interactions'])
    root.mkdir(parents=True, exist_ok=False)
    stages, plan_inputs = [], dict(inputs)
    for arm in ('learned', 'fixed_random'):
        directory = root / arm
        bank_path = directory / 'initial_bank.json'
        lock_probe_bank(dict(version='pulse_mechanism_' + arm, task=task,
            max_ticks=400, max_candidates_per_process=num_envs,
            label_interaction_budget=max_actual_interactions,
            members=[dict(frozen_policy=str(source_path), roles=['proposer', 'evaluator'])]), bank_path)
        arm_inputs = {**inputs, str(bank_path): _file_sha(bank_path)}
        spec_path = directory / 'spec.json'
        controller_mode = 'learned_residual' if arm == 'learned' else 'fixed_random'
        write(spec_path, dict(common, bank=str(bank_path), controller_mode=controller_mode, input_files=arm_inputs))
        plan_inputs.update(arm_inputs)
        plan_inputs[str(spec_path)] = _file_sha(spec_path)
        stages.append(dict(name=arm, execution_backend='cpu',
            argv=[python, 'JIT/cli/run_pulse_exploration.py', '--mode', 'loop',
                  '--spec', str(spec_path), '--output', str(directory / 'lineage')],
            cwd=str(repo), env=dict(PYTHONPATH=str(repo / 'JIT/src'),
                                   JAX_PLATFORMS='cpu', JIT_AUTO_PUBLISH='0'),
            max_interactions=arm_reservation,
            timeout_seconds=common['wait_timeout_seconds'] + rounds * 3 * 7200 + 600))
    plan = dict(schema='jit_gated_plan_v1',
        gate=dict(kind='user_requested_immediate', authorization='start_without_gpu_idle_wait'),
        wait_timeout_seconds=72 * 3600, max_interactions=2 * arm_reservation,
        input_files=plan_inputs, source_locks=source_locks, stages=stages)
    _validate(plan)
    write(root / 'plan.json', plan)
    write(root / 'budget.json', dict(schema='jit_pulse_mechanism_budget_v1',
        per_arm_theoretical=budgets, per_arm_actual_cap=max_actual_interactions,
        plan_reserved_interactions=2 * arm_reservation, requested_rounds=rounds,
        event_amplitude_cycle=list(zip(event_schedule, delta_schedule)),
        source_actor_sha256=source['actor_sha256'], source_normalizer_sha256=source['normalizer_sha256'],
        source=str(source_path), task_template=str(template_path),
        task_template_use='geometry and fixed-start identities only; no members, support, labels or costs imported',
        data_role='TRAIN', final_test_used=False, environment_interactions=0,
        status='prepared_not_launched'))
    write(root / 'ACTIVE_RUN.json', {'execution': str(root / 'execution/status.json')})
    (root / 'INDEX.md').write_text(
        '# Fixed-policy pulse mechanism comparison\n\n'
        'Prepared only; no simulation or training was launched by preparation.\n\n'
        f'Common frozen Actor: `{source["actor_sha256"]}`. Each arm owns a singleton bank, '
        'fresh nominal support, arrivals, visited ledger, optimizer state and results.\n\n'
        f'Two arms: learned residual and fixed uniform random residual. Each requests {rounds} rounds '
        f'of {num_envs} candidates, with an actual-interaction cap of {max_actual_interactions:,}. '
        'The eight-combination default schedule crosses both amplitudes with all four events. '
        'Both use the same original task reward and continuous stable recovery endpoint.\n\n'
        'The top-level CPU supervisor starts without GPU allocation; each GPU child waits for '
        'the gpu_idle gate. The plan is sequential and never retries a child. Runtime source '
        'drift invalidates the locks; prepare a fresh directory after code changes.\n\n'
        '[Execution plan](plan.json), [budget](budget.json), [learned declaration](learned/spec.json), '
        '[random declaration](fixed_random/spec.json), [watcher pointer](ACTIVE_RUN.json).\n\n'
        'Launch the plan with `cli/run_gated_plan.py --plan <root>/plan.json '
        '--output-dir <root>/execution --wait`. Start and verify the required active-run watcher '
        'alongside execution. Preparation does not claim notification delivery.\n', encoding='utf-8')
    return plan
