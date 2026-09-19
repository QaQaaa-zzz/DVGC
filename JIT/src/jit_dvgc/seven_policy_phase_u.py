"""Finite sequential Phase U retraining and seven-policy development comparison.

All policy membership is declared in data. Historical checkpoints remain immutable;
new Phase U identities are evaluated through a proven compatible full-task runtime.
"""
from __future__ import annotations

import copy
import csv
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

from .batched_pulse_comparison import batch_plan, verify_batch
from .evidence_integrity import canonical_sha256
from .jump_evidence_validation import read, write, file_sha

TARGET = 14_991_360
ONSETS = (0, 5, 10, 15)
SCHEDULE = (0, 737_280, 2_998_272, 7_495_680, 11_993_088, TARGET)
CONTRACT_FIELDS = ('horizon', 'pulse_steps', 'pulse_batch_mode', 'full_episode_rollout',
                   'controller_mode', 'delta_limit', 'success_criterion', 'reward_mode')


def _lock(paths, locks):
    for path in paths:
        path = str(Path(path).resolve())
        digest = file_sha(path)
        if path in locks and locks[path] != digest:
            raise ValueError('locked input drift: ' + path)
        locks[path] = digest


def _verify_locks(locks):
    for path, digest in locks.items():
        if file_sha(path) != digest:
            raise ValueError('locked input drift: ' + path)


def _safe_key(key):
    if not isinstance(key, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', key):
        raise ValueError('method/run key must be a safe path component')
    return key


def _execution_identity(repository):
    """Freeze the explicitly selected runnable code tree, including snapshots."""
    repository = Path(repository).resolve()
    required = ('JIT/cli/run_seven_policy_phase_u.py', 'JIT/cli/run_pulse_exploration.py',
                'JIT/src/jit_dvgc/seven_policy_phase_u.py', 'JIT/src/jit_dvgc/formal_training.py',
                'JIT/src/jit_dvgc/config.py', 'JIT/src/jit_dvgc/rsi_comparison.py')
    missing = [relative for relative in required if not (repository/relative).is_file()]
    if missing:
        raise ValueError('execution repository missing required entrypoint/module: ' + ', '.join(missing))
    files = sorted((repository/'JIT/src/jit_dvgc').rglob('*.py'))
    files += sorted((repository/'JIT/cli').glob('*.py'))
    locks = {}
    _lock(files, locks)
    relative_hashes = {str(Path(path).relative_to(repository)):sha for path,sha in locks.items()}
    commit = None
    root = subprocess.run(['git', '-C', str(repository), 'rev-parse', '--show-toplevel'],
                          capture_output=True, text=True, timeout=10)
    if root.returncode == 0 and Path(root.stdout.strip()).resolve() == repository:
        revision = subprocess.run(['git', '-C', str(repository), 'rev-parse', 'HEAD'],
                                  capture_output=True, text=True, check=True, timeout=10)
        commit = revision.stdout.strip()
    return dict(schema='jit_execution_code_identity_v1', repository=str(repository),
                git_commit=commit, code_tree_sha256=canonical_sha256(relative_hashes),
                input_files=locks, authority='Exact file hashes; git commit is provenance only')


def prepare_experiment(sources, historical_config, previous, output, *, training_seeds,
                       condition_seeds, execution_repository, python=None):
    """Lock four declared sources, three Actor warm starts and four fixed windows."""
    from .phase_u_warm_start import load_phase_u_actor_initialization
    from .iterative_probe_training import load_phase_initializer
    sources, historical_config, previous, output = [Path(p).resolve() for p in
                                                    (sources, historical_config, previous, output)]
    if output.exists():
        raise FileExistsError(output)
    execution = _execution_identity(execution_repository)
    declaration = read(sources)
    rows = declaration.get('sources', [])
    if declaration.get('schema') != 'jit_seven_policy_sources_v1' or len(rows) != 4:
        raise ValueError('exactly four source declarations required')
    if sum(row.get('warm_start') is True for row in rows) != 3:
        raise ValueError('exactly three warm-start descendants required')
    if (len(training_seeds) != 3 or len(condition_seeds) != 4 or
            any(type(seed) is not int or not 0 <= seed < 2**32 - 1000
                for seed in [*training_seeds, *condition_seeds])):
        raise ValueError('three training and four valid condition seeds required')
    source_spec = read(previous/'spec.json')
    if read(previous/'status.json').get('phase') != 'completed':
        raise ValueError('previous comparison must be completed')
    historical = read(historical_config)
    if historical.get('schema') != 'jit_phase_u_formal_v4':
        raise ValueError('historical resolved Phase U v4 config required')
    locks = dict(execution['input_files'])
    _lock((sources, historical_config, previous/'spec.json', previous/'status.json'), locks)
    for asset in ('xml', 'reference'):
        asset_path = (Path(execution['repository'])/historical['model'][asset+'_path']).resolve()
        if file_sha(asset_path) != historical['model'][asset+'_sha256']:
            raise ValueError('historical physical asset drift: ' + str(asset_path))
        _lock((asset_path,), locks)
    for path, digest in source_spec.get('input_files', {}).items():
        if file_sha(path) != digest:
            raise ValueError('previous source input drift: ' + path)
        locks[str(Path(path).resolve())] = digest
    methods, arms, configs = [], [], []
    common = None
    common_runtime = None
    identities = []
    for row in rows:
        allowed = {'key', 'label', 'template_path', 'frozen_policy', 'warm_start', 'descendant_key'}
        if set(row) - allowed or not allowed - {'descendant_key'} <= row.keys():
            raise ValueError('invalid source declaration fields')
        if type(row['warm_start']) is not bool:
            raise ValueError('warm_start must be boolean')
        key = _safe_key(row['key'])
        template_path = Path(row['template_path']).resolve()
        template = read(template_path)
        contract = {field: template[field] for field in CONTRACT_FIELDS}
        runtime = {k:v for k,v in template.items() if k not in
                   {'bank', 'proposer', 'phase_policy', 'seed', 'num_envs', 'round_index', 'pulse_start_schedule'}}
        if common_runtime is None:
            common_runtime = runtime
        if runtime != common_runtime:
            raise ValueError('source evaluation runtime semantics differ')
        if common is None:
            common = contract
        if contract != common:
            raise ValueError('source evaluation runtime contracts differ')
        if (contract['pulse_steps'] != 3 or contract['delta_limit'] != [.25]*4 or
                contract['controller_mode'] != 'fixed_random' or not contract['full_episode_rollout'] or
                contract['success_criterion'] != 'stable_forward_recovery' or contract['horizon'] <= 17):
            raise ValueError('fixed three-step full-task recovery contract required')
        bank_path = Path(template['bank']).resolve()
        members = [m for m in read(bank_path)['members'] if m['name'] == template['proposer']]
        if len(members) != 1:
            raise ValueError('source policy must resolve exactly once')
        member = members[0]
        frozen_path = Path(row['frozen_policy']).resolve()
        if row['warm_start']:
            load_phase_u_actor_initialization(frozen_path)
        policy = read(frozen_path)['policy']
        for field in ('checkpoint', 'formal_config', 'xml_sha256', 'actor_frame_fields',
                      'actor_task_fields', 'action_order', 'payload_sha256', 'actor_sha256', 'normalizer_sha256'):
            if policy.get(field) != member['policy'].get(field):
                raise ValueError('source bank/frozen policy identity differs: ' + field)
        checkpoint = Path(policy['checkpoint']).resolve()
        _lock((template_path, bank_path, frozen_path, member['frozen_policy'], policy['formal_config'], policy['source_formal_report'],
               checkpoint/'identity.json', checkpoint/'payload.pkl'), locks)
        for artifact in (template, read(policy['formal_config'])):
            _verify_locks(artifact.get('input_files', {}))
            locks.update(artifact.get('input_files', {}))
        if template.get('phase_policy'):
            if row['warm_start']:
                raise ValueError('warm start must identify the declared frozen unified source')
            phase = template['phase_policy']
            _verify_locks(phase['input_files'])
            _lock((phase['source_phase_config'], Path(phase['source_checkpoint'])/'identity.json',
                   Path(phase['source_checkpoint'])/'payload.pkl'), locks)
            locks.update(phase['input_files'])
            payload = load_phase_initializer(phase)
            identity = {field:list(getattr(payload.identity, field)) for field in
                        ('actor_frame_fields', 'actor_task_fields', 'action_order')}
            identity['xml_sha256'] = payload.identity.xml_sha256
            if canonical_sha256(read(phase['source_phase_config'])) != canonical_sha256(historical):
                raise ValueError('historical Phase U evaluation config differs')
        else:
            identity = {field:policy[field] for field in
                        ('xml_sha256', 'actor_frame_fields', 'actor_task_fields', 'action_order')}
        identities.append(identity)
        methods.append(dict(key=key, label=row['label'], template=template, source_frozen_policy=str(frozen_path)))
        if row['warm_start']:
            descendant = _safe_key(row['descendant_key'])
            run_id = _safe_key(output.name + '_' + descendant)
            config = copy.deepcopy(historical)
            config['ppo'].update(requested_transitions=TARGET, seed=training_seeds[len(arms)], num_evals=611)
            config['formal'].update(checkpoint_transitions=list(SCHEDULE), fixed_evaluation_transitions=list(SCHEDULE[1:]),
                                    resume_semantics='parameter_warm_start_optimizer_reset')
            config['initialization'] = dict(actor='warm_start_frozen_development', critic='fresh', optimizer='fresh',
                                             source_frozen_policy=str(frozen_path))
            config['training_reference'] = dict(resolved_config=str(historical_config), sha256=file_sha(historical_config))
            config['run_declaration'] = dict(run_id=run_id)
            config_path = output/'configs'/f'{descendant}.json'
            arms.append(dict(key=descendant, source=key, config=str(config_path), run_id=run_id,
                             frozen_policy=str(frozen_path), run_dir=str(output/'training'/run_id),
                             output_manifest=str(output/'frozen'/f'{descendant}.json')))
            configs.append((config_path, config))
    if not all(identity == identities[0] for identity in identities):
        raise ValueError('source observation/action/XML identities differ')
    if identities[0]['xml_sha256'] != historical['model']['xml_sha256']:
        raise ValueError('source and historical Phase U XML differ')
    for arm in arms:
        parent = next(m for m in methods if m['key'] == arm['source'])
        methods.append(dict(key=arm['key'], label=parent['label']+' + Phase U', source=arm['source'],
                            descendant_manifest=arm['output_manifest']))
    if len({m['key'] for m in methods}) != 7:
        raise ValueError('seven unique source/descendant keys required')
    output.mkdir(parents=True, exist_ok=False)
    for path, config in configs:
        path.parent.mkdir(exist_ok=True)
        write(path, config)
        _lock((path,), locks)
    write(output/'source_lock.json', dict(schema='jit_seven_policy_source_lock_v1', input_files=locks,
                                         source_identities=identities))
    horizon = common['horizon']
    diagnostic_max = 3 * 2 * len(SCHEDULE[1:]) * len(historical['ppo']['held_out_seeds']) * historical['ppo']['episode_horizon']
    spec = dict(schema='jit_seven_policy_phase_u_v1', output=str(output), repo=execution['repository'],
                source_comparison_repository=source_spec['repo'], execution_identity=execution,
                python=str(python or source_spec['python']), historical_config=str(historical_config),
                root_qpos_address=source_spec['root_qpos_address'], source_lock=str(output/'source_lock.json'),
                source_lock_sha256=file_sha(output/'source_lock.json'), methods=methods, arms=arms,
                conditions=[dict(onset=onset, episodes=1000, batch_size=256, seed=seed,
                                 output=str(output/'conditions'/f'onset_{onset:02d}'))
                            for onset, seed in zip(ONSETS, condition_seeds)],
                contract=common, training_transitions=3*TARGET, maximum_training_panel_interactions=diagnostic_max,
                maximum_evaluation_interactions=4*7*1000*horizon,
                maximum_interactions=3*TARGET+diagnostic_max+4*7*1000*horizon,
                stage_timeout_seconds=172800, role='development; one warm-start descendant per source; no TEST/JCE/JEL',
                notification_contract='Start detached watch_run_errors.py using ACTIVE_RUN.json; verify heartbeat before launch')
    write(output/'spec.json', spec)
    write(output/'status.json', dict(phase='prepared', spec_sha256=file_sha(output/'spec.json'),
                                     maximum_interactions=spec['maximum_interactions']))
    write(output/'ACTIVE_RUN.json', dict(name='Seven-policy Phase U retraining', execution=str(output/'status.json')))
    (output/'INDEX.md').write_text('# Seven-policy Phase U experiment\n\n'
        'Development evidence only. Three sequential Actor+normalizer warm starts; fresh Critic and optimizer.\n\n'
        '[Spec](spec.json) · [Source locks](source_lock.json) · [Status](status.json)\n\n'
        'Four separate 1,000-episode windows: steps 0–2, 5–7, 10–12, 15–17. '
        'All seven fixed policies use shared condition seeds and batch size 256. Raw arrays remain uncropped.\n')
    return spec


def _run_child(command, log, *, repo, timeout):
    env = dict(os.environ, JAX_PLATFORMS='cuda,cpu', XLA_PYTHON_CLIENT_PREALLOCATE='false',
               PYTHONPATH=str(Path(repo)/'JIT/src'), JIT_AUTO_PUBLISH='0', OMP_NUM_THREADS='4')
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with Path(log).open('x') as stream:
        child = subprocess.Popen(command, cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT)
        try:
            code = child.wait(timeout=timeout)
        except BaseException:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait(timeout=30)
            raise
    if code:
        raise RuntimeError(f'child exited {code}; see {log}')


def train_arm(spec_path, key):
    """Child-process entry point; use the real Phase U runner and experiment root."""
    from .formal_training import run_phase_u_formal
    spec = read(spec_path)
    arm = next(a for a in spec['arms'] if a['key'] == key)
    return run_phase_u_formal(Path(arm['config']), arm['run_id'],
                              actor_init_frozen_policy=Path(arm['frozen_policy']),
                              run_root=Path(arm['run_dir']).parent)


def freeze_descendant(spec, arm):
    """Verify final report, provenance and payload; never rewrite checkpoint identity."""
    from .provenance import verify_run
    from .config import load_config
    from .formal_training import FormalReport, validate_formal_report
    from .iterative_probe_training import load_phase_initializer
    from .handoff_bank import pytree_sha256
    run_dir = Path(arm['run_dir'])
    verification = verify_run(run_dir)
    if verification['status'] != 'completed' or verification['training_transitions'] != TARGET:
        raise ValueError('Phase U training did not complete its declared budget')
    if read(run_dir/'resolved_config.json') != read(arm['config']):
        raise ValueError('trained config differs from declaration')
    config = load_config(Path(arm['config']))
    report = read(run_dir/'formal_report.json')
    for field in ('checkpoint_transitions', 'evaluated_transitions'):
        report[field] = tuple(report[field])
    validated = validate_formal_report(FormalReport(**report), config=config)
    if validated.starting_training_transition != 0 or validated.resume_semantics != 'parameter_warm_start_optimizer_reset':
        raise ValueError('descendant must be fresh-budget Actor-only warm start')
    checkpoint = run_dir/'checkpoints'/f'transition_{TARGET}'
    phase = dict(name=arm['key'], source_checkpoint=str(checkpoint), source_phase_config=arm['config'],
                 runtime_compatibility=dict(schema='jit_phase_u_runtime_compatibility_v1', historical_config=spec['historical_config']),
                 input_files={})
    _lock((arm['config'], spec['historical_config'], checkpoint/'identity.json', checkpoint/'payload.pkl',
           run_dir/'formal_report.json', run_dir/'resolved_config.json', run_dir/'actor_initialization.json'), phase['input_files'])
    initialized = read(run_dir/'actor_initialization.json')
    if (initialized.get('source_frozen_policy') != arm['frozen_policy'] or
            not initialized.get('critic_fresh') or not initialized.get('optimizer_fresh')):
        raise ValueError('descendant Actor initialization provenance differs')
    payload = load_phase_initializer(phase)
    if payload.training_transitions != TARGET:
        raise ValueError('descendant final checkpoint transition differs')
    policy = dict(phase, config_sha256=payload.identity.config_sha256,
                  payload_sha256=file_sha(checkpoint/'payload.pkl'),
                  checkpoint_identity_sha256=file_sha(checkpoint/'identity.json'),
                  source_training_transitions=payload.training_transitions,
                  xml_sha256=payload.identity.xml_sha256, actor_sha256=pytree_sha256(payload.actor_params),
                  normalizer_sha256=pytree_sha256(payload.observation_normalizer), critic_sha256=pytree_sha256(payload.critic_params))
    for field in ('actor_frame_fields', 'actor_task_fields', 'action_order'):
        policy[field] = list(getattr(payload.identity, field))
    manifest = dict(schema='jit_frozen_phase_u_descendant_v1', status='frozen', immutable_parameters=True,
                    copied_checkpoint=False, policy_role='development_phase_u_checkpoint', data_role='train',
                    expert_switching_used=False, environment_interactions=0, source=arm['source'],
                    policy=policy, phase_policy=phase, verification=verification)
    manifest['freeze_protocol_sha256'] = canonical_sha256(manifest)
    path = Path(arm['output_manifest']); path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    write(path, manifest)
    return manifest


def _condition_spec(spec, condition):
    methods = []
    inputs = dict(read(spec['source_lock'])['input_files'])
    for method in spec['methods']:
        configured = copy.deepcopy(method)
        if method.get('descendant_manifest'):
            manifest_path = Path(method['descendant_manifest'])
            manifest = read(manifest_path)
            protocol = {k:v for k,v in manifest.items() if k != 'freeze_protocol_sha256'}
            if manifest.get('status') != 'frozen' or canonical_sha256(protocol) != manifest.get('freeze_protocol_sha256'):
                raise ValueError('descendant manifest identity drift')
            _verify_locks(manifest['phase_policy']['input_files'])
            parent = next(m for m in spec['methods'] if m['key'] == method['source'])
            configured['template'] = copy.deepcopy(parent['template'])
            configured['template']['phase_policy'] = manifest['phase_policy']
            _lock((manifest_path,), inputs)
            inputs.update(manifest['phase_policy']['input_files'])
        configured['template']['pulse_start_schedule'] = [condition['onset']]
        methods.append(configured)
    return dict(schema='jit_batched_paired_pulse_comparison_v1', output=condition['output'],
                repo=spec['repo'], python=spec['python'], methods=methods, input_files=inputs,
                episodes=condition['episodes'], batch_size=condition['batch_size'], seed=condition['seed'],
                batches=batch_plan(condition['episodes'], condition['batch_size'], condition['seed']),
                root_qpos_address=spec['root_qpos_address'], role=spec['role'], new_training_transitions=0,
                contract=dict(spec['contract'], pulse_start_schedule=[condition['onset']]),
                maximum_interactions=len(methods)*condition['episodes']*spec['contract']['horizon'])


def prepare_resume(previous, output, execution_repository):
    """Freeze an evaluation-only attempt; completed members remain immutable."""
    previous = Path(previous).resolve(); output = Path(output).resolve()
    old = read(previous/'spec.json')
    if file_sha(previous/'spec.json') != read(previous/'status.json')['spec_sha256']:
        raise ValueError('previous experiment declaration drift')
    if file_sha(old['source_lock']) != old['source_lock_sha256']:
        raise ValueError('previous source lock drift')
    _verify_locks(read(old['source_lock'])['input_files'])
    spec = copy.deepcopy(old)
    execution = _execution_identity(execution_repository)
    spec.update(output=str(output), repo=execution['repository'], execution_identity=execution)
    locks = dict(read(old['source_lock'])['input_files'])
    locks.update(execution['input_files'])
    _lock((previous/'spec.json', previous/'status.json'), locks)
    reused = {}; failed_cost = 0
    for arm in old['arms']:
        manifest = read(arm['output_manifest'])
        _verify_locks(manifest['phase_policy']['input_files'])
        verification = manifest['verification']
        if verification['status'] != 'completed' or verification['training_transitions'] != TARGET:
            raise ValueError('cannot resume before all training arms complete')
        _lock((arm['output_manifest'],), locks)
    for original, condition in zip(old['conditions'], spec['conditions']):
        condition['output'] = str(output/'conditions'/f'onset_{condition["onset"]:02d}')
        source = Path(original['output'])
        if not (source/'spec.json').exists():
            continue
        cspec = _condition_spec(old, original)
        if read(source/'spec.json') != cspec:
            raise ValueError('previous condition declaration drift')
        for batch in cspec['batches']:
            for method in cspec['methods']:
                directory = source/'batches'/f'{batch["index"]:04d}'/method['key']
                if not directory.exists():
                    continue
                expected = dict(method['template'], seed=batch['seed'], num_envs=batch['count'], round_index=batch['offset'])
                ev_path = directory.parent/f'{method["key"]}_spec.json'
                if read(ev_path) != expected:
                    raise ValueError('previous member declaration drift')
                status_path = directory/'status.json'
                if status_path.exists() and read(status_path)['phase'] == 'completed':
                    actual = read(directory/'hyperparameters.json')
                    if any(actual.get(k) != v for k,v in expected.items()):
                        raise ValueError('previous member execution contract drift')
                    files = [p for p in directory.rglob('*') if p.is_file()]
                    _lock(files + [ev_path], locks)
                    key = f'{condition["onset"]}/{batch["index"]}/{method["key"]}'
                    reused[key] = str(directory)
                elif (directory/'prefixes.npz').exists():
                    with np.load(directory/'prefixes.npz') as tape:
                        failed_cost += int(tape['prefix_mask'].shape[0] * tape['prefix_mask'].shape[1])
                    _lock((directory/'prefixes.npz',), locks)
                else:
                    # Unknown failed rollout cost is conservatively reserved in full.
                    failed_cost += batch['count'] * spec['contract']['horizon']
    spec['resume'] = dict(previous=str(previous), reused_members=reused,
                          inherited_failed_interactions=old.get('resume', {}).get('inherited_failed_interactions', 0)+failed_cost,
                          new_training_transitions=0)
    spec['maximum_interactions'] += failed_cost
    output.mkdir(parents=True, exist_ok=False)
    spec['source_lock'] = str(output/'source_lock.json')
    write(spec['source_lock'], dict(schema='jit_seven_policy_resume_lock_v1', input_files=locks))
    spec['source_lock_sha256'] = file_sha(spec['source_lock'])
    write(output/'spec.json', spec)
    write(output/'status.json', dict(phase='prepared', spec_sha256=file_sha(output/'spec.json')))
    write(output/'ACTIVE_RUN.json', dict(name='Seven-policy evaluation recovery', execution=str(output/'status.json')))
    (output/'INDEX.md').write_text('# Seven-policy evaluation recovery\n\n'
        '[Status](status.json) · [Spec](spec.json) · [Source locks](source_lock.json)\n\n'
        f'All three trainings reused; {len(reused)} completed member batches reused. '
        f'Failed-attempt interactions retained separately: {failed_cost}. Original artifacts preserved.\n')
    return spec


def run_experiment(spec_path):
    """Execute once, sequentially; retain failed attempts and stop before evaluation."""
    spec_path = Path(spec_path).resolve(); spec = read(spec_path); output = Path(spec['output'])
    original_status = read(output/'status.json')
    if original_status['phase'] != 'prepared':
        raise ValueError('experiment already attempted; preserve it and prepare a new output root')
    started = time.monotonic(); completed_arms = []; completed_conditions = []; stage = 'preflight'
    def status(phase, **extra):
        write(output/'status.json', dict(phase=phase, stage=stage, pid=os.getpid(),
            spec_sha256=original_status['spec_sha256'], completed_arms=completed_arms,
            completed_conditions=completed_conditions, wall_seconds=time.monotonic()-started,
            maximum_interactions=spec['maximum_interactions'], **extra))
    cli = Path(spec['repo'])/'JIT/cli'
    try:
        if file_sha(spec_path) != original_status['spec_sha256'] or file_sha(spec['source_lock']) != spec['source_lock_sha256']:
            raise ValueError('experiment declaration drift')
        _verify_locks(read(spec['source_lock'])['input_files'])
        for arm in spec['arms']:
            if spec.get('resume'):
                completed_arms.append(arm['key'])
                continue
            stage = 'training_' + arm['key']; status('running')
            _run_child([spec['python'], str(cli/'run_seven_policy_phase_u.py'), '_train-arm',
                        '--spec', str(spec_path), '--arm', arm['key']], output/'logs'/f'{stage}.log',
                       repo=spec['repo'], timeout=spec['stage_timeout_seconds'])
            freeze_descendant(spec, arm)
            completed_arms.append(arm['key']); status('running')
        for condition in spec['conditions']:
            stage = f'evaluation_onset_{condition["onset"]:02d}'; status('running')
            cspec = _condition_spec(spec, condition); directory = Path(condition['output'])
            directory.mkdir(parents=True, exist_ok=False); write(directory/'spec.json', cspec)
            receipts = []; write(directory/'status.json', dict(phase='running'))
            for batch in cspec['batches']:
                batch_dir = directory/'batches'/f'{batch["index"]:04d}'; batch_dir.mkdir(parents=True)
                for method in cspec['methods']:
                    stage = f'onset_{condition["onset"]:02d}_batch_{batch["index"]:04d}_{method["key"]}'; status('running')
                    ev = dict(method['template'], seed=batch['seed'], num_envs=batch['count'], round_index=batch['offset'])
                    ev_path = batch_dir/f'{method["key"]}_spec.json'; write(ev_path, ev)
                    reuse_key = f'{condition["onset"]}/{batch["index"]}/{method["key"]}'
                    reused = spec.get('resume', {}).get('reused_members', {}).get(reuse_key)
                    if reused:
                        (batch_dir/method['key']).symlink_to(reused, target_is_directory=True)
                        continue
                    _run_child([spec['python'], str(cli/'run_pulse_exploration.py'), '--mode', 'collect',
                                '--spec', str(ev_path), '--output', str(batch_dir/method['key'])],
                               batch_dir/f'{method["key"]}.log', repo=spec['repo'], timeout=spec['stage_timeout_seconds'])
                receipt = verify_batch(cspec, batch, batch_dir)
                write(batch_dir/'verification.json', receipt)
                receipts.append(dict(batch=batch, verification=str(batch_dir/'verification.json'),
                                     verification_sha256=file_sha(batch_dir/'verification.json'),
                                     charged_interactions=receipt['charged_interactions']))
                write(directory/'receipts.json', receipts)
            write(directory/'status.json', dict(phase='completed', completed_batches=len(receipts)))
            completed_conditions.append(condition['onset']); status('running')
        stage = 'reporting'; status('running')
        summary = report_experiment(spec_path)
        stage = 'complete'; status('completed', summary=summary)
        return summary
    except BaseException as exc:
        status('error', error=f'{type(exc).__name__}: {exc}', automatic_retry=False)
        raise


def report_experiment(spec_path):
    """Reverify raw batches, preserve official outcomes, and export all timing rows."""
    from collections import Counter
    from .analysis.batched_pulse_report import _wilson
    spec_path = Path(spec_path); spec = read(spec_path); output = Path(spec['output'])
    if (file_sha(spec['source_lock']) != spec['source_lock_sha256'] or
            file_sha(spec_path) != read(output/'status.json')['spec_sha256']):
        raise ValueError('experiment declaration drift')
    _verify_locks(read(spec['source_lock'])['input_files'])
    rows, traces, summaries = [], [], {}
    charged = active = 0
    bounds = [float('inf'), float('-inf'), float('inf'), float('-inf')]
    for condition in spec['conditions']:
        directory = Path(condition['output']); cspec = read(directory/'spec.json')
        if cspec != _condition_spec(spec, condition):
            raise ValueError('condition declaration drift')
        if read(directory/'status.json')['phase'] != 'completed':
            raise ValueError('condition incomplete')
        receipts = read(directory/'receipts.json')
        if len(receipts) != len(cspec['batches']):
            raise ValueError('condition batch denominator differs')
        counts = {m['key']:Counter() for m in spec['methods']}
        seen = {m['key']:set() for m in spec['methods']}
        for batch, receipt in zip(cspec['batches'], receipts):
            if batch != receipt['batch'] or file_sha(receipt['verification']) != receipt['verification_sha256']:
                raise ValueError('batch verification identity drift')
            verified = verify_batch(cspec, batch, Path(receipt['verification']).parent)
            if verified != read(receipt['verification']):
                raise ValueError('batch verification no longer matches raw arrays')
            charged += verified['charged_interactions']; active += verified['active_interactions']
            timing = {}
            for method in cspec['methods']:
                trace = Path(receipt['verification']).parent/method['key']/'prefixes.npz'
                traces.append((condition['onset'], method['key'], trace))
                with np.load(trace) as tape:
                    for lane in range(batch['count']):
                        ticks = np.flatnonzero(tape['prefix_mask'][:, lane])
                        points = tape['qpos'][ticks, lane][:, [spec['root_qpos_address'], spec['root_qpos_address']+2]]
                        bounds[0] = min(bounds[0], float(points[:,0].min())); bounds[1] = max(bounds[1], float(points[:,0].max()))
                        bounds[2] = min(bounds[2], float(points[:,1].min())); bounds[3] = max(bounds[3], float(points[:,1].max()))
                        contacts = ticks[tape['first_valid_contact'][ticks, lane]]
                        landing = float(tape['time'][contacts[0], lane]) if len(contacts) else None
                        success = bool(tape['success'][ticks[-1], lane]) and not bool(tape['physical_failure'][ticks[-1], lane])
                        recovery = float(tape['time'][ticks[-1], lane]) if success else None
                        timing[(method['key'], lane)] = dict(first_valid_contact_time_s=landing,
                            recovery_success_time_s=recovery,
                            contact_to_recovery_s=recovery-landing if landing is not None and recovery is not None else None,
                            maximum_recorded_recovery_ticks=int(tape['recovery_ticks'][ticks, lane].max()))
            for source_row in verified['rows']:
                row = dict(source_row, onset=condition['onset'])
                row.update(timing[(row['method'], row['lane'])]); rows.append(row)
                key = row['method']
                if row['episode'] in seen[key]:
                    raise ValueError('duplicate condition episode')
                seen[key].add(row['episode']); counts[key]['episodes'] += 1
                counts[key]['successes'] += int(row['success']); counts[key][row['terminal_reason']] += 1
        summaries[str(condition['onset'])] = {}
        for key, count in counts.items():
            if seen[key] != set(range(condition['episodes'])):
                raise ValueError('condition method denominator differs')
            low, high = _wilson(count['successes'], count['episodes'])
            summaries[str(condition['onset'])][key] = dict(count, failures=count['episodes']-count['successes'],
                success_rate=count['successes']/count['episodes'], wilson_95=[float(low), float(high)])
    training_total = training = 0
    for arm in spec['arms']:
        verification = read(arm['output_manifest'])['verification']
        training_total += verification['total_environment_transitions']
        training += verification['training_transitions']
    if training != spec['training_transitions'] or training_total+charged+spec.get('resume', {}).get('inherited_failed_interactions', 0) > spec['maximum_interactions']:
        raise ValueError('experiment cost reconciliation differs from budget')
    reports = output/'reports'; reports.mkdir(exist_ok=True)
    index = 0
    while (reports/f'report_{index:04d}').exists():
        index += 1
    destination = reports/f'report_{index:04d}'; destination.mkdir()
    csv_path = destination/'episodes.csv'
    with csv_path.open('x') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x0, x1, z0, z1 = bounds
    margin_x = max((x1-x0)*.02, .01); margin_z = max((z1-z0)*.02, .01)
    xedges = np.linspace(x0-margin_x, x1+margin_x, 161)
    zedges = np.linspace(z0-margin_z, z1+margin_z, 121)
    densities = {(c['onset'],m['key']):np.zeros((160,120)) for c in spec['conditions'] for m in spec['methods']}
    for onset, key, trace in traces:
        with np.load(trace) as tape:
            for lane in range(tape['prefix_mask'].shape[1]):
                ticks = np.flatnonzero(tape['prefix_mask'][:,lane])
                points = tape['qpos'][ticks,lane][:,[spec['root_qpos_address'],spec['root_qpos_address']+2]]
                densities[(onset,key)] += np.histogram2d(points[:,0],points[:,1],bins=(xedges,zedges))[0]/len(points)
    vmax = max(float(d.max()) for d in densities.values())/1000
    panels = []
    for condition in spec['conditions']:
        onset = condition['onset']; fig, axes = plt.subplots(4,2,figsize=(13,15),sharex=True,sharey=True,layout='constrained')
        for ax, method in zip(axes.flat, spec['methods']):
            count = summaries[str(onset)][method['key']]
            mesh = ax.pcolormesh(xedges, zedges, (densities[(onset,method['key'])]/1000).T,
                                 shading='auto',cmap='viridis',vmin=0,vmax=vmax)
            ax.set_title(f'{method["key"]}: {count["successes"]}/1000 recovery success')
            ax.set_xlabel('World x (m)'); ax.set_ylabel('Root z (m)')
        axes.flat[-1].set_visible(False)
        fig.colorbar(mesh,ax=list(axes.flat[:-1]),label='Episode-weighted occupancy / bin')
        fig.suptitle(f'Fixed pulse steps {onset}–{onset+2}; shared full extent; all 7,000 episodes')
        for suffix in ('png','pdf'):
            path = destination/f'onset_{onset:02d}_density.{suffix}';fig.savefig(path,dpi=160);panels.append(str(path))
        plt.close(fig)
    np.savez_compressed(destination/'density_arrays.npz',x_edges=xedges,z_edges=zedges,
                        **{f'onset_{onset:02d}_{key}':value for (onset,key),value in densities.items()})
    summary = dict(conditions=summaries, episodes=len(rows), training_transitions=training,
                   training_panel_interactions=training_total-training, evaluation_charged_interactions=charged,
                   evaluation_active_interactions=active, inherited_failed_interactions=spec.get('resume', {}).get('inherited_failed_interactions', 0),
                   charged_interactions=training_total+charged+spec.get('resume', {}).get('inherited_failed_interactions', 0),
                   maximum_interactions=spec['maximum_interactions'], episodes_csv=str(csv_path), density_panels=panels,
                   source_lock=spec['source_lock'], source_lock_sha256=spec['source_lock_sha256'],
                   config_sha256={arm['key']:file_sha(arm['config']) for arm in spec['arms']},
                   descendant_manifest_sha256={arm['key']:file_sha(arm['output_manifest']) for arm in spec['arms']},
                   timing_semantics='First recorded valid-contact transition; official recovery-success endpoint. No relabeling.',
                   confidence_interval_scope='Wilson binomial descriptive intervals across paired development draws; not independent training repeats.',
                   display_contract=dict(world_bounds=bounds, shared_axes=True, raw_arrays_uncropped=True, episodes_sampled=False),
                   role=spec['role'])
    write(destination/'summary.json', summary)
    text = ['# Seven-policy development comparison\n',
            'Official full-task stable recovery labels retained. All failures remain in the denominator.\n',
            '[Episode outcomes and contact/recovery timing](episodes.csv) · [Summary and cost reconciliation](summary.json)\n',
            'Wilson 95% intervals describe these paired development draws; they are not independent training repeats.\n']
    for condition in spec['conditions']:
        onset = condition['onset']
        text += [f'\n## Pulse steps {onset}–{onset+2}\n',
                 '| Method | Success | Failure | Rate | Wilson 95% |\n|---|---:|---:|---:|---|']
        for method in spec['methods']:
            count = summaries[str(onset)][method['key']]; low, high = count['wilson_95']
            text.append(f'| {method["label"]} | {count["successes"]}/1000 | {count["failures"]} | {count["success_rate"]:.3%} | {low:.3%}–{high:.3%} |')
        text.append(f'\n![All-episode density](onset_{onset:02d}_density.png)\n\n[PDF](onset_{onset:02d}_density.pdf)\n')
    failed_cost = spec.get('resume', {}).get('inherited_failed_interactions', 0)
    text.append(f'\nCharged transitions: {training:,} PPO + {training_total-training:,} training panels + {charged:,} comparison + {failed_cost:,} failed attempts = {training_total+charged+failed_cost:,}.\n')
    (destination/'INDEX.md').write_text('\n'.join(text))
    with (output/'INDEX.md').open('a') as stream:
        stream.write(f'\n[Verified seven-policy tables, timing CSV and density panels]({destination.relative_to(output)}/INDEX.md)\n')
    return summary
