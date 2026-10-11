"""Identity-bound Actor export and saved-observation parity, without physics.

This bundle is an offline interface artifact, not an engine task adapter or a
landing evaluation. The normalizer always comes from the same checkpoint.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from scipy.special import expit

from .checkpoint import CheckpointIdentity, load_checkpoint
from .config import canonical_sha256
from .constants import ACTION_ORDER, ACTOR_FRAME_FIELDS, ACTOR_TASK_FIELDS, CTRL_DT, N_SUBSTEPS, SIM_DT


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _resolve(path, root):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _timing_contract(xml_paths, configs, inputs):
    """Inspect the source override without compiling a model or instantiating it."""
    source = Path(__file__).resolve().parent
    paths = {name: source / name for name in ('constants.py', 'env.py', 'model.py')}
    for path in paths.values():
        inputs[str(path)] = _sha(path)
    env = ast.parse(paths['env.py'].read_text())
    model = ast.parse(paths['model.py'].read_text())
    calls = [node for node in ast.walk(env) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == 'create'
             and isinstance(node.func.value, ast.Name) and node.func.value.id == 'config_dict']
    env_override = any({kw.arg: ast.unparse(kw.value) for kw in node.keywords}.items()
                       >= {'ctrl_dt': 'CTRL_DT', 'sim_dt': 'SIM_DT', 'action_repeat': '1'}.items()
                       for node in calls)
    model_override = any(isinstance(node, ast.Assign) and ast.unparse(node.value) == 'SIM_DT'
                         and any(ast.unparse(target) == 'model.opt.timestep' for target in node.targets)
                         for node in ast.walk(model))
    if (not env_override or not model_override or SIM_DT != .005 or CTRL_DT != .020
            or N_SUBSTEPS != 4 or not np.isclose(SIM_DT * N_SUBSTEPS, CTRL_DT, rtol=0, atol=1e-12)):
        raise ValueError('runtime timing override changed; explicit adapter audit required')
    declared = False
    for config in configs:
        for key, expected in (('sim_dt', SIM_DT), ('ctrl_dt', CTRL_DT), ('control_dt', CTRL_DT)):
            if key in config:
                declared = True
                if config[key] != expected:
                    raise ValueError('JSON/runtime timing mismatch')
    raw = []
    for xml in sorted(xml_paths):
        option = ElementTree.parse(xml).getroot().find('option')
        raw.append(float(option.attrib['timestep']) if option is not None and 'timestep' in option.attrib else None)
    return dict(raw_xml_timestep_s=raw, physics_substeps_per_control=N_SUBSTEPS,
                json_declares_timing=declared, source_files={k: str(v) for k, v in paths.items()},
                validation='source_constants_and_override_audited; no environment instantiated')


def _task_contract(path, repository, identity):
    task = _read(path)
    if canonical_sha256(task) != identity['config_sha256']:
        raise ValueError('task config_sha256 identity mismatch')
    inputs = {str(path): _sha(path)}
    phases = {}
    for phase in ('up', 'down'):
        if phase + '_config_path' in task.get('inputs', {}):
            p = _resolve(task['inputs'][phase + '_config_path'], repository)
            phases[phase] = _read(p)
            if canonical_sha256(phases[phase]) != task['inputs'][phase + '_config_sha256']:
                raise ValueError(f'{phase} config_sha256 identity mismatch')
            inputs[str(p)] = _sha(p)
    configs = list(phases.values()) or [task]
    xml_paths = set()
    for config in configs:
        model = config['model']
        xml = _resolve(model['xml_path'], repository)
        if _sha(xml) != model['xml_sha256'] or model['xml_sha256'] != identity['xml_sha256']:
            raise ValueError('model xml_sha256 identity mismatch')
        inputs[str(xml)] = _sha(xml)
        xml_paths.add(xml)
        if config['action'].get('joint_target_semantics') != 'keyframe_centered_absolute':
            raise ValueError('unsupported action joint_target_semantics')
    contract = dict(success_criterion=task.get('success_criterion', 'not_declared'),
        success=phases.get('down', task).get('descent', {}),
        source_action=configs[0]['action'], control_dt_s=CTRL_DT, sim_dt_s=SIM_DT,
        timing=_timing_contract(xml_paths, [task] + configs, inputs),
        actor_dimensions=76, actor_history_frames=3,
        actor_frame_fields=list(ACTOR_FRAME_FIELDS), actor_task_fields=list(ACTOR_TASK_FIELDS),
        action_order=list(ACTION_ORDER), angular_velocity_unit='rad/s',
        actor_angular_velocity_frame='body_local', reward_angular_velocity_frame='world',
        root_velocity_conversion='v_root_world = v_com_world - omega_world cross r_com_world; exactly once',
        position_frame='world; meters; no trajectory recentering for scoring',
        reset_requirement='full observation history, controller/event state and engine solver state',
        scoring_status='declared source configuration only; no target scorer equivalence verified',
        checkpoint_snapshot_reset_contract=task.get('snapshot_reset_contract', 'not_declared'))
    if contract['success_criterion'] == 'stable_forward_recovery':
        required = ('recovery_ticks', 'continuous_stability', 'stable_min_forward_velocity',
                    'stable_max_abs_roll', 'stable_max_abs_pitch', 'stable_max_wheel_clearance',
                    'max_wheel_penetration', 'terminate_on_body_contact')
        if any(key not in contract['success'] for key in required):
            raise ValueError('incomplete stable_forward_recovery success contract')
        contract['success_endpoint'] = ('valid takeoff and contact followed by continuously stable recovery ticks; '
            'concurrent physical failure or timeout is not success; preserve source event ordering')
    return task, phases, inputs, contract


def _observations(manifest_path, xml_sha256, max_observations):
    manifest = _read(manifest_path)
    entries = manifest.get('entries', [])
    # Validate every role before opening even the first observation archive.
    if (manifest.get('schema') != 'jit_sim2sim_observations_v1'
            or manifest.get('final_test_used') is not False or not entries
            or any(row.get('role') not in ('TRAIN', 'SOLVER_DEV', 'DEV') for row in entries)):
        raise ValueError('saved observations require TRAIN/SOLVER_DEV/DEV role; TEST is forbidden')
    for field, expected in (('actor_frame_fields', list(ACTOR_FRAME_FIELDS)),
                            ('actor_task_fields', list(ACTOR_TASK_FIELDS)), ('xml_sha256', xml_sha256)):
        if manifest.get(field) != expected:
            raise ValueError(f'observation {field} identity mismatch')
    blocks, sources, count = [], [], 0
    for row in entries:
        path = _resolve(row['path'], manifest_path.parent)
        if _sha(path) != row['sha256']:
            raise ValueError('observation archive sha256 mismatch')
        with np.load(path, allow_pickle=False) as archive:
            values = np.asarray(archive[row['key']])
        if values.ndim != 2 or values.shape[1] != 76 or not len(values):
            raise ValueError('saved observations must be nonempty N x 76')
        if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
            raise ValueError('saved observations must be finite floating point')
        selected = min(len(values), max_observations - count)
        if selected:
            blocks.append(values[:selected].astype(np.float32))
            count += selected
        sources.append(dict(path=str(path), sha256=row['sha256'], key=row['key'], role=row['role'],
                            available_rows=len(values), used_rows=selected, selection='first rows in declared order'))
    return np.concatenate(blocks), sources


def _arrays(payload):
    params = payload.actor_params['params']
    if set(params) != {f'hidden_{i}' for i in range(4)}:
        raise ValueError('unsupported Actor architecture; expected 256 x 3 tanh-normal')
    arrays = {}
    for i, shape in enumerate(((76, 256), (256, 256), (256, 256), (256, 8))):
        for name, expected in (('kernel', shape), ('bias', (shape[1],))):
            value = np.asarray(params[f'hidden_{i}'][name])
            if value.shape != expected or not np.isfinite(value).all():
                raise ValueError('Actor shape/nonfinite mismatch')
            arrays[f'hidden_{i}_{name}'] = value
    norm = payload.observation_normalizer
    for name in ('mean', 'std'):
        value = np.asarray(getattr(norm, name)['state'])
        if value.shape != (76,) or not np.isfinite(value).all() or (name == 'std' and np.any(value <= 0)):
            raise ValueError('normalizer shape/nonfinite/nonpositive std mismatch')
        arrays[name] = value
    return arrays


def numpy_action(arrays, observations):
    """The existing Isaac NPZ Actor format, with deterministic tanh-normal mode."""
    x = (np.asarray(observations, dtype=np.float32) - arrays['mean']) / arrays['std']
    for i in range(4):
        x = x @ arrays[f'hidden_{i}_kernel'] + arrays[f'hidden_{i}_bias']
        if i < 3:
            x = x * expit(x)
    return np.tanh(x[..., :4])


def export_checkpoint(*, checkpoint, task_config, observations_manifest, output, repository,
                      expected_payload_sha256, expected_actor_sha256, expected_normalizer_sha256,
                      max_observations=1024):
    """Export a standard CheckpointPayload; reject implicit model/config defaults."""
    checkpoint, task_config, observations_manifest, output, repository = map(
        lambda p: Path(p).resolve(), (checkpoint, task_config, observations_manifest, output, repository))
    if output.exists():
        raise FileExistsError(output)
    if type(max_observations) is not int or not 0 < max_observations <= 65536:
        raise ValueError('max_observations must be in 1..65536')
    identity = _read(checkpoint / 'identity.json')
    if (_sha(checkpoint / 'payload.pkl') != expected_payload_sha256
            or identity.get('payload_sha256') != expected_payload_sha256):
        raise ValueError('checkpoint payload_sha256 identity mismatch')
    task, phases, inputs, contract = _task_contract(task_config, repository, identity)
    obs, sources = _observations(observations_manifest, identity['xml_sha256'], max_observations)
    payload = load_checkpoint(checkpoint, expected=CheckpointIdentity(identity['config_sha256'],
        identity['xml_sha256'], ACTOR_FRAME_FIELDS, ACTOR_TASK_FIELDS, ACTION_ORDER))
    import jax
    from .handoff_bank import pytree_sha256
    from .ppo import make_network_factory
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks
    actor_sha, norm_sha = map(pytree_sha256, (payload.actor_params, payload.observation_normalizer))
    if actor_sha != expected_actor_sha256 or norm_sha != expected_normalizer_sha256:
        raise ValueError('Actor/normalizer sha256 identity mismatch')
    arrays = _arrays(payload)
    net = make_network_factory()({'state': 76, 'privileged_state': 106}, 4,
                                preprocess_observations_fn=running_statistics.normalize)
    fn = networks.make_inference_fn(net)((payload.observation_normalizer, payload.actor_params,
                                        payload.critic_params), deterministic=True)
    reference = np.asarray(jax.vmap(lambda value: fn({'state': value}, jax.random.PRNGKey(0))[0])(obs))
    actual = numpy_action(arrays, obs)
    error = float(np.max(np.abs(reference - actual)))
    if not np.isfinite(error) or error > 3e-5:
        raise ValueError(f'saved-observation inference parity failed: max_abs_error={error}')
    inputs.update({str(checkpoint / 'identity.json'): _sha(checkpoint / 'identity.json'),
                   str(checkpoint / 'payload.pkl'): expected_payload_sha256,
                   str(observations_manifest): _sha(observations_manifest)})
    receipt = checkpoint / 'checkpoint_receipt.json'
    if receipt.exists():
        inputs[str(receipt)] = _sha(receipt)
    report = dict(schema='jit_sim2sim_offline_export_v1', status='offline_export_verified',
        source=dict(checkpoint=str(checkpoint), payload_sha256=expected_payload_sha256,
                    actor_sha256=actor_sha, normalizer_sha256=norm_sha,
                    training_transitions=payload.training_transitions),
        task_contract=contract, input_files=inputs, saved_observations=sources,
        parity=dict(observation_count=len(obs), max_abs_error=error, tolerance=3e-5, backend=jax.default_backend()),
        new_physics_control_steps=0, new_physics_substeps=0, training_updates=0, final_test_used=False,
        engine_evaluation_status='not_run', target_adapter_ready=False,
        limitations=['No environment, reset, contact or success scorer was executed.',
            'Existing external probes require explicit task adapter/scorer changes before retention evaluation.',
            'Checkpoint-bound task config does not certify any separate training/evaluation runtime overrides.',
            'Array inference parity is not sim2sim task success or observation-adapter parity.'])
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / 'actor.npz', **arrays)
    np.savez(output / 'reference_inference.npz', observations=obs, actions=reference)
    for name, data in [('identity.json', identity), ('task_config.json', task), ('phase_configs.json', phases)]:
        (output / name).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')
    report['output_files'] = {p.name: _sha(p) for p in sorted(output.iterdir())}
    (output / 'preflight.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return report
