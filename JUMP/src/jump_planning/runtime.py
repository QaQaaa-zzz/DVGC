"""CPU qualification of a frozen JIT Actor under an explicit new trigger adapter.

No MJX replay equivalence is implied. Contact labels use MuJoCo's contact
records/solver forces; support boxes are used only for geometric diagnostics.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import traceback
from xml.etree import ElementTree

import mujoco
import numpy as np


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_contacts(model, data, index, robot_ids, limits):
    """Read actual contacts, retaining proximity records without labeling them.

    A pair is active if it has positive normal force above the declared
    threshold, or penetration deeper than the distance tolerance. Positive
    distance alone is insufficient. Tiny zero-force touches are tolerated.
    """
    result = dict(front_contact=False, rear_contact=False, body_contact=False,
                  obstacle_contact=False, front_touch_x=None, rear_touch_x=None,
                  contact_pairs=[])
    wheel_names = {index.frontwheel_geom_id: 'front', index.rearwheel_geom_id: 'rear'}
    floor = index.floor_geom_id
    obstacle = index.obstacle_geom_id
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, contact_id, force)
        distance = float(contact.dist)
        active = (float(force[0]) > float(limits['contact_force_threshold']) or
                  distance < -float(limits['contact_distance_tolerance']))
        first, second = int(contact.geom1), int(contact.geom2)
        result['contact_pairs'].append(dict(
            geom1=first, geom2=second,
            name1=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, first),
            name2=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, second),
            distance=distance, normal_force=float(force[0]), force=force.tolist(),
            position=np.asarray(contact.pos).tolist(), active=bool(active)))
        if not active:
            continue
        pair = {first, second}
        if obstacle in pair and pair.intersection(robot_ids):
            result['obstacle_contact'] = True
        if floor not in pair:
            continue
        robot = second if first == floor else first
        if robot in wheel_names:
            name = wheel_names[robot]
            result[f'{name}_contact'] = True
            # Keep the first actual contact, not the wheel center. All contact
            # positions remain available in contact_pairs for an audit.
            if result[f'{name}_touch_x'] is None:
                result[f'{name}_touch_x'] = float(contact.pos[0])
        elif robot in robot_ids:
            result['body_contact'] = True
    return result


class ApproachQualification:
    """Latch qualification only after a continuous, valid grounded prefix."""

    def __init__(self, config):
        self.config = config
        self.grounded_since = None
        self.qualified_time = None

    def update(self, frame):
        if self.qualified_time is not None:
            return True
        limit = self.config['limits']
        valid = (frame['finite'] and frame['front_contact'] and frame['rear_contact']
                 and frame['vx'] >= limit['min_forward_speed']
                 and abs(frame['roll']) <= limit['max_roll']
                 and abs(frame['pitch']) <= limit['max_pitch']
                 and abs(frame['yaw']) <= limit['max_yaw']
                 and abs(frame['y']) <= limit['corridor_half_width'])
        if not valid:
            self.grounded_since = None
            return False
        if self.grounded_since is None:
            self.grounded_since = frame['time']
        if frame['time'] - self.grounded_since + 1e-10 >= self.config['timing']['settle_seconds']:
            self.qualified_time = frame['time']
        return self.qualified_time is not None


class TriggerAdapter:
    """One external trigger; clear the signal after ascent then a true apex."""

    def __init__(self, airborne_seconds, clearance, ascent_velocity=.05, descent_velocity=.05):
        self.airborne_seconds = float(airborne_seconds)
        self.clearance = float(clearance)
        self.ascent_velocity = float(ascent_velocity)
        self.descent_velocity = float(descent_velocity)
        self.triggered = False
        self.airborne_since = None
        self.airborne_seen = False
        self.ascent_seen = False
        self.apex_seen = False
        self.apex_time = None

    def signal(self, triggered):
        self.triggered = self.triggered or bool(triggered)
        return int(self.triggered and not self.apex_seen)

    def observe(self, frame, *, qualified, time):
        if not (qualified and self.triggered):
            return
        clear = (not frame['front_contact'] and not frame['rear_contact']
                 and min(frame['front_clearance'], frame['rear_clearance']) > self.clearance)
        if clear:
            if self.airborne_since is None:
                self.airborne_since = time
            if time - self.airborne_since + 1e-10 >= self.airborne_seconds:
                self.airborne_seen = True
            self.ascent_seen |= frame['vz'] >= self.ascent_velocity
        else:
            self.airborne_since = None
        if (self.airborne_seen and self.ascent_seen and not self.apex_seen
                and frame['vz'] <= -self.descent_velocity):
            self.apex_seen = True
            self.apex_time = time


class QualificationRuntime:
    """Reusable model and Actor with fully reset per-case closed-loop state."""

    actor_observation_size = 76
    privileged_observation_size = 106
    action_size = 4

    def __init__(self, config, policy_spec, source_root):
        import jax
        from jax import numpy as jp
        from jit_dvgc.action_mapping import map_action
        from jit_dvgc.checkpoint import CheckpointIdentity, load_checkpoint
        from jit_dvgc.config import load_config
        from jit_dvgc.constants import ACTION_ORDER, ACTOR_FRAME_FIELDS, ACTOR_TASK_FIELDS
        from jit_dvgc.geometry import build_geometry_contract, collision_support_bounds, quaternion_to_euler
        from jit_dvgc.model import load_host_model
        from jit_dvgc.observation import observable_frame
        from jit_dvgc.ppo import make_checkpoint_policy

        self.config = config
        self.policy_spec = policy_spec
        self.source_root = Path(source_root).resolve()
        self.worktree_root = Path(__file__).resolve().parents[3]
        self._jax, self._jp = jax, jp
        if jax.default_backend() != 'cpu':
            raise ValueError('Qualification runtime requires JAX_PLATFORMS=cpu before imports')
        timing = config['timing']
        if not (math.isclose(timing['sim_dt'], .005) and math.isclose(timing['control_dt'], .020)):
            raise ValueError('Qualification timing must remain 5 ms physics / 20 ms control')
        config_path = self.worktree_root / policy_spec.get(
            'up_config', config.get('jit_model_config', 'JIT/configs/phase_u_continuation_smoke.json'))
        resolved = load_config(config_path)
        action_config = config['action']
        resolved = replace(resolved, action=replace(
            resolved.action, base_rear_speed=float(action_config['base_rear_speed']),
            rear_speed_delta=float(action_config['rear_speed_delta']),
            joint_target_semantics=action_config['joint_target_semantics']))
        original = load_host_model(resolved)
        tree = ElementTree.parse(original.xml_path)
        compiler = tree.getroot().find('compiler')
        compiler.set('meshdir', str((original.xml_path.parent / compiler.get('meshdir', '')).resolve()))
        obstacle = tree.getroot().find(".//geom[@name='step']")
        scene = config['scene']
        obstacle.set('size', f"{scene['length']/2:.17g} {scene['half_width']:.17g} {scene['height']/2:.17g}")
        obstacle.set('pos', f"{scene['front_x']+scene['length']/2:.17g} 0 {scene['height']/2:.17g}")
        self.scene_xml = ElementTree.tostring(tree.getroot(), encoding='unicode')
        model = mujoco.MjModel.from_xml_string(self.scene_xml)
        model.opt.timestep = .005
        # Recompile the declared scene rather than leave stale broadphase bounds
        # by mutating geom_size after compilation. Robot physics must be equal.
        for field in ('body_mass', 'body_inertia', 'actuator_ctrlrange', 'actuator_forcerange',
                      'actuator_gainprm', 'actuator_biasprm', 'geom_friction'):
            if not np.array_equal(getattr(model, field), getattr(original.mj_model, field)):
                raise ValueError(f'scene override changed robot/actuator field {field}')
        self.bundle = replace(original, mj_model=model)
        self.model, self.index = model, self.bundle.model_index
        self.geometry = build_geometry_contract(model)
        self.robot_ids = set(np.asarray(self.geometry.robot_geom_ids).tolist())
        self._geometry_bounds = jax.jit(lambda positions, rotations: collision_support_bounds(
            positions, rotations, self.geometry.robot_geom_types, self.geometry.robot_geom_sizes))
        self._euler = jax.jit(quaternion_to_euler)
        self._map_action = jax.jit(lambda action, knee: map_action(action, knee, self.bundle.action_mapping))
        self._make_frame = observable_frame
        checkpoint = (self.source_root / policy_spec['checkpoint']).resolve()
        if not checkpoint.is_relative_to(self.source_root):
            raise ValueError('checkpoint escapes declared source root')
        identity_path = checkpoint / 'identity.json'
        identity = json.loads(identity_path.read_text())
        expected = CheckpointIdentity(
            config_sha256=identity['config_sha256'], xml_sha256=original.xml_sha256,
            actor_frame_fields=tuple(ACTOR_FRAME_FIELDS), actor_task_fields=tuple(ACTOR_TASK_FIELDS),
            action_order=tuple(ACTION_ORDER))
        for name, path in (('identity_sha256', identity_path), ('payload_sha256', checkpoint / 'payload.pkl')):
            if policy_spec.get(name) is not None and _hash(path) != policy_spec[name]:
                raise ValueError(f'checkpoint declared {name} mismatch')
        payload = load_checkpoint(checkpoint, expected=expected)
        self._policy = jax.jit(make_checkpoint_policy(self, payload, deterministic=True))
        self.data = mujoco.MjData(model)
        mesh_dir = Path(compiler.get('meshdir'))
        meshes = {str(mesh_dir / node.get('file')): _hash(mesh_dir / node.get('file'))
                  for node in tree.getroot().findall('.//asset/mesh')}
        git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=self.worktree_root,
                             check=True, text=True, capture_output=True).stdout.strip()
        self.metadata = dict(
            backend='host_mujoco_cpu', mujoco_version=mujoco.__version__, jax_version=jax.__version__,
            source_head=git, source_worktree=str(self.worktree_root),
            dependency_files_sha256={str(path.relative_to(self.worktree_root)): _hash(path)
                                     for path in sorted((self.worktree_root/'JIT/src/jit_dvgc').glob('*.py'))},
            checkpoint_path=str(checkpoint), checkpoint_identity=identity,
            checkpoint_identity_sha256=_hash(identity_path), checkpoint_payload_sha256=_hash(checkpoint/'payload.pkl'),
            original_xml_sha256=original.xml_sha256,
            scene_xml_sha256=hashlib.sha256(self.scene_xml.encode()).hexdigest(), mesh_sha256=meshes,
            host_loader_config=str(config_path), host_loader_config_sha256=_hash(config_path),
            nq=model.nq, nv=model.nv, nu=model.nu, total_mass=float(model.body_mass.sum()),
            payload_mass=original.payload_mass, actuator_names=list(original.actuator_names),
            actuator_ctrlrange=model.actuator_ctrlrange.tolist(), actuator_forcerange=model.actuator_forcerange.tolist(),
            scene_override=dict(scene), action=dict(action_config), sim_dt=.005, control_dt=.020,
            observation_frame_fields=list(ACTOR_FRAME_FIELDS), actor_observation_size=76,
            no_additional_actuator_delay_queue=True, deterministic_policy=True,
            reset_seed_effect='identity only; exact keyframe, no random physical perturbation',
            trigger_adapter='external latch until qualified airborne ascent then descent apex; .05m/s thresholds',
            contact_rule='actual contact record: normal_force > threshold OR dist < -distance_tolerance',
            state_sampling='mj_forward after each mj_step to align state, sensors, contacts and solver forces',
            qualification_not_mjx_replay_equivalence=True)
        self.on_frame = None
        self.reset(0)

    def reset(self, seed):
        from jit_dvgc.observation import initial_history
        initial = self.config['initial']
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, initial['keyframe'])
        if key_id < 0:
            raise ValueError('declared initial keyframe is missing')
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        self.data.qpos[self.index.root_qpos_address] = float(initial['x'])
        self.data.qvel[self.index.root_dof_address] = float(initial['forward_velocity'])
        self.history = initial_history()
        self.last_action = np.zeros(4, dtype=np.float32)
        self.data.ctrl[:] = np.asarray(self._map_action(self.last_action, self.data.qpos[self.index.knee_qpos_address]))
        mujoco.mj_forward(self.model, self.data)
        self.rng = self._jax.random.PRNGKey(int(seed))
        self.control_steps = 0
        self.physics_steps = 0
        self.signal = 0
        self.trigger_command_time = None
        self.qualified = False
        self.phase = 'approach'
        self.adapter = TriggerAdapter(self.config['timing']['airborne_seconds'], self.config['limits']['airborne_clearance'])
        self.policy_inference_seconds = 0.
        self.physics_seconds = 0.
        self.initial_frame = self._physical_frame(0, 0, np.zeros(76, dtype=float))
        return self.initial_frame

    def observation(self):
        from jit_dvgc.observation import actor_observation
        return actor_observation(self.history, self._jp.asarray(self.signal, dtype=self._jp.float32))

    def policy_action(self, signal):
        self.signal = int(signal)
        start = time.monotonic()
        result, _ = self._policy(dict(state=self.observation(),
                                     privileged_state=self._jp.zeros(106, dtype=self._jp.float32)), self.rng)
        action = np.asarray(result, dtype=np.float32)
        self.policy_inference_seconds += time.monotonic() - start
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError('frozen Actor produced an invalid action')
        return np.clip(action, -1., 1.)

    def _physical_frame(self, control_step, substep, actor_input):
        ids = np.asarray(self.geometry.robot_geom_ids)
        bounds = self._geometry_bounds(self.data.geom_xpos[ids], self.data.geom_xmat[ids])
        min_x, max_x, min_z = map(np.asarray, (bounds.min_x, bounds.max_x, bounds.min_z))
        wheel_positions = np.asarray(self.geometry.wheel_geom_positions)
        front_clearance, rear_clearance = min_z[wheel_positions]
        roll, pitch, yaw = map(float, self._euler(self.data.qpos[3:7]))
        finite = all(np.isfinite(value).all() for value in (
            self.data.qpos, self.data.qvel, self.data.sensordata, self.data.ctrl,
            self.data.qacc, self.data.geom_xpos))
        x, y, z = self.data.qpos[:3]
        vx, vy, vz, wx, wy, wz = self.data.qvel[:6]
        return dict(time=float(self.data.time), x=float(x), y=float(y), z=float(z),
                    vx=float(vx), vy=float(vy), vz=float(vz), wx=float(wx), wy=float(wy), wz=float(wz),
                    roll=roll, pitch=pitch, yaw=yaw, finite=bool(finite),
                    front_clearance=float(front_clearance), rear_clearance=float(rear_clearance),
                    robot_back_x=float(min(min_x)), robot_front_x=float(max(max_x)),
                    obstacle_relative_x=float(self.config['scene']['front_x'] - max(max_x)),
                    structure_clearance=float(min(min_z) - self.config['scene']['height']),
                    qpos=self.data.qpos.tolist(), qvel=self.data.qvel.tolist(),
                    action=self.last_action.tolist(), ctrl=self.data.ctrl.tolist(),
                    actor_input=np.asarray(actor_input).tolist(), signal=self.signal,
                    trigger_command_time=self.trigger_command_time,
                    control_step=control_step, substep=substep, phase=self.phase,
                    **read_contacts(self.model, self.data, self.index, self.robot_ids, self.config['limits']))

    def step(self, triggered):
        from jit_dvgc.observation import ObservableGeometry, advance_history
        if triggered and self.trigger_command_time is None:
            self.trigger_command_time = float(self.data.time)
        signal = self.adapter.signal(triggered)
        action = self.policy_action(signal)
        actor_input = np.asarray(self.observation())
        self.last_action = action
        self.data.ctrl[:] = np.asarray(self._map_action(action, self.data.qpos[self.index.knee_qpos_address]))
        self.control_steps += 1
        rows = []
        for substep in range(1, 5):
            start = time.monotonic()
            mujoco.mj_step(self.model, self.data)
            self.physics_steps += 1
            mujoco.mj_forward(self.model, self.data)
            self.physics_seconds += time.monotonic() - start
            row = self._physical_frame(self.control_steps, substep, actor_input)
            rows.append(row)
            keep_going = self.on_frame(row) if self.on_frame is not None else True
            self.adapter.observe(row, qualified=self.qualified, time=row['time'])
            if not keep_going:
                break
        row = rows[-1]
        frame = self._make_frame(self.data, self.index,
                                ObservableGeometry(obstacle_relative_x=row['obstacle_relative_x'],
                                                   root_height=row['z'], roll=row['roll'], pitch=row['pitch'], yaw=row['yaw']),
                                self._jp.asarray(action), self._jp.asarray(True))
        self.history = advance_history(self.history, frame)
        return rows


def run_case(config, policy_spec, source_root, case, *, runtime=None):
    """Execute one bounded case; reuse a runtime for the same config/policy.

    TaskMonitor is called inside each physical substep so no physics is stepped
    after a true terminal. Partial controls are charged using ceil(substeps/4).
    Exceptions after stepping retain already observed rows and exact counters.
    """
    from .oracle import TaskMonitor
    start = time.monotonic()
    rt = runtime or QualificationRuntime(config, policy_spec, source_root)
    if rt.config != config or rt.policy_spec != policy_spec:
        raise ValueError('cached runtime config/policy differs from requested case')
    rt.reset(case['seed'])
    rows = []
    qualifier = ApproachQualification(config)
    monitor = TaskMonitor(config)
    outcome = dict(status='ongoing', done=False, reason='ongoing')
    command_time = None
    delay = case.get('trigger_delay_seconds')
    qualified = qualifier.update(rt.initial_frame)
    outcome = monitor.update(rt.initial_frame, triggered=False, qualified=qualified)
    rt.initial_frame.update(qualified=qualified, triggered=False, oracle_status=outcome['status'],
                            oracle_reason=outcome['reason'])
    rows.append(rt.initial_frame)

    def on_frame(row):
        nonlocal outcome
        rt.qualified = qualifier.update(row)
        outcome = monitor.update(row, triggered=command_time is not None, qualified=rt.qualified)
        row.update(qualified=rt.qualified, triggered=command_time is not None,
                   oracle_status=outcome['status'], oracle_reason=outcome['reason'])
        if rt.qualified:
            row['phase'] = 'post_trigger' if command_time is not None else 'qualified_approach'
        rows.append(row)
        return not outcome['done']

    rt.on_frame = on_frame
    error = None
    try:
        episode_steps = math.ceil(config['timing']['episode_seconds'] / config['timing']['control_dt'])
        step_limit = min(episode_steps, int(config['budget']['max_control_steps']))
        while rt.control_steps < step_limit and not outcome['done']:
            if time.monotonic() - start > config['budget']['wall_seconds']:
                outcome = dict(status='unknown', done=True, reason='engineering_wall_timeout')
                break
            if qualifier.qualified_time is None and rt.data.time >= config['timing']['approach_timeout_seconds'] - 1e-10:
                outcome = monitor.finish()
                break
            if (command_time is None and delay is not None and qualifier.qualified_time is not None
                    and rt.data.time - qualifier.qualified_time + 1e-10 >= float(delay)):
                command_time = float(rt.data.time)
            rt.step(command_time is not None)
        if not outcome['done']:
            if step_limit < episode_steps:
                outcome = dict(status='unknown', done=True, reason='engineering_control_budget_exhausted')
            else:
                outcome = monitor.finish()
    except Exception as exc:
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        outcome = dict(status='unknown', done=True, reason='runtime_error')
    finally:
        rt.on_frame = None
    summary = dict(outcome)
    summary.update(case_name=case['name'], policy_name=policy_spec['name'], seed=case['seed'],
                   trigger_delay_seconds=delay, qualified=qualifier.qualified_time is not None,
                   qualification_time=qualifier.qualified_time, trigger_command_time=command_time,
                   adapter_apex_time=rt.adapter.apex_time, control_steps=rt.control_steps,
                   charged_control_steps=math.ceil(rt.physics_steps / 4), physics_steps=rt.physics_steps,
                   elapsed_simulation_seconds=float(rt.data.time), wall_seconds=time.monotonic()-start,
                   policy_inference_seconds=rt.policy_inference_seconds, physics_seconds=rt.physics_seconds,
                   runtime_metadata=rt.metadata, error=error)
    return summary, rows
