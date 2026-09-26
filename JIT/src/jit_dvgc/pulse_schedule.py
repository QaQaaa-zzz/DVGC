"""Declared pulse modes and causal, per-lane physical event scheduling."""


def controller_mode(spec):
    mode = spec.get('controller_mode', 'learned_residual')
    if mode == 'learned':
        mode = 'learned_residual'
    if mode not in ('learned_residual', 'fixed_random'):
        raise ValueError('unsupported pulse controller_mode')
    return mode


def initial_velocity_randomization(spec):
    """Validate the declared reset velocity-noise contract.

    Values are symmetric half-ranges in SI units.  The zero-default keeps
    historical fixed-reset experiments byte-for-byte compatible.
    """
    raw = spec.get('initial_velocity_randomization', {}) or {}
    if not isinstance(raw, dict):
        raise ValueError('initial_velocity_randomization must be a mapping')
    defaults = {
        'forward_m_s': 0.0, 'lateral_m_s': 0.0,
        'angular_rad_s': 0.0, 'joint_rad_s': 0.0,
    }
    result = {}
    import math
    for name, default in defaults.items():
        value = raw.get(name, default)
        if type(value) not in (int, float) or not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f'invalid initial velocity randomization: {name}')
        result[name] = float(value)
    return result


def sample_initial_velocity_noise(keys, model_index, qvel_size, spec):
    """Sample per-lane qvel noise without changing qpos or root x."""
    import jax
    import jax.numpy as jp
    bounds = initial_velocity_randomization(spec)
    fields = (
        (model_index.root_dof_address + 0, bounds['forward_m_s']),
        (model_index.root_dof_address + 1, bounds['lateral_m_s']),
        (model_index.root_dof_address + 3, bounds['angular_rad_s']),
        (model_index.root_dof_address + 4, bounds['angular_rad_s']),
        (model_index.root_dof_address + 5, bounds['angular_rad_s']),
        (model_index.steering_dof_address, bounds['joint_rad_s']),
        (model_index.hip_dof_address, bounds['joint_rad_s']),
        (model_index.knee_dof_address, bounds['joint_rad_s']),
    )
    if any(i < 0 or i >= qvel_size for i, _ in fields):
        raise ValueError('initial velocity randomization index outside qvel')
    def one(key):
        noise = jp.zeros((qvel_size,), dtype=jp.float32)
        subkeys = jax.random.split(key, len(fields))
        for subkey, (index, bound) in zip(subkeys, fields):
            noise = noise.at[index].set(jax.random.uniform(subkey, (), minval=-bound, maxval=bound))
        return noise
    return jax.vmap(one)(keys)


def collection_steps(spec, delays, event):
    """Opt-in fixed-random evaluation runs through the declared episode horizon."""
    if spec.get('full_episode_rollout'):
        if controller_mode(spec) != 'fixed_random':
            raise ValueError('full episode collection is a fixed-random evaluation only')
        return spec['horizon']
    return spec['horizon'] if event else int(delays.max()) + spec['pulse_steps']


def descent_clearance(spec):
    import math
    value = spec.get('pulse_descent_clearance', .10)
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('finite positive pulse_descent_clearance required')
    return float(value)


def selected_event(spec):
    schedule = spec.get('pulse_event_schedule')
    if schedule and spec.get('pulse_batch_mode')=='mixed' and not spec.get('nominal_source_rollout'):
        raise ValueError('mixed fixed onsets cannot combine event scheduling')
    if schedule is None:
        return None
    if (not isinstance(schedule, list) or not schedule or
            any(event not in ('start', 'liftoff', 'apex', 'descent') for event in schedule)):
        raise ValueError('invalid pulse_event_schedule')
    if spec.get('nominal_source_rollout'):
        return None
    if type(spec.get('round_index', 0)) is not int or spec.get('round_index', 0) < 0:
        raise ValueError('nonnegative pulse round_index required')
    return schedule[spec.get('round_index', 0) % len(schedule)]


def event_ready(event, *, tick, front_clearance, rear_clearance,
                previous_vertical_velocity, vertical_velocity, valid_contact_seen,
                min_airborne_clearance, min_descent_velocity, descent_clearance=.10):
    """Evaluate the observed pre-action state, never a post-action phase label.

    Liftoff requires both wheels above the declared airborne threshold. Apex is
    the first observed positive-to-nonpositive vertical velocity crossing while
    airborne; descent requires negative velocity and the near-contact clearance
    threshold while both wheels remain airborne. These
    are control-tick observations, not interpolated substep event times.
    """
    import jax.numpy as jp
    airborne = ((jp.asarray(front_clearance) > min_airborne_clearance) &
                (jp.asarray(rear_clearance) > min_airborne_clearance) &
                ~jp.asarray(valid_contact_seen, bool))
    if event == 'start':
        return jp.full_like(airborne, tick == 0, dtype=bool)
    if event == 'liftoff':
        return airborne
    if event == 'apex':
        return airborne & (jp.asarray(previous_vertical_velocity) > 0) & (jp.asarray(vertical_velocity) <= 0)
    if event == 'descent':
        near_contact = jp.minimum(front_clearance, rear_clearance) <= descent_clearance
        return airborne & near_contact & (jp.asarray(vertical_velocity) <= -min_descent_velocity)
    raise ValueError('unsupported pulse event')


def pulse_activity(trigger_tick, applied_steps, ready, alive, tick, pulse_steps):
    """Latch each lane once and return the action mask before advancing physics."""
    import jax.numpy as jp
    trigger_tick = jp.asarray(trigger_tick)
    trigger_tick = jp.where((trigger_tick < 0) & ready & alive, tick, trigger_tick)
    active = alive & (trigger_tick >= 0) & (jp.asarray(applied_steps) < pulse_steps)
    return trigger_tick, active


def lane_onsets(spec, index):
    """Balanced fixed onset allocation, with rotating remainder across batches."""
    import numpy as np
    from .pulse_exploration import pulse_delay
    mode=spec.get('pulse_batch_mode','single')
    if mode not in ('single','mixed'):
        raise ValueError('unsupported pulse_batch_mode')
    n=spec['num_envs']
    if type(n) is not int or n<1:raise ValueError('positive num_envs required')
    if mode=='single' or spec.get('nominal_source_rollout'):
        return np.full(n,pulse_delay(spec,index),np.int32)
    if spec.get('pulse_event_schedule'):raise ValueError('mixed fixed onsets cannot combine event scheduling')
    pulse_delay(spec,index)  # Validate the unchanged fixed-onset contract.
    schedule=np.asarray(spec['pulse_start_schedule'],np.int32)
    return schedule[(np.arange(n)+index)%len(schedule)]
