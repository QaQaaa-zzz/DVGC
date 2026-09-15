"""Declared pulse modes and causal, per-lane physical event scheduling."""


def controller_mode(spec):
    mode = spec.get('controller_mode', 'learned_residual')
    if mode == 'learned':
        mode = 'learned_residual'
    if mode not in ('learned_residual', 'fixed_random'):
        raise ValueError('unsupported pulse controller_mode')
    return mode


def descent_clearance(spec):
    import math
    value = spec.get('pulse_descent_clearance', .10)
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('finite positive pulse_descent_clearance required')
    return float(value)


def selected_event(spec):
    schedule = spec.get('pulse_event_schedule')
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
