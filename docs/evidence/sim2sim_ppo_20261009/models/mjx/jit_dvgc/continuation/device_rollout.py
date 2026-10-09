"""Bounded device-side first-landing rollouts, with explicit inactive-lane cost.

No reset or interpolation occurs here. Host-side snapshot/identity checks remain
mandatory. Batched execution must pass the production execution comparison before
it is used for an experiment; mathematical vmap is not a GPU equivalence claim.
"""
from __future__ import annotations

import jax
import jax.numpy as jp


def event_flags(state):
    return jp.stack((state.info['up_events'].apex_seen,
                     state.info['phase_transitioned'],
                     state.info['down_events'].recovery_success,
                     state.info['down_events'].valid_contact_seen)).astype(bool)


def finite(state, action):
    return (jp.all(jp.isfinite(state.data.qpos)) & jp.all(jp.isfinite(state.data.qvel))
            & jp.all(jp.isfinite(state.obs['state'])) & jp.all(jp.isfinite(action)))


def _shared_warp(path):
    # MuJoCo registers DataWarp as a custom vmappable: these leaves are global
    # scratch/capacity buffers, not per-world state. Follow the installed schema.
    if len(path) < 2 or getattr(path[-2], 'name', None) != '_impl':
        return False
    from mujoco.mjx.warp.types import DATA_NON_VMAP
    return getattr(path[-1], 'name', None) in DATA_NON_VMAP


def stack_worlds(states):
    """Stack restored states while retaining Warp's shared scratch layout."""
    return jax.tree_util.tree_map_with_path(
        lambda path, *xs: xs[0] if _shared_warp(path) else jp.stack(xs), *states)


def take_world(states, index):
    """Extract logical state; shared scratch is not an individual contact log."""
    return jax.tree_util.tree_map_with_path(
        lambda path, x: x if _shared_warp(path) else x[index], states)


def repeat_worlds(states, count):
    """Capacity-test replication only: these are not new scientific candidates."""
    if type(count) is not int or count <= 0:
        raise ValueError('positive world count required')
    indices = jp.arange(count) % states.obs['state'].shape[0]
    return jax.tree_util.tree_map_with_path(
        lambda path, x: x if _shared_warp(path) else x[indices], states)


def prepare_parallel_worlds(states, env, count):
    """Allocate global Warp scratch for count worlds; preserve logical state."""
    impl = getattr(states.data, '_impl', None)
    if impl is None or not hasattr(impl, 'naconmax'):
        return states
    from mujoco import mjx
    from mujoco.mjx.warp.types import DATA_NON_VMAP
    scratch = mjx.make_data(env.mj_model, impl='warp',
        naconmax=max(int(env.resolved_config.model['naconmax']), count*64),
        naccdmax=max(env._reset_data_naccdmax() or 0, count*64),
        njmax=int(env.resolved_config.model['njmax']))
    data = states.data.replace(_impl=impl.replace(
        **{name: getattr(scratch._impl, name) for name in DATA_NON_VMAP}))
    return states.replace(data=data, info={**states.info,
        'parallel_capacity_exceeded': jp.zeros(count, dtype=bool)})


def checked_physics_step(model, data, action, n_substeps):
    """Accumulate scratch saturation across every physics substep, not just the last."""
    from mujoco import mjx
    def single(carry, _):
        data, exceeded = carry
        data = mjx.step(model, data.replace(ctrl=action))
        impl = data._impl
        overflow = (jp.any(impl.nacon >= impl.naconmax)
                    | jp.any(impl.ncollision >= impl.naccdmax)
                    | jp.any(data.nefc >= impl.njmax))
        return (data, exceeded | overflow), None
    return jax.lax.scan(single, (data, jp.asarray(False)), (), n_substeps)[0]


def make_device_rollout(policy, step, max_ticks, *, vectorized=False):
    """One compiled call per batch; terminate each lane at its first endpoint.

    The device map may execute the step for inactive lanes. Their returned state is frozen,
    but n_lanes * loop_ticks is conservatively counted as simulator work.
    """
    if max_ticks <= 0:
        raise ValueError('positive horizon required')

    def run(states, candidate_keys):
        size = candidate_keys.shape[0]
        flags = jax.vmap(event_flags)(states)
        active = jp.ones(size, dtype=bool)
        initial = (jp.asarray(0, jp.int32), states, jp.zeros(size, jp.int32),
                   active, jp.zeros(size, bool), flags)

        def condition(carry):
            tick, _, _, live, _, _ = carry
            return (tick < max_ticks) & jp.any(live)

        def body(carry):
            tick, state, counts, live, bad, flags = carry
            keys = jax.vmap(lambda key: jax.random.fold_in(key, tick))(candidate_keys)
            result = jax.vmap(policy)(state.obs, keys)
            actions = result[0] if isinstance(result, tuple) else result
            if actions.shape != (size, 4):
                raise ValueError('policy must return four actions per lane')
            action_ok = jp.all(jp.isfinite(actions), axis=1)
            safe_actions = jp.where((live & action_ok)[:, None], actions, 0.)
            # Legacy map keeps single-world scratch. The vectorized path uses
            # DataWarp's custom batching with properly sized shared scratch.
            if vectorized:
                next_state = jax.vmap(step)(state, safe_actions)
            else:
                next_state = jax.lax.map(lambda pair: step(pair[0], pair[1]),
                                         (state, safe_actions))
            good = jax.vmap(finite)(next_state, actions) & ~next_state.info['expert_switching_used']
            good = good & ~next_state.info.get('parallel_capacity_exceeded', jp.asarray(False))
            bad = bad | (live & ~good)
            next_flags = jax.vmap(event_flags)(next_state)
            flags = flags | (live[:, None] & next_flags)
            def select(new, old):
                shape = (size,) + (1,) * (new.ndim - 1)
                return jp.where(live.reshape(shape), new, old)
            if vectorized:
                state = jax.tree_util.tree_map_with_path(
                    lambda path, new, old: new if _shared_warp(path) else select(new, old),
                    next_state, state)
            else:
                state = jax.tree_util.tree_map(select, next_state, state)
            counts = counts + live.astype(jp.int32)
            live = live & good & ~next_state.done.astype(bool) & ~flags[:, 3]
            return tick + 1, state, counts, live, bad, flags

        tick, states, counts, _, bad, flags = jax.lax.while_loop(condition, body, initial)
        return states, counts, bad, flags, tick * size

    return jax.jit(run)
