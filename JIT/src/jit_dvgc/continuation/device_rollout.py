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


def make_device_rollout(policy, step, max_ticks):
    """One compiled call per batch; terminate each lane at its first endpoint.

    vmap may execute the step for inactive lanes. Their returned state is frozen,
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
            next_state = jax.vmap(step)(state, safe_actions)
            good = jax.vmap(finite)(next_state, actions) & ~next_state.info['expert_switching_used']
            bad = bad | (live & ~good)
            next_flags = jax.vmap(event_flags)(next_state)
            flags = flags | (live[:, None] & next_flags)
            def select(new, old):
                shape = (size,) + (1,) * (new.ndim - 1)
                return jp.where(live.reshape(shape), new, old)
            state = jax.tree_util.tree_map(select, next_state, state)
            counts = counts + live.astype(jp.int32)
            live = live & good & ~next_state.done.astype(bool) & ~flags[:, 3]
            return tick + 1, state, counts, live, bad, flags

        tick, states, counts, _, bad, flags = jax.lax.while_loop(condition, body, initial)
        return states, counts, bad, flags, tick * size

    return jax.jit(run)
