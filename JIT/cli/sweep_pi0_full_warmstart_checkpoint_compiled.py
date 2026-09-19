#!/usr/bin/env python3
"""Warp-safe fast wrapper for one B intermediate-checkpoint quickcheck.

This wrapper reuses the existing checkpoint/bank/report plumbing from
``sweep_pi0_full_warmstart_checkpoints.py`` but replaces the broken outer-vmap
rollout with a fully compiled device-side ``lax.map``.  Each trajectory remains
an unbatched MuJoCo-Warp world, which is required by the current MJX-Warp FFI,
while the Python per-state/per-tick dispatch and per-tick host synchronization
are removed.

No training is performed and the pi0 baseline outcomes are still reused from
the completed B final gate.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import sweep_pi0_full_warmstart_checkpoints as base


def _make_compiled_map_rollout(env, policy, max_ticks: int):
    """Compile many independent rollouts without vmapping MuJoCo-Warp Data."""

    max_ticks = int(max_ticks)

    def rollout_one(initial_state, seed, start_phase):
        del start_phase  # classification remains on host, matching the gate

        apex_seen = jnp.asarray(initial_state.info["up_events"].apex_seen, dtype=bool)
        phase_transitioned = jnp.asarray(
            initial_state.info["phase_transitioned"], dtype=bool
        )
        recovery_success = jnp.asarray(
            initial_state.info["down_events"].recovery_success, dtype=bool
        )
        expert_switching_seen = jnp.asarray(
            initial_state.info["expert_switching_used"], dtype=bool
        )

        carry = (
            jnp.asarray(0, dtype=jnp.int32),
            initial_state,
            apex_seen,
            phase_transitioned,
            recovery_success,
            expert_switching_seen,
        )

        def cond_fn(value):
            tick, state, *_ = value
            return jnp.logical_and(
                tick < max_ticks,
                jnp.logical_not(jnp.asarray(state.done, dtype=bool)),
            )

        def body_fn(value):
            (
                tick,
                state,
                apex_seen,
                phase_transitioned,
                recovery_success,
                expert_switching_seen,
            ) = value
            key = jax.random.fold_in(jax.random.PRNGKey(seed), tick)
            result = policy(state.obs, key)
            action = result[0] if isinstance(result, tuple) else result
            state = env.step(state, action)
            return (
                tick + jnp.asarray(1, dtype=jnp.int32),
                state,
                jnp.logical_or(
                    apex_seen,
                    jnp.asarray(state.info["up_events"].apex_seen, dtype=bool),
                ),
                jnp.logical_or(
                    phase_transitioned,
                    jnp.asarray(state.info["phase_transitioned"], dtype=bool),
                ),
                jnp.logical_or(
                    recovery_success,
                    jnp.asarray(state.info["down_events"].recovery_success, dtype=bool),
                ),
                jnp.logical_or(
                    expert_switching_seen,
                    jnp.asarray(state.info["expert_switching_used"], dtype=bool),
                ),
            )

        (
            ticks,
            final_state,
            apex_seen,
            phase_transitioned,
            recovery_success,
            expert_switching_seen,
        ) = jax.lax.while_loop(cond_fn, body_fn, carry)

        # Return only the small terminal summary.  Returning the entire MJX State
        # would materialize large contact/constraint buffers for every trajectory.
        return (
            ticks,
            jnp.asarray(final_state.done, dtype=bool),
            jnp.asarray(final_state.info["success"], dtype=bool),
            jnp.asarray(final_state.info["physical_failure"], dtype=bool),
            jnp.asarray(final_state.info["timeout"], dtype=bool),
            apex_seen,
            phase_transitioned,
            recovery_success,
            expert_switching_seen,
        )

    def run_many(stacked_state, seeds, start_phases):
        # lax.map slices one initial State at a time before env.step.  This keeps
        # MuJoCo-Warp Data unbatched at the FFI boundary, unlike outer vmap.
        def mapped(args):
            state, seed, start_phase = args
            return rollout_one(state, seed, start_phase)

        return jax.lax.map(mapped, (stacked_state, seeds, start_phases))

    return jax.jit(run_many)


def _candidate_results_from_compiled(
    output,
    *,
    start_phases: list[int],
    real_count: int,
    max_ticks: int,
):
    (
        ticks,
        done,
        terminal_success,
        physical_failure,
        timeout,
        apex_seen,
        phase_transitioned,
        recovery_success,
        expert_switching_seen,
    ) = jax.device_get(output)

    ticks = np.asarray(ticks)[:real_count]
    done = np.asarray(done).astype(bool)[:real_count]
    terminal_success = np.asarray(terminal_success).astype(bool)[:real_count]
    physical_failure = np.asarray(physical_failure).astype(bool)[:real_count]
    timeout = np.asarray(timeout).astype(bool)[:real_count]
    apex_seen = np.asarray(apex_seen).astype(bool)[:real_count]
    phase_transitioned = np.asarray(phase_transitioned).astype(bool)[:real_count]
    recovery_success = np.asarray(recovery_success).astype(bool)[:real_count]
    expert_switching_seen = np.asarray(expert_switching_seen).astype(bool)[:real_count]

    if bool(np.any(expert_switching_seen)):
        raise ValueError("compiled checkpoint rollout unexpectedly used expert switching")

    results = []
    for index in range(real_count):
        interactions = int(ticks[index])
        positive, outcome = base.gate_mod.classify_unified_continuation_outcome(
            start_phase=int(start_phases[index]),
            terminal_success=bool(terminal_success[index]),
            physical_failure=bool(physical_failure[index]),
            timeout=bool(timeout[index]),
            done=bool(done[index]),
            apex_seen=bool(apex_seen[index]),
            phase_transitioned=bool(phase_transitioned[index]),
            recovery_success=bool(recovery_success[index]),
            reached_rollout_horizon=(
                interactions >= int(max_ticks) and not bool(done[index])
            ),
        )
        results.append(
            {
                "success": bool(positive),
                "outcome_class": str(outcome),
                "environment_interactions": interactions,
            }
        )
    return results


def main() -> int:
    # Monkeypatch only the execution engine.  All checkpoint identity checks,
    # bank construction, baseline reuse, outcome classification, report paths,
    # and summary logic stay exactly in the existing sweep implementation.
    base._make_batched_rollout = _make_compiled_map_rollout
    base._candidate_results_from_batch = _candidate_results_from_compiled
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
