"""Optional fused canonical restoration, retaining single-world Warp forward.

This backend changes dispatch scheduling, not snapshot semantics. It requires
bounded GPU state and suffix parity evidence before selection by an experiment.
The default remains serial. Returned states must still pass through
``prepare_parallel_worlds`` before a vectorized physics continuation.
"""
from typing import Any, Sequence
import jax
from .unified_envelope_snapshot import (
    UnifiedEnvelopeSnapshot, _restore_unified_envelope_payload,
    restore_unified_envelope_snapshot, validate_unified_envelope_snapshot_runtime,
)
from .continuation.device_rollout import stack_worlds

RESTORE_FIELDS = (
    'qpos', 'qvel', 'ctrl', 'observation_fifo', 'history_valid_count',
    'last_action', 'rng', 'up_events', 'down_events', 'active_phase',
    'start_phase', 'phase_transitioned', 'episode_step', 'phase_episode_step',
    'episode_return', 'reset_from_soft_tube', 'source_tick',
    'parent_group_index', 'tube_entry_index', 'tube_global_index',
)


def snapshot_restore_payload(snapshot: UnifiedEnvelopeSnapshot) -> dict:
    """Numeric dynamic inputs; provenance strings stay in mandatory host checks."""
    return {name: getattr(snapshot, name) for name in RESTORE_FIELDS}


def _fused_restore(env):
    # Store on this environment: different models/reward configuration cannot
    # accidentally share a compiled closure. No global cache holds GPU worlds.
    restore = getattr(env, '_fused_snapshot_restore', None)
    if restore is None:
        restore = jax.jit(lambda payload: _restore_unified_envelope_payload(payload, env))
        env._fused_snapshot_restore = restore
    return restore


def restore_snapshot_batch(snapshots: Sequence[UnifiedEnvelopeSnapshot], env: Any,
                           *, backend: str = 'serial', fresh: bool = True):
    """Restore complete snapshots, freshen canonical counters, then Warp-stack.

    ``serial`` exactly follows the historical eager path; ``fused`` compiles the
    shared numerical reconstruction once and dispatches it once per snapshot.
    No vmap of forward and no new simulator integration steps occur here.
    """
    if not snapshots:
        raise ValueError('snapshot restoration requires a nonempty batch')
    if backend not in ('serial', 'fused'):
        raise ValueError('unsupported snapshot restore backend: ' + str(backend))
    for snapshot in snapshots:
        validate_unified_envelope_snapshot_runtime(snapshot, env)
    restore = _fused_restore(env) if backend == 'fused' else None
    from .unified_continuation_labels import fresh_unified_continuation_state
    states = []
    for snapshot in snapshots:
        state = (restore(snapshot_restore_payload(snapshot)) if restore is not None
                 else restore_unified_envelope_snapshot(snapshot, env))
        if fresh:
            state = fresh_unified_continuation_state(state)
        states.append(state)
    return stack_worlds(states)
