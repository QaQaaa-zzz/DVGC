"""Deduplicate canonical restoration only within one preserved-context batch."""
import json
import math
from pathlib import Path

import jax.numpy as jp
import numpy as np

from . import unified_envelope_snapshot as snapshots


def restore_preserved_snapshot_rows(rows, env, *, deduplicate=True):
    """Return ordered immutable states and an auditable restoration receipt.

    Every row is loaded and validated before any numerical reconstruction. Only
    exact path, payload, context and recorded-time matches share a restoration;
    the cache expires with this call. ``deduplicate=False`` provides the same
    canonical numerical path with one restore per row for matched validation.
    """
    if type(deduplicate) is not bool:
        raise ValueError('deduplicate must be a boolean')
    if not rows:
        raise ValueError('preserved snapshot batch must not be empty')
    validated = []
    for row in rows:
        directory = Path(row['snapshot']).absolute()
        identity_path = directory / 'identity.json'
        identity = json.loads(identity_path.read_text(encoding='utf-8'))
        snapshot = snapshots.load_unified_envelope_snapshot(directory)
        if json.loads(identity_path.read_text(encoding='utf-8')) != identity:
            raise ValueError('suffix snapshot identity changed during validation')
        context = snapshots.snapshot_context_sha256(snapshot)
        if context != row['snapshot_context_sha256']:
            raise ValueError('suffix snapshot identity changed')
        snapshots.validate_unified_envelope_snapshot_runtime(snapshot, env)
        if 'snapshot_time' not in row:
            raise ValueError('retention snapshot requires recorded time')
        value = row['snapshot_time']
        if isinstance(value, (bool, str, bytes)):
            raise ValueError('retention snapshot time must be finite and nonnegative')
        try:
            recorded_time = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('invalid retention snapshot time') from exc
        if not math.isfinite(recorded_time) or recorded_time < 0:
            raise ValueError('retention snapshot time must be finite and nonnegative')
        key = (str(directory), identity['payload_sha256'], context, recorded_time.hex())
        validated.append((snapshot, recorded_time, key))

    states, restored, row_to_restore = [], {}, []
    restore_count = 0
    for snapshot, recorded_time, key in validated:
        if deduplicate and key in restored:
            state, restore_index = restored[key]
        else:
            state = snapshots.restore_unified_envelope_snapshot(snapshot, env)
            if not np.array_equal(np.asarray(state.obs['state']), np.asarray(snapshot.observation)):
                raise ValueError('restored Actor observation/history differs')
            state = state.replace(data=state.data.replace(time=jp.asarray(recorded_time, jp.float32)))
            restore_index = restore_count
            restore_count += 1
            if deduplicate:
                restored[key] = (state, restore_index)
        states.append(state)
        row_to_restore.append(restore_index)
    return states, dict(rows=len(states), restores=restore_count,
        reused_rows=len(states)-restore_count, deduplicate=deduplicate,
        row_to_restore=row_to_restore)
