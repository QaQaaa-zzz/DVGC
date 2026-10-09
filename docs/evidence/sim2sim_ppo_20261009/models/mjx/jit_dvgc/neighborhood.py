"""Immutable CPU lookup of evidence known before a collection batch.

This module does not add evidence or infer timestamps. The caller must pass only
completed earlier batches and keep this snapshot fixed for all rollout steps.
Counts and fractions refer to unique source/phase/context records, not visits.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy.spatial import cKDTree

FIELDS = (
    'root_x_m', 'root_y_m', 'root_z_m', 'root_vx_mps', 'root_vy_mps',
    'root_vz_mps', 'roll_deg', 'pitch_deg', 'yaw_deg', 'root_wx_degps',
    'root_wy_degps', 'root_wz_degps',
)
DEFAULT_CONFIG = dict(
    neighbors=16,
    medium_halfwidths=(.2, .05, .1, .4, .2, .6, 2., 5., 5., 10., 20., 20.),
    far_scale=2.,
)


def _config(config):
    values = dict(DEFAULT_CONFIG)
    if config is not None:
        unknown = set(config) - set(DEFAULT_CONFIG) - {
            'feature_dim', 'summary_dim', 'base_dim', 'stats_dim'}
        if unknown:
            raise ValueError(f'unknown neighborhood configuration: {sorted(unknown)}')
        values.update(config)
    k = values['neighbors']
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError('neighbors must be a positive integer')
    widths = np.asarray(values['medium_halfwidths'], dtype=np.float64)
    if widths.shape != (12,) or not np.all(np.isfinite(widths)) or np.any(widths <= 0):
        raise ValueError('medium_halfwidths must contain twelve finite positive widths')
    if not np.isfinite(values['far_scale']) or values['far_scale'] != 2:
        raise ValueError('far_scale must be 2 for the two-radius observation contract')
    for name, expected in [('feature_dim', 17), ('base_dim', 106), ('stats_dim', 8)]:
        if name in values and values[name] != expected:
            raise ValueError(f'{name} must be {expected}')
    if 'summary_dim' in values and (not isinstance(values['summary_dim'], (int, np.integer))
                                  or isinstance(values['summary_dim'], (bool, np.bool_))
                                  or values['summary_dim'] <= 0):
        raise ValueError('summary_dim must be a positive integer')
    values['medium_halfwidths'] = tuple(float(v) for v in widths)
    return values


def augmentation_dim(config=None):
    """Number of neighborhood features appended to the legacy observation."""
    return int(_config(config)['neighbors']) * 17 + 8


def _phase(value):
    if isinstance(value, str) and value in ('upstream', 'downstream'):
        return 0 if value == 'upstream' else 1
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and value in (0, 1):
        return int(value)
    raise ValueError('phase must be upstream/downstream or integer 0/1')


def _known(label):
    # Unknown, missing, and evaluator errors never become negative evidence.
    return label if label in (0, 1) else None


def _freeze_array(values):
    array = np.asarray(values, dtype=np.float64)
    # A bytes backing prevents writes even if writeable is later toggled.
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class _PhaseIndex:
    coordinates: np.ndarray
    flags: np.ndarray
    tree: cKDTree


@dataclass(frozen=True, init=False)
class FrozenNeighborhood:
    """A source-specific, defensively copied lookup, constructed once per batch.

    Rows require ``coordinates`` (mapping in FIELDS units, or a length-12 vector),
    ``phase`` and ``snapshot_context_sha256`` (``context_sha256`` also accepted).
    Explicit source identity takes precedence over the first evaluator attempt.
    Later evaluator actors do not retag a row. Duplicate contexts merge known
    outcomes separately for current and repair evidence; contradictory outcomes
    in either category quarantine that category to unknown. Duplicate context
    coordinates must agree exactly. No visit counts enter this observation.
    """
    source_actor_sha256: str
    config: Mapping
    record_count: int
    _widths: np.ndarray
    _indices: tuple

    def __init__(self, rows, source_actor_sha256, config=None):
        values = _config(config)
        if not isinstance(source_actor_sha256, str) or not source_actor_sha256:
            raise ValueError('source_actor_sha256 must be a nonempty string')
        widths = _freeze_array(values['medium_halfwidths'])
        records = {}
        for row in rows:
            attempts = row.get('attempts') or []
            source = (row['source_actor_sha256'] if 'source_actor_sha256' in row
                      else attempts[0].get('actor_sha256') if attempts else None)
            if source != source_actor_sha256:
                continue
            phase = _phase(row['phase'])
            context = row.get('snapshot_context_sha256', row.get('context_sha256'))
            if not isinstance(context, str) or not context:
                raise ValueError('source rows need a complete context identity')
            raw = row['coordinates']
            coordinates = np.asarray([raw[name] for name in FIELDS] if isinstance(raw, Mapping)
                                     else raw, dtype=np.float64)
            if coordinates.shape != (12,) or not np.all(np.isfinite(coordinates)):
                raise ValueError('record coordinates must be twelve finite values')
            key = (phase, context)
            if key not in records:
                records[key] = (coordinates.copy(), set(), set())
            elif not np.array_equal(records[key][0], coordinates):
                raise ValueError('duplicate context has inconsistent coordinates')
            current = _known(row['initial_label'] if 'initial_label' in row
                             else attempts[0].get('label') if attempts else None)
            repair = _known(row.get('label')) if row.get('learning_attempted', False) else None
            if current is not None:
                records[key][1].add(current)
            if repair is not None:
                records[key][2].add(repair)
        indices = []
        for phase in (0, 1):
            # Canonical context order supplies the final tie break independently
            # of insertion order, including equal coordinates and distances.
            selected = [records[key] for key in sorted(records) if key[0] == phase]
            if not selected:
                indices.append(None)
                continue
            coordinates = _freeze_array([r[0] / widths for r in selected])
            flags = _freeze_array([[float(r[1] == {1}), float(r[1] == {0}),
                                    float(r[2] == {1}), float(r[2] == {0})] for r in selected])
            indices.append(_PhaseIndex(coordinates, flags, cKDTree(coordinates, copy_data=True)))
        object.__setattr__(self, 'source_actor_sha256', source_actor_sha256)
        object.__setattr__(self, 'config', MappingProxyType(values))
        object.__setattr__(self, 'record_count', len(records))
        object.__setattr__(self, '_widths', widths)
        object.__setattr__(self, '_indices', tuple(indices))

    def query(self, coordinates, phases):
        """Return float32 [N, K*17+8]; relative vectors point query -> record.

        Chebyshev distance in medium halfwidth units defines both neighbor
        ordering and inclusive cutoffs (near <=1, far <=2). The far set contains
        the near set. Nearest distances use the same units and empty sentinel 2.
        All statistics use complete radius sets, even when top-K truncates rows.
        """
        coordinates = np.asarray(coordinates, dtype=np.float64)
        if coordinates.ndim != 2 or coordinates.shape[1] != 12 or not np.all(np.isfinite(coordinates)):
            raise ValueError('query coordinates must be a finite N x 12 array')
        phases = list(phases)
        if len(phases) != len(coordinates):
            raise ValueError('one phase is required per query')
        phases = np.asarray([_phase(p) for p in phases], dtype=np.int8)
        k = self.config['neighbors']
        output = np.zeros((len(coordinates), k * 17 + 8), dtype=np.float32)
        neighbors = output[:, :k * 17].reshape(-1, k, 17)
        stats = output[:, k * 17:]
        stats[:, -2:] = 2
        normalized = coordinates / self._widths
        for phase, index in enumerate(self._indices):
            lanes = np.flatnonzero(phases == phase)
            if index is None or not len(lanes):
                continue
            # One batched CPU tree lookup per phase. Never rebuild at query time.
            candidates = index.tree.query_ball_point(normalized[lanes], r=2., p=np.inf)
            for lane, ids in zip(lanes, candidates):
                if not len(ids):
                    continue
                ids = np.asarray(ids, dtype=np.intp)
                relative = index.coordinates[ids] - normalized[lane]
                distance = np.max(np.abs(relative), axis=1)
                near = distance <= 1.
                for mask, count_column, fraction_column, nearest_column in (
                    (near, 0, 2, 6), (distance <= 2., 1, 4, 7)):
                    count = int(mask.sum())
                    stats[lane, count_column] = np.log1p(count)
                    if count:
                        stats[lane, fraction_column:fraction_column + 2] = index.flags[ids[mask], :2].mean(axis=0)
                        stats[lane, nearest_column] = min(2., float(distance[mask].min()))
                positions = np.flatnonzero(near)
                positions = positions[np.lexsort((ids[positions], distance[positions]))[:k]]
                n = len(positions)
                neighbors[lane, :n, :12] = relative[positions]
                neighbors[lane, :n, 12:16] = index.flags[ids[positions]]
                neighbors[lane, :n, 16] = 1
        return output
