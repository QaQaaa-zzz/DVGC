"""Explicit immutable sampling snapshot for one locked generator corpus.

No object-id or mutable-dict cache: callers rebuild this snapshot when the corpus
changes. Only sampling columns are retained; each buffer owns immutable bytes so
neither source mutation nor re-enabling NumPy write flags changes this index.
"""
from dataclasses import dataclass
import numpy as np
from .data import validate_trace, trajectory_identity


def _immutable_array(value, dtype=None):
    array = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class IndexedTrajectory:
    observations: np.ndarray
    actions: np.ndarray
    origins: np.ndarray
    starts: range
    root_id: str | None
    onset: int | None
    trace_start_step: int | None
    trajectory_sha256: str | None = None


@dataclass(frozen=True)
class CompiledCorpusIndex:
    names: tuple
    probabilities: tuple
    # Ordered groups -> sorted (ancestor, original-order trajectories).
    groups: tuple
    includes_trajectory_identity: bool = False


def compile_corpus_index(corpus, *, include_trajectory_identity=False):
    """Validate once and snapshot the corpus without altering its sampling law.

    This does not admit new data: the supplied corpus must already have passed
    the existing TRAIN/provenance admission protocol. Window starts match
    build_action_windows(horizon=16, max_per_phase=None) exactly.
    """
    from .feedback_data import MIX
    names = tuple(MIX)
    groups = []
    for group in names:
        by_ancestor = {}
        for trace in corpus['groups'][group]:
            n = validate_trace(trace)
            if n < 16:
                raise ValueError('corpus trajectory has no full H16 window')
            a, m = trace['arrays'], trace['metadata']
            row = IndexedTrajectory(
                _immutable_array(a['actor_observation_before']),
                _immutable_array(a['normalized_action_executed'], np.float32),
                _immutable_array(a['action_origin']), range(n - 16 + 1),
                m.get('root_id'), m.get('onset'), m.get('trace_start_step'),
                trajectory_identity(trace) if include_trajectory_identity else None)
            by_ancestor.setdefault(m['root_episode_id'], []).append(row)
        groups.append(tuple((ancestor, tuple(by_ancestor[ancestor]))
                            for ancestor in sorted(by_ancestor)))
    return CompiledCorpusIndex(names, tuple(corpus['realized_mix'][k] for k in names), tuple(groups),
                               include_trajectory_identity)
