import hashlib
import os
from pathlib import Path
import numpy as np
import pytest
from jit_dvgc.generative_bridge import verified_trace_cache as cache_module


def saved(tmp_path, name='trace.npz', value=0):
    path = tmp_path / name
    arrays = dict(mask=np.ones((5, 17), bool), values=np.arange(5*17*4).reshape(5, 17, 4) + value)
    np.savez_compressed(path, **arrays)
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), arrays


def test_seventeen_lanes_read_hash_decompress_once_and_are_immutable(tmp_path):
    path, sha, expected = saved(tmp_path)
    cache = cache_module.VerifiedTraceCache(max_bytes=100000)
    for lane in range(17):
        arrays = cache.lane(path, sha, lane)
        for name, values in expected.items():
            np.testing.assert_array_equal(arrays[name], values[:, lane])
            with pytest.raises(ValueError):
                arrays[name].setflags(write=True)
    assert cache.stats['reads'] == cache.stats['decompressions'] == 1
    assert cache.stats['hash_bytes'] == path.stat().st_size
    assert cache.stats['hits'] == 16


def test_changed_file_with_restored_mtime_is_rejected(tmp_path):
    path, sha, _ = saved(tmp_path)
    cache = cache_module.VerifiedTraceCache()
    cache.load(path, sha)
    before = path.stat()
    with path.open('r+b') as stream:
        stream.seek(40)
        original = stream.read(1)
        stream.seek(40)
        stream.write(bytes([original[0] ^ 1]))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match='hash'):
        cache.load(path, sha)


def test_lru_eviction_and_schema_separation(tmp_path):
    path, sha, expected = saved(tmp_path)
    other, other_sha, _ = saved(tmp_path, 'other.npz', value=1)
    nbytes = sum(a.nbytes for a in expected.values())
    cache = cache_module.VerifiedTraceCache(max_bytes=nbytes, max_entries=2)
    cache.load(path, sha, schema='v1')
    cache.load(other, other_sha, schema='v1')
    cache.load(path, sha, schema='v1')
    assert cache.stats['reads'] == 3 and cache.stats['evictions'] == 2
    assert cache.resident_bytes <= nbytes
    cache.load(path, sha, schema='v2')
    assert cache.stats['decompressions'] == 4


def test_lane_rejects_noncontiguous_mask_and_allows_explicit_empty(tmp_path):
    path = tmp_path / 'trace.npz'
    np.savez_compressed(path, mask=np.array([[1, 0], [0, 0], [1, 0]], bool))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    cache = cache_module.VerifiedTraceCache()
    with pytest.raises(ValueError, match='noncontiguous'):
        cache.lane(path, sha, 0)
    with pytest.raises(ValueError, match='missing'):
        cache.lane(path, sha, 1)
    assert cache.lane(path, sha, 1, require_nonempty=False)['mask'].shape == (0,)


def test_feedback_evaluation_shares_arrays_but_never_caches_labels(tmp_path, monkeypatch):
    from jit_dvgc.generative_bridge.feedback_data import trace_from_evaluation
    path = tmp_path / 'evaluation.npz'
    observations = np.arange(5*17*76).reshape(5, 17, 76)
    np.savez_compressed(path, mask=np.ones((5, 17), bool),
                        actor_observation_before=observations,
                        physical_failure=np.zeros((5, 17), bool))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    cache = cache_module.VerifiedTraceCache()
    monkeypatch.setattr(cache_module, 'TRACE_CACHE', cache)
    metadata = dict(success_criterion='stable_forward_recovery', model_sha256='model',
                    actor_sha256='actor', normalizer_sha256='norm', root_context_sha256='context')
    attempt = dict(recording_schema='jit_actor_success_trace_v1_1', action_origin='actor_only',
                   outcome='stable_forward_recovery', model_sha256='model', actor_sha256='actor',
                   normalizer_sha256='norm', snapshot_context_sha256='context', trace=str(path),
                   trace_sha256=sha, label=1)
    for lane in range(17):
        result = trace_from_evaluation({**attempt, 'trace_lane': lane}, metadata)
        np.testing.assert_array_equal(result['arrays']['actor_observation_before'], observations[:, lane])
        assert result['metadata']['full_success'] is True
    result = trace_from_evaluation({**attempt, 'trace_lane': 0, 'label': 0}, metadata)
    assert result['metadata']['full_success'] is False
    assert cache.stats['reads'] == cache.stats['decompressions'] == 1


def test_unavailable_file_notifications_conservatively_rehash(tmp_path, monkeypatch):
    path, sha, _ = saved(tmp_path)
    cache = cache_module.VerifiedTraceCache()
    monkeypatch.setattr(cache._changes, 'watch', lambda path: False)
    cache.load(path, sha)
    cache.load(path, sha)
    assert cache.stats['reads'] == 2
    assert cache.stats['decompressions'] == 1


def test_replaced_inode_with_restored_mtime_is_rejected(tmp_path):
    path, sha, _ = saved(tmp_path)
    cache = cache_module.VerifiedTraceCache()
    cache.load(path, sha)
    before = path.stat()
    replacement, _, _ = saved(tmp_path, 'replacement.npz', value=2)
    os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
    os.replace(replacement, path)
    with pytest.raises(ValueError, match='hash'):
        cache.load(path, sha)
