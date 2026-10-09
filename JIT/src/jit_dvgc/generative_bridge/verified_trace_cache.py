"""Bounded cache of SHA-verified NPZ bytes; no admission or outcome decisions.

Content and declared schema identify arrays. File signatures only permit reuse
of an already verified immutable snapshot; a changed signature requires hashing
again, including changes which restore mtime. Arrays have immutable bytes owners.
"""
from collections import OrderedDict
import ctypes
import hashlib
import io
import os
import struct
from pathlib import Path
from threading import RLock
from types import MappingProxyType
import numpy as np

TRACE_ARRAY_SCHEMA = 'jit_trace_arrays_npz_v1'


def _signature(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


class _FileChanges:
    """Linux write notifications cover same-clock-tick changes hidden by stat."""
    def __init__(self):
        self.fd = -1
        self.paths = {}
        self.lib = ctypes.CDLL(None, use_errno=True)
        if hasattr(self.lib, 'inotify_init1'):
            self.fd = self.lib.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)

    def watch(self, path):
        if self.fd < 0:
            return False
        # IN_MODIFY, IN_ATTRIB, IN_CLOSE_WRITE, IN_DELETE_SELF, IN_MOVE_SELF.
        wd = self.lib.inotify_add_watch(self.fd, os.fsencode(path), 0x2 | 0x4 | 0x8 | 0x400 | 0x800)
        if wd < 0:
            return False
        for old_wd, old_path in list(self.paths.items()):
            if old_path == str(path) and old_wd != wd:
                self.lib.inotify_rm_watch(self.fd, old_wd)
                self.paths.pop(old_wd, None)
        self.paths[wd] = str(path)
        return True

    def poll(self):
        changed = set()
        if self.fd < 0:
            return None
        while True:
            try:
                payload = os.read(self.fd, 65536)
            except BlockingIOError:
                break
            if not payload:
                break
            offset = 0
            while offset < len(payload):
                wd, mask, _, length = struct.unpack_from('iIII', payload, offset)
                offset += 16 + length
                if mask & 0x4000:  # IN_Q_OVERFLOW: no cached signature is trustworthy.
                    return None
                if wd in self.paths:
                    changed.add(self.paths[wd])
                    if mask & 0x8000:  # IN_IGNORED
                        self.paths.pop(wd, None)
        return changed

    def retain(self, paths):
        for wd, path in list(self.paths.items()):
            if path not in paths:
                self.lib.inotify_rm_watch(self.fd, wd)
                self.paths.pop(wd, None)

    def __del__(self):
        if self.fd >= 0:
            os.close(self.fd)


class VerifiedTraceCache:
    def __init__(self, max_bytes=256 * 1024**2, max_entries=128):
        if type(max_bytes) is not int or max_bytes <= 0 or type(max_entries) is not int or max_entries <= 0:
            raise ValueError('positive cache byte and entry bounds required')
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.resident_bytes = 0
        self.stats = dict(reads=0, decompressions=0, hash_bytes=0, hits=0, evictions=0)
        self._entries = OrderedDict()
        self._verified_paths = OrderedDict()
        self._lock = RLock()
        self._changes = _FileChanges()

    def load(self, path, expected_sha256, *, schema=TRACE_ARRAY_SCHEMA):
        """Return immutable arrays, verifying the exact bytes that are decompressed."""
        path = Path(path).absolute()
        if not isinstance(schema, str) or not schema:
            raise ValueError('nonempty array schema required')
        key = (expected_sha256, schema)
        path_key = (str(path), key)
        with self._lock:
            self._changes.retain({k[0] for k in self._verified_paths} | {str(path)})
            watched = self._changes.watch(path)
            changed = self._changes.poll()
            if changed is None:
                self._verified_paths.clear()
            elif changed:
                self._verified_paths = OrderedDict(
                    (k, v) for k, v in self._verified_paths.items() if k[0] not in changed)
            signature = _signature(path.stat())
            if watched and self._verified_paths.get(path_key) == signature and key in self._entries:
                self.stats['hits'] += 1
                self._entries.move_to_end(key)
                self._verified_paths.move_to_end(path_key)
                return self._entries[key][0]
            with path.open('rb') as stream:
                before = _signature(os.fstat(stream.fileno()))
                payload = stream.read()
                after = _signature(os.fstat(stream.fileno()))
            self.stats['reads'] += 1
            self.stats['hash_bytes'] += len(payload)
            if before != after or before != signature or after != _signature(path.stat()):
                raise ValueError('trace file changed during verified read')
            if hashlib.sha256(payload).hexdigest() != expected_sha256:
                raise ValueError('trace payload hash changed')
            if key in self._entries:
                arrays = self._entries[key][0]
                self._entries.move_to_end(key)
            else:
                with np.load(io.BytesIO(payload), allow_pickle=False) as raw:
                    arrays = {}
                    for name in raw.files:
                        value = raw[name]
                        arrays[name] = np.frombuffer(value.tobytes(order='C'), dtype=value.dtype).reshape(value.shape)
                arrays = MappingProxyType(arrays)
                self.stats['decompressions'] += 1
                size = sum(a.nbytes for a in arrays.values())
                if size > self.max_bytes:
                    self._changes.retain({k[0] for k in self._verified_paths})
                    return arrays
                while self._entries and (self.resident_bytes + size > self.max_bytes
                                         or len(self._entries) >= self.max_entries):
                    old_key, (_, old_size) = self._entries.popitem(last=False)
                    self.resident_bytes -= old_size
                    self.stats['evictions'] += 1
                    self._verified_paths = OrderedDict(
                        (k, v) for k, v in self._verified_paths.items() if k[1] != old_key)
                self._entries[key] = (arrays, size)
                self.resident_bytes += size
            if watched:
                self._verified_paths[path_key] = signature
                self._verified_paths.move_to_end(path_key)
            else:
                self._verified_paths.pop(path_key, None)
            while len(self._verified_paths) > self.max_entries * 4:
                self._verified_paths.popitem(last=False)
            self._changes.retain({k[0] for k in self._verified_paths})
            return arrays

    def lane(self, path, expected_sha256, lane, *, schema=TRACE_ARRAY_SCHEMA,
             mask_key='mask', require_nonempty=True):
        """Slice a real contiguous lane prefix without copying full batch arrays."""
        arrays = self.load(path, expected_sha256, schema=schema)
        mask = arrays[mask_key]
        if (mask.ndim != 2 or type(lane) not in (int, np.int32, np.int64)
                or lane < 0 or lane >= mask.shape[1]):
            raise ValueError('invalid trace lane')
        ids = np.flatnonzero(mask[:, lane])
        if require_nonempty and not len(ids):
            raise ValueError('missing real lane')
        if not np.array_equal(ids, np.arange(len(ids))):
            raise ValueError('noncontiguous real lane')
        return {name: array[:len(ids), lane] for name, array in arrays.items()}


TRACE_CACHE = VerifiedTraceCache()
