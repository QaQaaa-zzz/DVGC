"""Operation-local SHA256 reuse; no persistent trust or skipped hash comparisons."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import hashlib
import time
from pathlib import Path

_cache = ContextVar('evidence_hash_cache', default=None)


@contextmanager
def hash_cache_scope():
    if _cache.get() is not None:
        yield
        return
    token = _cache.set({})
    try:
        yield
    finally:
        _cache.reset(token)


def scoped_hashes(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with hash_cache_scope():
            return function(*args, **kwargs)
    return wrapped


def file_sha256(path):
    path = Path(path)
    cache = _cache.get()
    before = path.stat()
    identity = (before.st_dev, before.st_ino, before.st_size,
                before.st_mtime_ns, before.st_ctime_ns)
    key = (str(path.absolute()), identity)
    # Avoid timestamp-resolution collisions for files being created/rewritten.
    settled = time.time_ns() - before.st_ctime_ns > 1_000_000_000
    if cache is not None and settled and key in cache:
        return cache[key]
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    after = path.stat()
    if identity != (after.st_dev, after.st_ino, after.st_size,
                    after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError(f'evidence changed while hashing: {path}')
    result = digest.hexdigest()
    if cache is not None:
        cache[key] = result
    return result
