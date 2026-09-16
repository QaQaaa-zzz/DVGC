import hashlib
import os
import time
from pathlib import Path

from jit_dvgc.config import file_sha256
from jit_dvgc.evidence_hash_cache import hash_cache_scope


def test_reuse_change_and_scope(tmp_path, monkeypatch):
    p = tmp_path / 'evidence'; p.write_bytes(b'one')
    time.sleep(1.05)
    original = Path.open
    reads = []
    def counted(self, *args, **kwargs):
        if self == p and args and args[0] == 'rb': reads.append(self)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', counted)
    with hash_cache_scope():
        assert file_sha256(p) == file_sha256(p) == hashlib.sha256(b'one').hexdigest()
        assert len(reads) == 1
        before = p.stat()
        p.write_bytes(b'two')
        os.utime(p, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert file_sha256(p) == hashlib.sha256(b'two').hexdigest()
        assert len(reads) == 2
    file_sha256(p)
    assert len(reads) == 3


def test_replacement_and_nested_scope(tmp_path):
    p = tmp_path / 'evidence'; p.write_bytes(b'old')
    with hash_cache_scope():
        old = file_sha256(p)
        with hash_cache_scope(): assert file_sha256(p) == old
        other = tmp_path / 'new'; other.write_bytes(b'new'); other.replace(p)
        assert file_sha256(p) != old
