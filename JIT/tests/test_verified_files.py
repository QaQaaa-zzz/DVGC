import hashlib,os
import pytest
from jit_dvgc.generative_bridge.verified_files import VerifiedFileCache


def test_repeated_verification_reads_once_and_rejects_tamper(tmp_path):
    p=tmp_path/'lock';p.write_bytes(b'abc');sha=hashlib.sha256(b'abc').hexdigest();c=VerifiedFileCache()
    for _ in range(4):c.verify(p,sha)
    assert c.stats['reads']==1 and c.stats['hits']==3
    old=p.stat();p.write_bytes(b'xyz');os.utime(p,ns=(old.st_atime_ns,old.st_mtime_ns))
    with pytest.raises(ValueError,match='hash mismatch'):c.verify(p,sha)


def test_replace_inode_and_eviction(tmp_path):
    c=VerifiedFileCache(max_entries=1)
    for name in ('a','b'):
        p=tmp_path/name;p.write_bytes(name.encode());c.verify(p,hashlib.sha256(name.encode()).hexdigest())
    assert len(c.entries)==1
    p=tmp_path/'a';replacement=tmp_path/'new';replacement.write_bytes(b'b');replacement.replace(p)
    with pytest.raises(ValueError):c.verify(p,hashlib.sha256(b'a').hexdigest())
