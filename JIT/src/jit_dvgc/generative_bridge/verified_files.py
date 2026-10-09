"""Bounded process-local SHA verification cache with kernel mutation invalidation."""
from collections import OrderedDict
import hashlib
import os
from pathlib import Path
from threading import RLock
from .verified_trace_cache import _FileChanges, _signature
from .performance import measure


class VerifiedFileCache:
    def __init__(self,max_entries=32768):
        self.max_entries=max_entries
        self.entries=OrderedDict();self.changes=_FileChanges();self.lock=RLock()
        self.stats=dict(hash_bytes=0,reads=0,hits=0)

    def _watch(self,path):
        return self.changes.watch(path)

    def verify(self,path,expected):
        path=str(Path(path).absolute())
        with self.lock:
            watched=self._watch(path)
            changed=self.changes.poll()
            if changed is None:self.entries.clear()
            else:
                for item in changed:self.entries.pop(item,None)
            signature=_signature(os.stat(path))
            if watched and self.entries.get(path)==(signature,expected):
                self.entries.move_to_end(path);self.stats['hits']+=1
                return expected
            with measure('hash_verify',input_bytes=signature[2],hash_bytes=signature[2],cache_hit=False):
                with open(path,'rb') as stream:
                    before=_signature(os.fstat(stream.fileno()));h=hashlib.sha256()
                    for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
                    after=_signature(os.fstat(stream.fileno()))
                if before!=signature or before!=after or after!=_signature(os.stat(path)):
                    raise ValueError('locked input changed during read: '+path)
                digest=h.hexdigest();self.stats['reads']+=1;self.stats['hash_bytes']+=signature[2]
                if digest!=expected:raise ValueError('locked input hash mismatch: '+path)
            if watched:self.entries[path]=(signature,expected)
            while len(self.entries)>self.max_entries:
                old,_=self.entries.popitem(last=False)
                self.changes.forget(old)
            return expected


LOCK_CACHE=VerifiedFileCache()
