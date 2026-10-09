"""Lossless per-update provenance with bounded arrays and interned metadata.

The SHA-linked manifest is published only after the numeric artifact is fsynced.
Legacy list-of-dictionaries metrics remain readable through the same accessor.
"""
import json
import os
from pathlib import Path
import numpy as np

from .contracts import file_sha
from .protocol import atomic_json

_NUMERIC={'window_start','window_end_exclusive','window_start_step','window_end_step_exclusive'}


def _sync_directory(path):
    fd=os.open(path,os.O_RDONLY | os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


class CompactProvenance:
    def __init__(self,*,updates,batch_size):
        self.capacity=updates;self.batch_size=batch_size;self.count=0
        self.fields=None;self.arrays={};self.dictionaries={};self.lookups={}
        self.updates=np.empty(updates,dtype=np.int64)
        self.state_updates=np.empty(updates,dtype=np.int64)
        self.seeds=np.empty(updates,dtype=np.uint32)

    def append(self,rows,*,update,state_updates,sampling_seed):
        if len(rows)!=self.batch_size or self.count>=self.capacity:
            raise ValueError('provenance dimensions differ from declared budget')
        fields=tuple(sorted(rows[0]))
        if self.fields is None:
            self.fields=fields
            for field in fields:
                self.arrays[field]=np.empty((self.capacity,self.batch_size),
                    dtype=np.int64 if field in _NUMERIC else np.uint32)
                if field in _NUMERIC:
                    self.arrays[field+'__present']=np.empty((self.capacity,self.batch_size),dtype=np.bool_)
                else:self.dictionaries[field]=[];self.lookups[field]={}
        if fields!=self.fields or any(tuple(sorted(row))!=fields for row in rows):
            raise ValueError('provenance schema changed mid-stage')
        for column,row in enumerate(rows):
            for field,value in row.items():
                if field in _NUMERIC:
                    if value is not None and (type(value) is not int or not np.iinfo(np.int64).min<=value<=np.iinfo(np.int64).max):
                        raise ValueError('invalid provenance integer')
                    self.arrays[field][self.count,column]=0 if value is None else value
                    self.arrays[field+'__present'][self.count,column]=value is not None
                else:
                    key=json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':'))
                    table=self.lookups[field]
                    if key not in table:
                        table[key]=len(table);self.dictionaries[field].append(json.loads(key))
                    self.arrays[field][self.count,column]=table[key]
        self.updates[self.count]=update;self.state_updates[self.count]=state_updates
        self.seeds[self.count]=sampling_seed;self.count+=1

    def save(self,root):
        root=Path(root);root.mkdir(parents=True,exist_ok=True)
        arrays=root/'sample_provenance.npz'
        # One numeric allocation per field; no update x batch nested Python rows.
        with arrays.open('xb') as stream:
            np.savez_compressed(stream,**{k:v[:self.count] for k,v in self.arrays.items()},
                update=self.updates[:self.count],state_updates=self.state_updates[:self.count],
                sampling_seed=self.seeds[:self.count])
            stream.flush();os.fsync(stream.fileno())
        manifest={'schema':'jit_compact_sample_provenance_v1','arrays':arrays.name,
            'arrays_sha256':file_sha(arrays),'updates':self.count,'batch_size':self.batch_size,
            'fields':list(self.fields or ()),'numeric_fields':sorted(_NUMERIC & set(self.fields or ())),
            'dictionaries':self.dictionaries}
        path=root/'sample_provenance.json'
        atomic_json(path,manifest);_sync_directory(root)
        return {'schema':manifest['schema'],'manifest':path.name,'manifest_sha256':file_sha(path),
                'arrays':arrays.name,'arrays_sha256':manifest['arrays_sha256'],
                'updates':self.count,'batch_size':self.batch_size}


def load_sample_provenance(root,pointer,update_index):
    """Reconstruct one complete legacy-shaped batch; verify both linked artifacts."""
    if isinstance(pointer,list):return pointer
    root=Path(root);path=root/pointer['manifest']
    if file_sha(path)!=pointer['manifest_sha256']:raise ValueError('provenance manifest hash mismatch')
    manifest=json.loads(path.read_text());arrays=root/manifest['arrays']
    if (manifest['schema']!='jit_compact_sample_provenance_v1'
        or file_sha(arrays)!=manifest['arrays_sha256']):
        raise ValueError('provenance arrays schema/hash mismatch')
    if not 0<=update_index<manifest['updates']:raise IndexError(update_index)
    result=[]
    with np.load(arrays,allow_pickle=False) as archive:
        values={key:archive[key][update_index] for key in archive.files
                if key not in ('update','state_updates','sampling_seed')}
        for i in range(manifest['batch_size']):
            row={}
            for field in manifest['fields']:
                code=int(values[field][i])
                row[field]=(code if values[field+'__present'][i] else None) if field in manifest['numeric_fields'] else manifest['dictionaries'][field][code]
            result.append(row)
    return result
