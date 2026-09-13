"""Delayed witnesses enter future reset support without relabeling pending rows."""
from copy import deepcopy
import json

import pytest

from .test_exploration_pool import source, captured, result, write
from jit_dvgc import exploration_pool as pool_module
from jit_dvgc import unified_envelope_snapshot as snapshots
from jit_dvgc.exploration_loop_support import append_witnessed_support
from jit_dvgc.iterative_probe_training import SUPPORT_SCHEMA


@pytest.fixture
def support_source(source):
    root,_=source
    pool=pool_module.import_candidates(root)
    template=next(iter(pool['entries'].values()))
    original=snapshots.load_unified_envelope_snapshot(template['snapshot_dir'])
    entries=[];inputs={}
    for phase,value in [('upstream',0),('downstream',1)]:
        snap=deepcopy(original);snap.active_phase=value;snap.rng[1]=100+value
        directory=root/'inherited'/phase
        snapshots.save_unified_envelope_snapshot(directory,snap)
        entries.append(dict(key=phase,phase=phase,snapshot=str(directory),state_sha256=snapshots.physical_state_sha256(snap),
            snapshot_context_sha256=snapshots.snapshot_context_sha256(snap),trajectory_id='old-'+phase,
            witnessed=True,sampling_weight=.7 if value else .3))
        for filename in ('identity.json','snapshot.pkl'):inputs[str(directory/filename)]=pool_module._sha(directory/filename)
    base=dict(schema=SUPPORT_SCHEMA,role='train',final_test_used=False,inputs=inputs,entries=entries)
    base['support_sha256']=pool_module.canonical_sha256(base)
    return root,pool,base


def test_only_later_positive_is_appended_with_original_weights_and_context(support_source):
    root,pool,base=support_source
    key,entry=next(iter(pool['entries'].items()))
    negative=pool_module.add_evaluation(pool,key,result(entry,0),round_index=0)
    path=root/'negative.json';write(path,negative)
    view=append_witnessed_support(base,negative,path)
    assert view['entries']==base['entries'] and view['appended_witnessed_count']==0
    positive=pool_module.add_evaluation(negative,key,result(entry,1,'b'),round_index=1)
    path=root/'positive.json';write(path,positive)
    inherited=deepcopy(base)
    view=append_witnessed_support(base,positive,path)
    assert base==inherited and view['entries'][:2]==base['entries']
    added=view['entries'][2]
    assert added['key']==key and added['snapshot_context_sha256']==entry['context_sha256']
    assert added['first_witness_round']==1 and added['witness_bank_sha256']=='b'*64
    assert added['sampling_weight']==.7 and added['witnessed'] is True
    assert len(pool_module.select_pending(positive,10))==1
    assert append_witnessed_support(base,positive,path)==view


def test_context_acquisition_duplicates_are_not_duplicate_reset_mass(support_source):
    root,pool,base=support_source
    key,entry=next(iter(pool['entries'].items()))
    pool=pool_module.add_evaluation(pool,key,result(entry,1),round_index=0)
    duplicate=deepcopy(pool['entries'][key]);duplicate['acquisition_protocol_sha256']='b'*64
    duplicate['key']=pool_module.candidate_key(duplicate['state_sha256'],duplicate['context_sha256'],'b'*64)
    pool['entries'][duplicate['key']]=duplicate;pool_module._seal(pool)
    path=root/'pool.json';write(path,pool)
    support=append_witnessed_support(base,pool,path)
    assert support['appended_witnessed_count']==1
    assert len(support['entries'])==3 and len(pool['entries'])==3


@pytest.mark.parametrize('tamper',['pool_file','snapshot','positive','phase'])
def test_tampered_support_input_or_positive_cannot_be_admitted(support_source,tamper):
    root,pool,base=support_source
    key,entry=next(iter(pool['entries'].items()))
    pool=pool_module.add_evaluation(pool,key,result(entry,1),round_index=0)
    path=root/'pool.json';write(path,pool)
    if tamper=='pool_file':write(path,{})
    elif tamper=='snapshot':
        from pathlib import Path
        (Path(entry['snapshot_dir'])/'snapshot.pkl').write_bytes(b'bad')
    elif tamper=='positive':
        pool['entries'][key]['observations'][0]['result']['state_sha256']='0'*64
        pool_module._seal(pool);write(path,pool)
    else:
        pool['entries'][key]['phase']='upstream';pool_module._seal(pool);write(path,pool)
    with pytest.raises(ValueError):append_witnessed_support(base,pool,path)
