import json
import pytest
from jit_dvgc.generative_bridge.performance import interval_union,measure


def test_union_does_not_double_count_overlap():
    assert interval_union([(3,6),(1,4),(8,9)])==[[1.,6.],[8.,9.]]
    with pytest.raises(ValueError):interval_union([(4,3)])


def test_failed_event_keeps_exception_and_fields(tmp_path):
    path=tmp_path/'events.jsonl'
    with pytest.raises(RuntimeError):
        with measure('hash_verify',path=path,hash_bytes=19):raise RuntimeError('bad hash')
    row=json.loads(path.read_text())
    assert row['status']=='failed' and row['hash_bytes']==19
    assert row['elapsed_seconds']>=0 and row['gpu_peak_memory_bytes'] is None
