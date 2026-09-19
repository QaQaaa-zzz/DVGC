import json
from pathlib import Path
import pytest
from jit_dvgc.pulse_evaluation_batches import evaluation_shards, evaluate_batched


def test_shards_bound_memory_and_preserve_original_nonterminal_rng_lanes():
    rows=[dict(index=i,snapshot_context_sha256=str(i),prefix_terminal=i==2) for i in range(7)]
    shards=evaluation_shards(rows,3)
    assert [len(x[0]) for x in shards]==[3,3,1]
    assert [x[1] for x in shards]==[[0,1],[2,3,4],[5]]
    assert [x[2] for x in shards]==[6,6,6]
    assert [r['index'] for s,_,_ in shards for r in s]==list(range(7))
    for n in [0,-1,True,2.5]:
        with pytest.raises(ValueError):evaluation_shards(rows,n)


def setup_case(tmp_path,monkeypatch,*,fail=False,drift=False):
    import jit_dvgc.pulse_evaluation_batches as mod
    rows=[dict(index=i,snapshot_context_sha256=str(i),prefix_terminal=False) for i in range(5)]
    path=tmp_path/'candidates.json';path.write_text(json.dumps(rows))
    spec=dict(candidates=str(path),evaluation_batch_size=2,order=['pi'],horizon=400,
        budget=2000,python='/usr/bin/python3',repo=str(tmp_path),gate={'kind':'test'},
        source_locks={},input_files={},wait_timeout_seconds=1,stage_timeout_seconds=1)
    calls=[]
    def execute(plan_path,output,**kw):
        plan=json.loads(plan_path.read_text());stage=plan['stages'][0];args=stage['argv'];s=json.loads(Path(args[args.index('--spec')+1]).read_text());dest=Path(args[args.index('--output')+1]);dest.mkdir()
        calls.append(s)
        if fail:return {'phase':'failed'}
        rows=json.loads(Path(s['candidates']).read_text())
        if drift:rows[0]['snapshot_context_sha256']='changed'
        for r in rows:r.update(label=1,witness='pi',attempts=[])
        (dest/'results.json').write_text(json.dumps(rows));(dest/'status.json').write_text(json.dumps(dict(phase='completed',charged_interactions=len(rows)*10,active_interactions=len(rows)*8,padding_interactions=len(rows)*2,peak_rss_kib=1234)))
        return {'phase':'completed'}
    monkeypatch.setattr(mod,'run_gated_plan',execute)
    return spec,calls


def test_merge_exact_order_cost_and_no_recursive_sharding(tmp_path,monkeypatch):
    spec,calls=setup_case(tmp_path,monkeypatch);out=tmp_path/'result';evaluate_batched(spec,out)
    assert len(calls)==3 and all('evaluation_batch_size' not in s for s in calls)
    assert [s['suffix_rng_indices'] for s in calls]==[[0,1],[2,3],[4]]
    assert [r['index'] for r in json.loads((out/'results.json').read_text())]==list(range(5))
    s=json.loads((out/'status.json').read_text());assert s['charged_interactions']==50 and s['padding_interactions']==10 and s['peak_child_rss_kib']==1234


@pytest.mark.parametrize('kind',['fail','drift'])
def test_child_failure_or_changed_context_never_publishes_results(tmp_path,monkeypatch,kind):
    spec,calls=setup_case(tmp_path,monkeypatch,**{kind:True});out=tmp_path/'result'
    with pytest.raises((ValueError,RuntimeError)):evaluate_batched(spec,out)
    assert len(calls)==1 and not (out/'results.json').exists()
    assert json.loads((out/'status.json').read_text())['phase']=='error'


def test_failed_shard_keeps_reservation_separate_from_measured_completed_cost(tmp_path,monkeypatch):
    import jit_dvgc.pulse_evaluation_batches as mod
    spec,_=setup_case(tmp_path,monkeypatch)
    monkeypatch.setattr(mod,'run_gated_plan',lambda *a,**kw:dict(phase='failed',reserved_interactions=800))
    out=tmp_path/'result'
    with pytest.raises(RuntimeError):evaluate_batched(spec,out)
    s=json.loads((out/'status.json').read_text())
    assert s['charged_interactions']==0
    assert s['incomplete_reserved_interactions']==800
    assert s['accounting']=='completed_shards_only_with_incomplete_reservation'
    assert s['active_shard']==0


def test_resume_completed_shards_requires_locks_and_charges_only_new_work(tmp_path,monkeypatch):
    from jit_dvgc.probe_bank import _file_sha
    spec,calls=setup_case(tmp_path,monkeypatch)
    old=tmp_path/'old';evaluate_batched(spec,old)
    # Only the first shard is complete in this interrupted run.
    for i in (1,2):
        (old/f'shard_{i:04d}/evaluation/status.json').unlink()
    spec['resume_evaluation_root']=str(old)
    paths=[old/'shard_0000'/p for p in ('spec.json','evaluation/status.json','evaluation/results.json')]
    spec['input_files']={str(p):_file_sha(p) for p in paths}
    calls.clear();out=tmp_path/'resumed';evaluate_batched(spec,out)
    s=json.loads((out/'status.json').read_text())
    assert len(calls)==2
    assert s['charged_interactions']==30 and s['reused_interactions']==20
    assert len(json.loads((out/'results.json').read_text()))==5
    spec['resume_evaluation_root']=str(out)
    for i in range(3):
        for filename in ('spec.json','evaluation/status.json','evaluation/results.json'):
            p=(out/f'shard_{i:04d}'/filename).resolve()
            spec['input_files'][str(p)]=_file_sha(p)
    calls.clear();evaluate_batched(spec,tmp_path/'twice')
    assert not calls
    assert json.loads((tmp_path/'twice/status.json').read_text())['charged_interactions']==0
    paths[-1].write_text('[]')
    with pytest.raises(ValueError,match='lock'):
        evaluate_batched(spec,tmp_path/'tampered')
