"""Budget/selection/history integration with offline suffix fixtures only."""
import json
from pathlib import Path

import pytest

from .test_exploration_pool import source, captured, result, write
from jit_dvgc import exploration_pool as pool_module
from jit_dvgc import exploration_reevaluation as m
from jit_dvgc import exploration_continuation
from jit_dvgc import unified_envelope_snapshot as snapshots


@pytest.fixture
def setup(source, monkeypatch):
    root,_=source
    pool=pool_module.import_candidates(root)
    pool_path=root/'pool.json';write(pool_path,pool)
    bank=dict(bank_sha256='a'*64,version='fixture',max_ticks=4,members=[
        dict(name=n,roles=['proposer','evaluator'],policy=dict(actor_sha256='c'*64,payload_sha256='d'*64)) for n in ['first','second']])
    bank_path=root/'bank.json';write(bank_path,bank)
    monkeypatch.setattr(m,'load_probe_bank',lambda path: json.loads(Path(path).read_text()))
    calls=[]
    class Evaluator:
        def __init__(self,path,order,horizon,output,budget):
            self.bank=json.loads(Path(path).read_text());self.output=Path(output)
            self.output.mkdir();self.charged_interactions=0;self.order=order
        def evaluate(self,snapshot):
            context=snapshots.snapshot_context_sha256(snapshot)
            calls.append(context);self.charged_interactions+=3
            directory=self.output/context/'snapshot'
            snapshots.save_unified_envelope_snapshot(directory,snapshot)
            entry=dict(state_sha256=snapshots.physical_state_sha256(snapshot),context_sha256=context,snapshot_dir=str(directory))
            r=result(entry,1)
            r['bank_sha256']=self.bank['bank_sha256']
            r['witness']=self.order[0]
            r['labels']={name: (1 if name==self.order[0] else None) for name in self.order}
            r['attempts'][0].update(evaluator=self.order[0],actor_sha256='c'*64,payload_sha256='d'*64)
            r['attempts'][0].pop('receipt_sha256');m._seal(r['attempts'][0])
            r.pop('receipt_sha256');r=m._seal(r)
            write(directory.parent/'result.json',r)
            return r
    monkeypatch.setattr(exploration_continuation,'FrozenSuffixEvaluator',Evaluator)
    return root,pool_path,bank_path,pool,calls


def test_bounded_evaluation_preserves_old_pool_and_incremental_evidence(setup):
    root,pool_path,bank_path,pool,calls=setup
    original=pool_path.read_bytes()
    report=m.run(pool_path,bank_path,order=['first','second'],horizon=4,budget=4,output=root/'reeval',round_index=1)
    assert report['status']=='budget_exhausted' and report['charged_interactions']==3
    assert len(calls)==1 and len(report['unevaluated_keys'])==1
    assert pool_path.read_bytes()==original
    final=json.loads((root/'reeval/pool.json').read_text())
    assert len(pool_module.select_pending(final,5))==1
    assert (root/'reeval/step_000000/pool.json').exists()
    assert not (root/'reeval/step_000001').exists()
    with pytest.raises(FileExistsError):
        m.run(pool_path,bank_path,order=['first','second'],horizon=4,budget=4,output=root/'reeval',round_index=1)


def test_subset_bank_is_explicit_and_requested_key_can_be_revisited(setup):
    root,pool_path,bank_path,pool,calls=setup
    key=next(iter(pool['entries']))
    report=m.run(pool_path,bank_path,order=['second'],horizon=4,budget=4,output=root/'subset',round_index=1,keys=[key])
    subset=json.loads((root/'subset/evaluation_bank.json').read_text())
    assert subset['parent_bank_sha256']=='a'*64
    assert [x['name'] for x in subset['members'] if 'evaluator' in x['roles']]==['second']
    assert report['status']=='selection_complete'
    new_path=root/'subset/pool.json'
    report=m.run(new_path,bank_path,order=['second'],horizon=4,budget=4,output=root/'again',round_index=2,keys=[key])
    final=json.loads((root/'again/pool.json').read_text())
    assert [x['round_index'] for x in final['entries'][key]['observations']]==[1,2]


def test_zero_budget_no_attempt_and_empty_selection(setup):
    root,pool_path,bank_path,pool,calls=setup
    report=m.run(pool_path,bank_path,order=['first'],horizon=4,budget=0,output=root/'zero',round_index=0)
    assert report['status']=='budget_exhausted' and report['charged_interactions']==0 and not calls
    report=m.run(pool_path,bank_path,order=['first'],horizon=4,budget=0,output=root/'empty',round_index=0,limit=0)
    assert report['status']=='selection_complete' and report['selected_count']==0


@pytest.mark.parametrize('override',[dict(order=['absent']),dict(keys=['missing']),dict(horizon=5),dict(round_index=-1)])
def test_invalid_predeclaration_never_constructs_output_or_runs(setup,override):
    root,pool_path,bank_path,pool,calls=setup
    args=dict(order=['first'],horizon=4,budget=4,output=root/'invalid',round_index=0)
    args.update(override)
    with pytest.raises(ValueError):m.run(pool_path,bank_path,**args)
    assert not calls and not (root/'invalid').exists()


def test_same_context_acquisition_keys_reuse_one_charged_suffix(setup):
    from copy import deepcopy
    root,pool_path,bank_path,pool,calls=setup
    entry=deepcopy(next(iter(pool['entries'].values())))
    original_key=entry['key']
    entry['acquisition_protocol_sha256']='b'*64
    entry['key']=pool_module.candidate_key(entry['state_sha256'],entry['context_sha256'],entry['acquisition_protocol_sha256'])
    pool['entries'][entry['key']]=entry
    pool_module._seal(pool);write(pool_path,pool)
    report=m.run(pool_path,bank_path,order=['first'],horizon=4,budget=4,output=root/'reuse',round_index=1,keys=[original_key,entry['key']])
    assert len(calls)==1 and report['charged_interactions']==3 and report['unique_contexts']==1
    assert len(report['completed_keys'])==2
    receipt=json.loads((root/'reuse/step_000001/receipt.json').read_text())
    assert receipt['reused_context'] and receipt['charged_interactions']==0


def test_dispatch_failure_preserves_cost_and_pool(setup,monkeypatch):
    root,pool_path,bank_path,pool,calls=setup
    def fail(self,snapshot):
        self.charged_interactions+=1
        raise RuntimeError('dispatch error')
    monkeypatch.setattr(exploration_continuation.FrozenSuffixEvaluator,'evaluate',fail)
    with pytest.raises(RuntimeError,match='dispatch error'):
        m.run(pool_path,bank_path,order=['first'],horizon=4,budget=4,output=root/'failure',round_index=1)
    failure=json.loads((root/'failure/failure.json').read_text())
    assert failure['charged_interactions']==1 and not failure['completed_keys']
    assert json.loads((root/'failure/pool.json').read_text())==pool
