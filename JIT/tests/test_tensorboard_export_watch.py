import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest


def test_watch_opens_only_existing_sources_and_closes_completed(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('export_tb',Path(__file__).parents[1]/'cli/export_tensorboard.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source=tmp_path/'metrics.jsonl';source.write_text('{"step":1,"reward":2}\n')
    status=tmp_path/'status.json';status.write_text('{"status":"completed"}')
    manifest=tmp_path/'sources.json';manifest.write_text(json.dumps({'sources':[
        {'name':'current','path':str(source),'step':'step','completion_status':str(status)},
        {'name':'future','path':str(tmp_path/'future.jsonl'),'step':'step'}]}))
    opened=[];closed=[]
    class Writer:
        def __init__(self,path):self.path=path;opened.append(path)
        def add_text(self,*a):pass
        def add_scalar(self,*a):pass
        def flush(self):pass
        def close(self):closed.append(self.path)
    monkeypatch.setitem(sys.modules,'tensorboardX',SimpleNamespace(SummaryWriter=Writer))
    monkeypatch.setattr(sys,'argv',['export','--manifest',str(manifest),'--output',str(tmp_path/'output'),'--watch'])
    calls=[]
    def sleep(_):
        calls.append(1)
        if len(calls)==2:raise KeyboardInterrupt
    monkeypatch.setattr(module.time,'sleep',sleep)
    with pytest.raises(KeyboardInterrupt):module.main()
    assert len(opened)==1 and opened[0].endswith('/current')
    assert closed==opened


def load_exporter():
    spec = importlib.util.spec_from_file_location(
        'export_tb_grouped', Path(__file__).parents[1] / 'cli/export_tensorboard.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def invoke_exporter(module, tmp_path, monkeypatch, sources, watch=False):
    manifest = tmp_path / 'sources.json'
    manifest.write_text(json.dumps({'sources': sources}))
    argv = ['export', '--manifest', str(manifest), '--output', str(tmp_path / 'output')]
    monkeypatch.setattr(sys, 'argv', argv + (['--watch'] if watch else []))
    module.main()


def test_shared_run_keeps_later_completed_source_and_offsets_steps(tmp_path, monkeypatch):
    # Losing the shared writer or using local steps hides the second training round.
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    module = load_exporter()
    first = tmp_path / 'first.jsonl'
    first.write_text('{"step":10,"reward":2}\n')
    future = tmp_path / 'future.jsonl'
    status = tmp_path / 'status.json'
    status.write_text('{"status":"completed"}')
    sources = [dict(name=name, run='student', path=str(path), step='step',
                    step_offset=offset, completion_status=str(status))
               for name, path, offset in [('round1', first, 0), ('round2', future, 10)]]
    polls = []
    def sleep(_):
        polls.append(1)
        if len(polls) == 1:
            future.write_text('{"step":10,"reward":3}\n')
        else:
            raise KeyboardInterrupt
    monkeypatch.setattr(module.time, 'sleep', sleep)
    with pytest.raises(KeyboardInterrupt):
        invoke_exporter(module, tmp_path, monkeypatch, sources, watch=True)
    assert (tmp_path / 'output' / 'student').is_dir()
    events = EventAccumulator(str(tmp_path / 'output' / 'student')).Reload()
    assert [(event.step, event.value) for event in events.Scalars('reward')] == [(10, 2), (20, 3)]
    receipt = json.loads((tmp_path / 'output' / 'receipt.json').read_text())
    assert receipt['scalar_count'] == 2
    assert 'step_offset' in receipt['note']


@pytest.mark.parametrize('overrides', [
    [{'name': 'duplicate'}, {'name': 'duplicate'}],
    [{'step_offset': -1}], [{'step_offset': 1.5}], [{'step_offset': True}],
])
def test_invalid_source_identity_or_offset_rejected(tmp_path, monkeypatch, overrides):
    module = load_exporter()
    path = tmp_path / 'metrics.jsonl'
    path.write_text('{"step":1,"reward":2}\n')
    sources = [dict(dict(name=f'source{i}', path=str(path), step='step'), **override)
               for i, override in enumerate(overrides)]
    with pytest.raises(ValueError):
        invoke_exporter(module, tmp_path, monkeypatch, sources)


def test_shared_run_rejects_overlapping_scalar_steps(tmp_path, monkeypatch):
    module = load_exporter()
    path = tmp_path / 'metrics.jsonl'
    path.write_text('{"step":1,"reward":2}\n')
    sources = [dict(name=name, run='student', path=str(path), step='step')
               for name in ['first', 'second']]
    with pytest.raises(ValueError, match='overlap'):
        invoke_exporter(module, tmp_path, monkeypatch, sources)


def test_default_source_name_and_local_steps_preserved(tmp_path, monkeypatch):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    module = load_exporter()
    path = tmp_path / 'metrics.jsonl'
    path.write_text('{"step":7,"reward":2}\n')
    invoke_exporter(module, tmp_path, monkeypatch,
                    [dict(name='original', path=str(path), step='step')])
    events = EventAccumulator(str(tmp_path / 'output' / 'original')).Reload()
    assert [(event.step, event.value) for event in events.Scalars('reward')] == [(7, 2)]


def test_declared_offset_must_match_initialization_receipt(tmp_path, monkeypatch):
    module = load_exporter()
    metrics = tmp_path / 'metrics.jsonl'
    metrics.write_text('{"step":10,"reward":2}\n')
    receipt = tmp_path / 'initialization.json'
    receipt.write_text('{"lifetime_transition_offset":20}')
    source = dict(name='student', path=str(metrics), step='step', step_offset=10,
                  step_offset_receipt=dict(path=str(receipt), field='lifetime_transition_offset'))
    with pytest.raises(ValueError, match='step_offset'):
        invoke_exporter(module, tmp_path, monkeypatch, [source])
    assert not list((tmp_path / 'output').rglob('events.out.tfevents.*'))


def test_watch_defers_source_until_offset_receipt_exists(tmp_path, monkeypatch):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    module = load_exporter()
    metrics = tmp_path / 'metrics.jsonl'
    metrics.write_text('{"step":10,"reward":2}\n')
    receipt = tmp_path / 'initialization.json'
    status = tmp_path / 'status.json'
    status.write_text('{"status":"completed"}')
    source = dict(name='student', path=str(metrics), step='step', step_offset=20,
                  completion_status=str(status),
                  step_offset_receipt=dict(path=str(receipt), field='lifetime_transition_offset'))
    polls = []
    def sleep(_):
        polls.append(1)
        if len(polls) == 1:
            assert not list((tmp_path / 'output').rglob('events.out.tfevents.*'))
            receipt.write_text('{"lifetime_transition_offset":20}')
        else:
            raise KeyboardInterrupt
    monkeypatch.setattr(module.time, 'sleep', sleep)
    with pytest.raises(KeyboardInterrupt):
        invoke_exporter(module, tmp_path, monkeypatch, [source], watch=True)
    events = EventAccumulator(str(tmp_path / 'output' / 'student')).Reload()
    assert [(event.step, event.value) for event in events.Scalars('reward')] == [(30, 2)]


@pytest.mark.parametrize('receipt_content', [None, '{"lifetime_transition_offset":true}',
                                           '{"lifetime_transition_offset":20.0}'])
def test_offset_receipt_missing_or_noninteger_rejected(tmp_path, monkeypatch, receipt_content):
    module = load_exporter()
    metrics = tmp_path / 'metrics.jsonl'
    metrics.write_text('{"step":10,"reward":2}\n')
    receipt = tmp_path / 'initialization.json'
    if receipt_content is not None:
        receipt.write_text(receipt_content)
    source = dict(name='student', path=str(metrics), step='step', step_offset=20,
                  step_offset_receipt=dict(path=str(receipt), field='lifetime_transition_offset'))
    with pytest.raises((FileNotFoundError, ValueError)):
        invoke_exporter(module, tmp_path, monkeypatch, [source])
    assert not list((tmp_path / 'output').rglob('events.out.tfevents.*'))
