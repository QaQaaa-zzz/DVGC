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
